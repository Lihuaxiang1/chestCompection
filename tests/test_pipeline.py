import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import NUM_CLASSES
from dataset import (
    ResizeWithPad,
    StudyImageDataset,
    _split_studies_by_subject,
)
from metrics import aggregate_study_logits
from predict import find_test_studies
from train import ModelEMA


class PipelineTests(unittest.TestCase):
    def test_resize_with_pad_preserves_full_aspect_ratio(self):
        image = Image.fromarray(np.full((100, 200), 255, dtype=np.uint8))
        output = ResizeWithPad(64)(image)
        self.assertEqual(output.size, (64, 64))
        nonzero = np.argwhere(np.asarray(output) > 0)
        height, width = np.ptp(nonzero, axis=0) + 1
        self.assertEqual((height, width), (32, 64))

    def test_study_logits_are_averaged_before_sigmoid(self):
        logits = np.array([[0.0, 2.0], [2.0, 0.0], [-2.0, 1.0]], dtype=np.float32)
        targets = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        probs, study_targets, keys = aggregate_study_logits(
            logits, targets, ["a", "a", "b"],
        )
        self.assertEqual(keys, ["a", "b"])
        np.testing.assert_allclose(probs[0], 1.0 / (1.0 + np.exp(-1.0)))
        np.testing.assert_array_equal(study_targets, [[1, 0], [0, 1]])

    def test_multiview_study_weights_sum_to_one(self):
        row = {"Subject_id": "p1", "Study_id": "s1"}
        row.update({class_id: float(class_id == 0) for class_id in range(NUM_CLASSES)})
        labels = pd.DataFrame([row])
        study_map = {("p1", "s1"): [Path("one.png"), Path("two.png")]}
        dataset = StudyImageDataset(labels, study_map, transform=lambda image: image)
        weights = [record[3] for record in dataset.records]
        self.assertEqual(weights, [0.5, 0.5])
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_patient_split_is_disjoint_and_label_aware(self):
        rows = []
        for index in range(20):
            row = {"Subject_id": f"p{index:02d}", "Study_id": f"s{index:02d}"}
            row.update({class_id: 0.0 for class_id in range(NUM_CLASSES)})
            row[0] = 1.0
            if index < 4:
                row[1] = 1.0
            if 4 <= index < 8:
                row[2] = 1.0
            rows.append(row)
        labels = pd.DataFrame(rows)
        train, val = _split_studies_by_subject(labels, seed=42)
        self.assertFalse(set(train["Subject_id"]) & set(val["Subject_id"]))
        self.assertEqual(val["Subject_id"].nunique(), 2)
        self.assertGreater(val[1].sum(), 0)
        self.assertGreater(val[2].sum(), 0)

    def test_ema_warmup_tracks_early_updates(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        ema = ModelEMA(model, decay=0.999)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)
        self.assertGreater(float(ema.module.weight.item()), 0.5)
        self.assertEqual(ema.updates, 1)

    def test_mixed_test_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.fromarray(np.zeros((8, 8), dtype=np.uint8))
            image.save(root / "flat.png")
            nested = root / "subject" / "study"
            nested.mkdir(parents=True)
            image.save(nested / "nested.png")
            with self.assertRaisesRegex(ValueError, "mixes flat images"):
                find_test_studies(root)

    def test_missing_test_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "does-not-exist"
            self.assertEqual(find_test_studies(missing), [])


if __name__ == "__main__":
    unittest.main()
