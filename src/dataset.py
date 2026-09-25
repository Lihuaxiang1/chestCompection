"""Data loading for flat smoke-test data and official multi-view Study data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from config import (
    BATCH_SIZE, DISEASE_LABELS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    NUM_CLASSES, NUM_WORKERS, SEED, TEST_IMG_DIR, TRAIN_CSV, TRAIN_IMG_DIR,
    VAL_CSV, VAL_IMG_DIR,
)

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


class ResizeWithPad:
    """Resize a PIL image without cropping, then center-pad it to a square."""

    def __init__(self, image_size, content_scale=1.0):
        if not 0 < content_scale <= 1.0:
            raise ValueError("content_scale must be in (0, 1]")
        self.image_size = int(image_size)
        self.content_size = max(1, int(round(self.image_size * content_scale)))

    def __call__(self, image):
        contained = ImageOps.contain(
            image,
            (self.content_size, self.content_size),
            method=Image.Resampling.BILINEAR,
        )
        width, height = contained.size
        pad_width = self.image_size - width
        pad_height = self.image_size - height
        border = (
            pad_width // 2,
            pad_height // 2,
            pad_width - pad_width // 2,
            pad_height - pad_height // 2,
        )
        return ImageOps.expand(contained, border=border, fill=0)


def get_train_transforms(image_size=IMAGE_SIZE):
    """Conservative augmentation that keeps the full chest in frame."""
    return transforms.Compose([
        ResizeWithPad(image_size, content_scale=0.96),
        transforms.RandomAffine(
            degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02), fill=0,
        ),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size=IMAGE_SIZE):
    """Deterministic validation and test transform."""
    return transforms.Compose([
        ResizeWithPad(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _validate_binary_targets(values, source):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains missing or non-numeric labels")
    invalid = np.setdiff1d(np.unique(values), [0.0, 1.0])
    if len(invalid):
        raise ValueError(f"{source} contains labels outside 0/1: {invalid.tolist()}")


class SimpleDataset(Dataset):
    """One image per row, used by the local synthetic smoke-test data."""

    def __init__(self, csv_path, img_dir, transform, has_labels=True, dataframe=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.has_labels = has_labels
        if dataframe is not None:
            self.df = dataframe.reset_index(drop=True).copy()
        elif csv_path is not None and Path(csv_path).exists():
            self.df = pd.read_csv(csv_path)
        else:
            self.df = pd.DataFrame()

        self.label_cols = [name for name in DISEASE_LABELS if name in self.df.columns]
        if has_labels and len(self.label_cols) != NUM_CLASSES:
            missing = sorted(set(DISEASE_LABELS) - set(self.label_cols))
            raise ValueError(f"Missing label columns in {csv_path}: {missing}")

        self.image_col = None
        if len(self.df):
            for column in ["Image Index", "image_id", "filename", "image_path", "Path"]:
                if column in self.df.columns:
                    self.image_col = column
                    break
            if self.image_col is None:
                self.image_col = self.df.columns[0]

        if self.has_labels and len(self.df):
            _validate_binary_targets(self.df[self.label_cols].values, str(csv_path))
            disease_sum = self.df[DISEASE_LABELS[:-1]].sum(axis=1)
            inconsistent = ((disease_sum > 0) & (self.df[DISEASE_LABELS[-1]] == 1)).sum()
            if inconsistent:
                print(f"[WARN] {inconsistent} rows mark No Finding together with disease labels")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = str(row[self.image_col])
        image_path = self.img_dir / image_name
        if not image_path.exists():
            image_path = self.img_dir / Path(image_name).name
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as opened:
            image = self.transform(opened.convert("RGB"))
        target = (
            torch.tensor(row[self.label_cols].values.astype(np.float32))
            if self.has_labels else torch.zeros(NUM_CLASSES, dtype=torch.float32)
        )
        return {
            "image": image,
            "target": target,
            "study_key": image_name,
            "sample_weight": torch.tensor(1.0, dtype=torch.float32),
        }

    def get_image_ids(self):
        return self.df[self.image_col].astype(str).tolist() if self.image_col else []


def load_labels_long_format(csv_path):
    """Convert positive long-format rows to one binary vector per Subject/Study."""
    df = pd.read_csv(csv_path, dtype={"Subject_id": str, "Study_id": str})
    required_ids = {"Subject_id", "Study_id"}
    if not required_ids.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain Subject_id and Study_id")
    df["Subject_id"] = df["Subject_id"].astype(str)
    df["Study_id"] = df["Study_id"].astype(str)

    if "Predict_class" not in df.columns:
        numeric_columns = [column for column in range(NUM_CLASSES) if column in df.columns]
        string_columns = [str(column) for column in range(NUM_CLASSES) if str(column) in df.columns]
        columns = numeric_columns if len(numeric_columns) == NUM_CLASSES else string_columns
        if len(columns) != NUM_CLASSES:
            raise ValueError(f"{csv_path} has no Predict_class or complete 0..{NUM_CLASSES - 1} columns")
        output = df[["Subject_id", "Study_id"] + columns].copy()
        output.columns = ["Subject_id", "Study_id"] + list(range(NUM_CLASSES))
        _validate_binary_targets(output[list(range(NUM_CLASSES))].values, str(csv_path))
        return output.drop_duplicates(["Subject_id", "Study_id"])

    classes = pd.to_numeric(df["Predict_class"], errors="raise").astype(int)
    if not classes.between(0, NUM_CLASSES - 1).all():
        invalid = sorted(classes[~classes.between(0, NUM_CLASSES - 1)].unique().tolist())
        raise ValueError(f"{csv_path} contains invalid Predict_class values: {invalid}")
    work = df[["Subject_id", "Study_id"]].copy()
    work["Predict_class"] = classes
    work["_positive"] = 1
    pivot = work.pivot_table(
        index=["Subject_id", "Study_id"], columns="Predict_class",
        values="_positive", aggfunc="max", fill_value=0,
    )
    for class_id in range(NUM_CLASSES):
        if class_id not in pivot.columns:
            pivot[class_id] = 0
    output = pivot[list(range(NUM_CLASSES))].reset_index()
    no_finding_id = NUM_CLASSES - 1
    disease_ids = list(range(no_finding_id))
    inconsistent = ((output[no_finding_id] == 1) & (output[disease_ids].sum(axis=1) > 0)).sum()
    if inconsistent:
        print(f"[WARN] {inconsistent} studies mark No Finding together with disease classes")
    return output


def find_study_images(img_dir):
    """Map (Subject_id, Study_id) to every image in that study."""
    img_dir = Path(img_dir)
    study_map = {}
    if not img_dir.exists():
        return study_map
    for subject_dir in sorted(path for path in img_dir.iterdir() if path.is_dir()):
        for study_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            images = []
            for extension in IMAGE_EXTENSIONS:
                images.extend(study_dir.glob(extension))
            if images:
                key = (str(subject_dir.name), str(study_dir.name))
                if key in study_map:
                    raise ValueError(f"Duplicate study directory: {key}")
                study_map[key] = sorted(set(images))
    return study_map


class StudyImageDataset(Dataset):
    """One record per image while preserving Study identity for aggregation."""

    def __init__(self, labels_df, study_map, transform, has_labels=True):
        self.labels_df = labels_df.reset_index(drop=True).copy()
        self.study_map = study_map
        self.transform = transform
        self.has_labels = has_labels
        self.records = []
        self.num_studies = 0

        label_lookup = {}
        if has_labels:
            for _, row in self.labels_df.iterrows():
                key = (str(row["Subject_id"]), str(row["Study_id"]))
                label_lookup[key] = row[list(range(NUM_CLASSES))].values.astype(np.float32)
            if label_lookup:
                _validate_binary_targets(np.stack(list(label_lookup.values())), "study labels")
            keys = [key for key in label_lookup if key in study_map]
            missing = len(label_lookup) - len(keys)
            if missing:
                print(f"[WARN] {missing} labeled studies have no image directory")
        else:
            keys = list(study_map.keys())

        self.num_studies = len(keys)
        for key in keys:
            images = study_map[key]
            target = label_lookup.get(key, np.zeros(NUM_CLASSES, dtype=np.float32))
            sample_weight = 1.0 / len(images)
            study_key = f"{key[0]}::{key[1]}"
            for image_path in images:
                self.records.append((image_path, target, study_key, sample_weight))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, target, study_key, sample_weight = self.records[index]
        with Image.open(image_path) as opened:
            image = self.transform(opened.convert("RGB"))
        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.float32),
            "study_key": study_key,
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
        }


def detect_format(csv_path, img_dir):
    csv_path = Path(csv_path)
    img_dir = Path(img_dir)
    if csv_path.exists():
        columns = pd.read_csv(csv_path, nrows=1).columns
        if "Predict_class" in columns or "Study_id" in columns:
            return "study"
        if any(label in columns for label in DISEASE_LABELS):
            return "simple"
    if img_dir.exists() and any(path.is_dir() for path in img_dir.iterdir()):
        return "study"
    return "simple"


def compute_pos_weights_simple(dataset):
    values = dataset.df[DISEASE_LABELS].values.astype(np.float32)
    positives = values.sum(axis=0)
    negatives = len(values) - positives
    ratios = np.ones_like(positives, dtype=np.float32)
    np.divide(negatives, positives, out=ratios, where=positives > 0)
    return torch.tensor(ratios, dtype=torch.float32)


def compute_pos_weights_study(labels_df):
    values = labels_df[list(range(NUM_CLASSES))].values.astype(np.float32)
    positives = values.sum(axis=0)
    negatives = len(values) - positives
    ratios = np.ones_like(positives, dtype=np.float32)
    np.divide(negatives, positives, out=ratios, where=positives > 0)
    return torch.tensor(ratios, dtype=torch.float32)


def _subject_set(csv_path):
    if not Path(csv_path).exists():
        return set()
    frame = pd.read_csv(
        csv_path, usecols=lambda column: column == "Subject_id",
        dtype={"Subject_id": str},
    )
    return set(frame["Subject_id"].astype(str)) if "Subject_id" in frame else set()


def audit_train_val_leakage():
    overlap = _subject_set(TRAIN_CSV) & _subject_set(VAL_CSV)
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(f"Subject leakage between train and val: {preview} ({len(overlap)} total)")


def _split_simple(csv_path, img_dir, seed, image_size):
    frame = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(frame))
    n_val = max(1, int(len(frame) * 0.1))
    return (
        SimpleDataset(None, img_dir, get_train_transforms(image_size), dataframe=frame.iloc[indices[n_val:]]),
        SimpleDataset(None, img_dir, get_val_transforms(image_size), dataframe=frame.iloc[indices[:n_val]]),
    )


def _split_studies_by_subject(labels_df, seed):
    """Create a patient-disjoint split while approximately preserving label rates."""
    subject_targets = labels_df.copy()
    subject_targets["Subject_id"] = subject_targets["Subject_id"].astype(str)
    subject_targets = subject_targets.groupby("Subject_id")[list(range(NUM_CLASSES))].max()
    subjects = subject_targets.index.to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * 0.1))

    values = subject_targets.loc[subjects].to_numpy(dtype=np.float32)
    target_counts = values.sum(axis=0) * (n_val / max(len(subjects), 1))
    selected = []
    selected_mask = np.zeros(len(subjects), dtype=bool)
    current_counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    tie_breakers = rng.random(len(subjects)) * 1e-8

    while len(selected) < n_val:
        deficits = np.clip(target_counts - current_counts, 0.0, None)
        normalizer = np.maximum(target_counts, 1.0)
        gains = (values * (deficits / normalizer)).sum(axis=1)
        overflow = (
            np.clip(current_counts + values - target_counts, 0.0, None)
            / normalizer
        ).sum(axis=1)
        scores = gains - 0.2 * overflow + tie_breakers
        scores[selected_mask] = -np.inf
        chosen = int(np.argmax(scores))
        selected.append(chosen)
        selected_mask[chosen] = True
        current_counts += values[chosen]

    val_subjects = set(subjects[selected])
    mask = labels_df["Subject_id"].astype(str).isin(val_subjects)
    return labels_df[~mask].copy(), labels_df[mask].copy()


def build_dataloaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=SEED, image_size=IMAGE_SIZE,
):
    """Build loaders whose validation outputs can be aggregated at Study level."""
    train_ds = val_ds = test_ds = None
    train_labels_for_weights = None
    train_format = detect_format(TRAIN_CSV, TRAIN_IMG_DIR)
    has_separate_val = VAL_CSV.exists() and VAL_IMG_DIR.exists()

    if TRAIN_CSV.exists() and TRAIN_IMG_DIR.exists():
        if train_format == "study":
            labels = load_labels_long_format(TRAIN_CSV)
            study_map = find_study_images(TRAIN_IMG_DIR)
            if has_separate_val:
                audit_train_val_leakage()
                train_labels_for_weights = labels
                train_ds = StudyImageDataset(labels, study_map, get_train_transforms(image_size))
            else:
                train_labels, val_labels = _split_studies_by_subject(labels, seed)
                train_labels_for_weights = train_labels
                train_ds = StudyImageDataset(train_labels, study_map, get_train_transforms(image_size))
                val_ds = StudyImageDataset(val_labels, study_map, get_val_transforms(image_size))
        elif has_separate_val:
            train_ds = SimpleDataset(TRAIN_CSV, TRAIN_IMG_DIR, get_train_transforms(image_size))
        else:
            train_ds, val_ds = _split_simple(TRAIN_CSV, TRAIN_IMG_DIR, seed, image_size)
        print(f"训练集: {len(train_ds):,} 张图片")
    else:
        print(f"[WARN] Training data not found: {TRAIN_CSV} / {TRAIN_IMG_DIR}")

    if has_separate_val:
        val_format = detect_format(VAL_CSV, VAL_IMG_DIR)
        if val_format == "study":
            labels = load_labels_long_format(VAL_CSV)
            val_ds = StudyImageDataset(labels, find_study_images(VAL_IMG_DIR), get_val_transforms(image_size))
        else:
            val_ds = SimpleDataset(VAL_CSV, VAL_IMG_DIR, get_val_transforms(image_size))
    if val_ds is not None:
        studies = getattr(val_ds, "num_studies", len(val_ds))
        print(f"验证集: {len(val_ds):,} 张图片 / {studies:,} 个 Study")

    if TEST_IMG_DIR.exists():
        test_format = detect_format(TEST_IMG_DIR / "labels-not-required.csv", TEST_IMG_DIR)
        if test_format == "study":
            test_ds = StudyImageDataset(
                pd.DataFrame(), find_study_images(TEST_IMG_DIR),
                get_val_transforms(image_size), has_labels=False,
            )
        else:
            images = []
            for extension in IMAGE_EXTENSIONS:
                images.extend(TEST_IMG_DIR.glob(extension))
            frame = pd.DataFrame({"image_path": [path.name for path in sorted(set(images))]})
            test_ds = SimpleDataset(
                None, TEST_IMG_DIR, get_val_transforms(image_size), has_labels=False, dataframe=frame,
            )
        print(f"测试集: {len(test_ds):,} 张图片")

    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, generator=generator, **loader_options) if train_ds else None
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options) if val_ds else None
    test_loader = DataLoader(test_ds, shuffle=False, **loader_options) if test_ds else None

    if isinstance(train_ds, StudyImageDataset):
        pos_weights = compute_pos_weights_study(train_labels_for_weights)
    elif isinstance(train_ds, SimpleDataset):
        pos_weights = compute_pos_weights_simple(train_ds)
    elif isinstance(train_ds, Subset):
        pos_weights = compute_pos_weights_simple(train_ds.dataset)
    else:
        pos_weights = torch.ones(NUM_CLASSES)
    return train_loader, val_loader, test_loader, pos_weights


if __name__ == "__main__":
    fake = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    print("train transform:", get_train_transforms()(fake).shape)
    print("val transform:", get_val_transforms()(fake).shape)
    train_loader, val_loader, _, weights = build_dataloaders(batch_size=4)
    if train_loader:
        batch = next(iter(train_loader))
        print("images:", batch["image"].shape)
        print("targets:", batch["target"].shape)
        print("study keys:", batch["study_key"][:2])
        print("positive weights:", weights)
