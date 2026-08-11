"""Official NTU protocol generation and validated explicit split manifests."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mcformer.data.manifest import Manifest, SampleRecord

NTU60_CS_TRAIN_SUBJECTS = frozenset(
    {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
)
NTU60_CV_TRAIN_CAMERAS = frozenset({2, 3})
NTU120_CS_TRAIN_SUBJECTS = frozenset(
    {
        1,
        2,
        4,
        5,
        8,
        9,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        25,
        27,
        28,
        31,
        34,
        35,
        38,
        45,
        46,
        47,
        49,
        50,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        70,
        74,
        78,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        89,
        91,
        92,
        93,
        94,
        95,
        97,
        98,
        100,
        103,
    }
)


class ProtocolError(ValueError):
    """Raised when split membership is incomplete, overlapping, or invalid."""


@dataclass(frozen=True, slots=True)
class ProtocolSplit:
    """Stable IDs for train, validation, and test partitions."""

    dataset: str
    protocol: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    validation_strategy: str
    validation_seed: int = 2026

    def validate(self, manifest: Manifest) -> None:
        """Assert disjointness, completeness, identity, and class coverage."""

        partitions = {
            "train": set(self.train),
            "validation": set(self.validation),
            "test": set(self.test),
        }
        if any(not values for values in partitions.values()):
            empty = [name for name, values in partitions.items() if not values]
            raise ProtocolError(f"Empty partitions: {empty}")
        names = tuple(partitions)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = partitions[left] & partitions[right]
                if overlap:
                    raise ProtocolError(f"{left}/{right} overlap: {sorted(overlap)[:10]}")
        manifest_ids = {record.sample_id for record in manifest}
        split_ids = set().union(*partitions.values())
        if split_ids != manifest_ids:
            missing = sorted(manifest_ids - split_ids)
            unknown = sorted(split_ids - manifest_ids)
            raise ProtocolError(f"Split mismatch; missing={missing[:10]}, unknown={unknown[:10]}")
        all_labels = {record.label_id for record in manifest}
        for name in ("train", "test"):
            labels = {manifest.by_id(sample_id).label_id for sample_id in partitions[name]}
            if labels != all_labels:
                raise ProtocolError(f"{name} lacks labels: {sorted(all_labels - labels)}")

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": self.dataset,
            "protocol": self.protocol,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "validation_strategy": self.validation_strategy,
            "validation_seed": self.validation_seed,
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read_json(cls, path: str | Path, manifest: Manifest) -> ProtocolSplit:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            split = cls(
                dataset=str(raw["dataset"]),
                protocol=str(raw["protocol"]),
                train=tuple(sorted(raw["train"])),
                validation=tuple(sorted(raw["validation"])),
                test=tuple(sorted(raw["test"])),
                validation_strategy=str(raw["validation_strategy"]),
                validation_seed=int(raw.get("validation_seed", 2026)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError(f"Invalid protocol file: {error}") from error
        if split.dataset != manifest[0].dataset:
            raise ProtocolError("Protocol dataset does not match manifest")
        split.validate(manifest)
        return split


def official_ntu_train_test(
    manifest: Manifest, protocol: str
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Apply the published NTU60 or NTU120 CS/CV/CSet rule."""

    dataset = manifest[0].dataset
    normalized = protocol.lower()
    train: list[SampleRecord] = []
    test: list[SampleRecord] = []
    for record in manifest:
        try:
            subject = int(record.subject_id or "")
            camera = int(record.camera_or_view_id or "")
            setup = int(record.setup_id or "")
        except ValueError as error:
            raise ProtocolError(f"Invalid NTU metadata for {record.sample_id}") from error
        if dataset == "ntu_rgbd_60" and normalized == "cs":
            is_train = subject in NTU60_CS_TRAIN_SUBJECTS
        elif dataset == "ntu_rgbd_60" and normalized == "cv":
            is_train = camera in NTU60_CV_TRAIN_CAMERAS
        elif dataset == "ntu_rgbd_120" and normalized == "cs":
            is_train = subject in NTU120_CS_TRAIN_SUBJECTS
        elif dataset == "ntu_rgbd_120" and normalized in {"cset", "xset"}:
            is_train = setup % 2 == 0
        else:
            raise ProtocolError(f"Unsupported dataset/protocol: {dataset}/{protocol}")
        (train if is_train else test).append(record)
    return train, test


def _stratified_clip_validation(
    records: Sequence[SampleRecord], fraction: float, seed: int
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    grouped: dict[int, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label_id].append(record)
    rng = random.Random(seed)
    train: list[SampleRecord] = []
    validation: list[SampleRecord] = []
    for label_records in grouped.values():
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        count = min(max(round(len(shuffled) * fraction), 1), max(len(shuffled) - 1, 0))
        validation.extend(shuffled[:count])
        train.extend(shuffled[count:])
    return train, validation


def _group_validation(
    records: Sequence[SampleRecord],
    *,
    group_field: str,
    fraction: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]] | None:
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        value = getattr(record, group_field)
        if value is None:
            return None
        groups[value].append(record)
    if len(groups) < 5:
        return None
    all_labels = {record.label_id for record in records}
    target = max(1, round(len(records) * fraction))
    candidates = sorted(groups)
    random.Random(seed).shuffle(candidates)
    selected: list[str] = []
    selected_count = 0
    validation_labels: set[int] = set()
    while validation_labels != all_labels:
        choices: list[tuple[int, int, int, str]] = []
        for priority, group in enumerate(candidates):
            if group in selected:
                continue
            proposed = set((*selected, group))
            remaining = [
                record for record in records if getattr(record, group_field) not in proposed
            ]
            if {record.label_id for record in remaining} != all_labels:
                continue
            group_labels = {record.label_id for record in groups[group]}
            new_labels = len(group_labels - validation_labels)
            choices.append(
                (
                    -new_labels,
                    abs(selected_count + len(groups[group]) - target),
                    priority,
                    group,
                )
            )
        if not choices or -min(choices)[0] == 0:
            return None
        group = min(choices)[3]
        selected.append(group)
        selected_count += len(groups[group])
        validation_labels.update(record.label_id for record in groups[group])

    for group in candidates:
        if group in selected:
            continue
        proposed_error = abs(selected_count + len(groups[group]) - target)
        if proposed_error >= abs(selected_count - target):
            continue
        proposed = set((*selected, group))
        remaining_labels = {
            record.label_id for record in records if getattr(record, group_field) not in proposed
        }
        if remaining_labels == all_labels:
            selected.append(group)
            selected_count += len(groups[group])
    if not selected:
        return None
    selected_set = set(selected)
    train = [record for record in records if getattr(record, group_field) not in selected_set]
    validation = [record for record in records if getattr(record, group_field) in selected_set]
    if {record.label_id for record in validation} != all_labels:
        return None
    return train, validation


def build_protocol_split(
    manifest: Manifest,
    *,
    protocol: str,
    validation_fraction: float = 0.10,
    validation_seed: int = 2026,
    explicit_train_test: Mapping[str, Iterable[str]] | None = None,
) -> ProtocolSplit:
    """Create an official test split and the fixed release validation split."""

    if not 0 < validation_fraction < 1:
        raise ProtocolError("validation_fraction must lie strictly between zero and one")
    dataset = manifest[0].dataset
    if explicit_train_test is None:
        official_train, test = official_ntu_train_test(manifest, protocol)
    else:
        try:
            train_ids = set(explicit_train_test["train"])
            test_ids = set(explicit_train_test["test"])
        except KeyError as error:
            raise ProtocolError("Explicit protocol requires train and test ID lists") from error
        if train_ids & test_ids:
            raise ProtocolError("Explicit train and test lists overlap")
        official_train = [record for record in manifest if record.sample_id in train_ids]
        test = [record for record in manifest if record.sample_id in test_ids]
        selected_ids = {record.sample_id for record in (*official_train, *test)}
        if selected_ids != {record.sample_id for record in manifest}:
            raise ProtocolError("Explicit train/test lists must cover the complete manifest")

    protocol_name = protocol.lower()
    group_field = (
        "subject_id"
        if protocol_name == "cs"
        else "camera_or_view_id"
        if protocol_name in {"cv", "cv1", "cv2"}
        else "setup_id"
    )
    grouped = _group_validation(
        official_train,
        group_field=group_field,
        fraction=validation_fraction,
        seed=validation_seed,
    )
    if grouped is None:
        train, validation = _stratified_clip_validation(
            official_train, validation_fraction, validation_seed
        )
        strategy = "class_stratified_clip_fallback"
    else:
        train, validation = grouped
        strategy = f"group_aware:{group_field}"
    split = ProtocolSplit(
        dataset=dataset,
        protocol=protocol_name,
        train=tuple(sorted(record.sample_id for record in train)),
        validation=tuple(sorted(record.sample_id for record in validation)),
        test=tuple(sorted(record.sample_id for record in test)),
        validation_strategy=strategy,
        validation_seed=validation_seed,
    )
    split.validate(manifest)
    return split


def read_explicit_train_test(path: str | Path) -> dict[str, list[str]]:
    """Read official train/test IDs from a small JSON protocol source."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(key in raw for key in ("train", "test")):
        raise ProtocolError("Protocol source must contain train and test lists")
    return {"train": list(raw["train"]), "test": list(raw["test"])}
