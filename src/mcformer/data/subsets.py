"""Resolve human-readable diagnostic subsets against a validated manifest."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from mcformer.data.manifest import Manifest, ManifestError


def normalize_label_name(value: str) -> str:
    """Normalize punctuation and case while preserving alphanumeric order."""

    return "".join(re.findall(r"[a-z0-9]+", value.casefold()))


def resolve_label_ids(manifest: Manifest, names: Iterable[str]) -> tuple[int, ...]:
    """Resolve names exactly after normalization; reject missing or ambiguous names."""

    labels: dict[str, list[tuple[int, str]]] = {}
    for record in manifest:
        pair = (record.label_id, record.label_name)
        bucket = labels.setdefault(normalize_label_name(record.label_name), [])
        if pair not in bucket:
            bucket.append(pair)
    resolved: list[int] = []
    for requested in names:
        matches = labels.get(normalize_label_name(requested), [])
        if len(matches) != 1:
            raise ManifestError(
                f"Diagnostic label {requested!r} resolved to {len(matches)} labels: {matches}"
            )
        resolved.append(matches[0][0])
    if len(resolved) != len(set(resolved)):
        raise ManifestError("Diagnostic label list contains duplicates")
    return tuple(resolved)


def load_subset_names(path: str | Path, subset: str) -> tuple[str, ...]:
    """Load one flat label-name list from a diagnostic subset JSON file."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        names = value[subset]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ManifestError(f"Cannot load diagnostic subset {subset!r}: {error}") from error
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ManifestError(f"Diagnostic subset {subset!r} must be a non-empty string list")
    return tuple(names)
