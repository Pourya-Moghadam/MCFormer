"""Machine-readable and publication-ready artifact writers for E02--E05."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from mcformer.reproducibility import write_json_atomic


class RawLatex(str):
    """Explicitly marked LaTeX fragment that must not be escaped by the row writer."""


def write_csv(path: str | Path, header: tuple[str, ...], rows: Sequence[Sequence[object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def write_latex_rows(path: str | Path, rows: list[tuple[object, ...]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        " & ".join(
            str(value) if isinstance(value, RawLatex) else latex_escape(str(value)) for value in row
        )
        + r" \\"
        for row in rows
    )
    destination.write_text(text + "\n", encoding="utf-8")


def write_matrix(path: str | Path, matrix: tuple[tuple[float, ...], ...]) -> None:
    write_csv(path, tuple(f"predicted_{index}" for index in range(len(matrix))), list(matrix))


def write_analysis_provenance(path: str | Path, value: dict[str, object]) -> None:
    write_json_atomic(path, value)


def write_json_lines(path: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
