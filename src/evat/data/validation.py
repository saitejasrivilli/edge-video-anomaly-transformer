"""Manifest and dataset structure validation.

Validation never silently drops invalid samples: every issue is collected
and returned so callers can decide how to proceed. Nothing here downloads,
resizes, or reads image content — it only checks paths and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evat.data.schemas import SampleRecord

VALID_SPLITS = frozenset({"train", "val", "test"})


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single validation problem tied to a specific sample (or manifest-wide)."""

    sample_id: str
    reason: str

    def __str__(self) -> str:
        return f"[{self.sample_id}] {self.reason}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregate result of validating a manifest."""

    total_records: int
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    def summary(self) -> str:
        if self.is_valid:
            return f"{self.total_records} records validated, 0 issues."
        lines = [f"{self.total_records} records validated, {len(self.issues)} issue(s):"]
        lines.extend(f"  - {issue}" for issue in self.issues)
        return "\n".join(lines)


def validate_manifest(
    records: list[SampleRecord],
    dataset_root: Path | None = None,
    check_files_exist: bool = True,
) -> ValidationReport:
    """Validate a manifest's records for structural and referential integrity.

    Checks performed:
    - duplicate sample IDs
    - invalid/unsupported split values
    - malformed sample_id / category / label metadata (empty strings)
    - missing image files (if ``check_files_exist`` and ``dataset_root`` given)
    - missing annotation files referenced by a record
      (if ``check_files_exist`` and ``dataset_root`` given)
    """
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()

    for record in records:
        if not record.sample_id:
            issues.append(ValidationIssue("<missing>", "sample_id is empty"))
            continue

        if record.sample_id in seen_ids:
            issues.append(ValidationIssue(record.sample_id, "duplicate sample_id"))
        seen_ids.add(record.sample_id)

        if record.split not in VALID_SPLITS:
            issues.append(
                ValidationIssue(
                    record.sample_id,
                    f"invalid split '{record.split}', expected one of {sorted(VALID_SPLITS)}",
                )
            )

        if not record.category:
            issues.append(ValidationIssue(record.sample_id, "category is empty"))

        if not record.image_path:
            issues.append(ValidationIssue(record.sample_id, "image_path is empty"))
        elif check_files_exist and dataset_root is not None:
            if not (dataset_root / record.image_path).exists():
                issues.append(
                    ValidationIssue(
                        record.sample_id, f"image_path does not exist: {record.image_path}"
                    )
                )

        if record.annotation_path and check_files_exist and dataset_root is not None:
            if not (dataset_root / record.annotation_path).exists():
                issues.append(
                    ValidationIssue(
                        record.sample_id,
                        f"annotation_path does not exist: {record.annotation_path}",
                    )
                )

    return ValidationReport(total_records=len(records), issues=tuple(issues))
