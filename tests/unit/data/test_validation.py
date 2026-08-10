from evat.data.schemas import SampleRecord
from evat.data.validation import validate_manifest


def _record(**overrides) -> SampleRecord:
    base = dict(
        dataset="mvtec_ad",
        version="v1",
        split="train",
        sample_id="cat/train/good/000",
        category="cat",
        label="good",
        image_path="cat/train/good/000.png",
        annotation_path=None,
    )
    base.update(overrides)
    return SampleRecord(**base)


def test_valid_manifest_has_no_issues(tmp_path):
    image = tmp_path / "cat" / "train" / "good" / "000.png"
    image.parent.mkdir(parents=True)
    image.touch()

    report = validate_manifest([_record()], dataset_root=tmp_path)

    assert report.is_valid
    assert report.total_records == 1


def test_detects_duplicate_sample_ids():
    records = [_record(sample_id="dup"), _record(sample_id="dup")]

    report = validate_manifest(records, check_files_exist=False)

    assert not report.is_valid
    assert any("duplicate sample_id" in str(issue) for issue in report.issues)


def test_detects_invalid_split():
    report = validate_manifest([_record(split="bogus")], check_files_exist=False)

    assert not report.is_valid
    assert any("invalid split" in str(issue) for issue in report.issues)


def test_detects_missing_image_file(tmp_path):
    report = validate_manifest([_record()], dataset_root=tmp_path)

    assert not report.is_valid
    assert any("image_path does not exist" in str(issue) for issue in report.issues)


def test_detects_missing_annotation_reference(tmp_path):
    image = tmp_path / "cat" / "test" / "defect" / "000.png"
    image.parent.mkdir(parents=True)
    image.touch()

    record = _record(
        split="test",
        label="defect",
        image_path="cat/test/defect/000.png",
        annotation_path="cat/ground_truth/defect/000_mask.png",
    )

    report = validate_manifest([record], dataset_root=tmp_path)

    assert not report.is_valid
    assert any("annotation_path does not exist" in str(issue) for issue in report.issues)


def test_detects_empty_sample_id():
    report = validate_manifest([_record(sample_id="")], check_files_exist=False)

    assert not report.is_valid
    assert any("sample_id is empty" in str(issue) for issue in report.issues)
