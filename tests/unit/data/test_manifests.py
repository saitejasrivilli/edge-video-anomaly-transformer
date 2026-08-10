from evat.data.manifests import read_manifest, write_manifest
from evat.data.schemas import SampleRecord


def _record(sample_id: str = "cat/train/good/000") -> SampleRecord:
    return SampleRecord(
        dataset="mvtec_ad",
        version="test-version",
        split="train",
        sample_id=sample_id,
        category="cat",
        label="good",
        image_path="cat/train/good/000.png",
        annotation_path=None,
    )


def test_write_then_read_manifest_round_trips(tmp_path):
    output = tmp_path / "manifest.jsonl"
    records = [_record("a"), _record("b")]

    written = write_manifest(records, output)
    assert written == 2

    read_back = list(read_manifest(output))
    assert [r.sample_id for r in read_back] == ["a", "b"]
    assert read_back[0] == records[0]


def test_manifest_generation_is_deterministic(tmp_path):
    records = [_record("a"), _record("b")]
    out1 = tmp_path / "m1.jsonl"
    out2 = tmp_path / "m2.jsonl"

    write_manifest(records, out1)
    write_manifest(records, out2)

    assert out1.read_text() == out2.read_text()
