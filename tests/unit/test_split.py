from torch_dataloader_utils.split import Split


def test_split_fields():
    s = Split(id=0, files=["f1.parquet", "f2.parquet"])
    assert s.id == 0
    assert s.files == ["f1.parquet", "f2.parquet"]
    assert s.row_count is None
    assert s.size_bytes is None
