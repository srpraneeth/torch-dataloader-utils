from torch_dataloader_utils.split import Split, generate_splits


FILES = [f"f{i}.parquet" for i in range(9)]  # f0.parquet … f8.parquet


# --- Split dataclass ---

def test_split_defaults():
    s = Split(id=0)
    assert s.files == []
    assert s.row_count is None
    assert s.size_bytes is None


def test_split_with_files():
    s = Split(id=1, files=["a.parquet", "b.parquet"])
    assert s.id == 1
    assert s.files == ["a.parquet", "b.parquet"]


# --- generate_splits: distribution ---

def test_even_distribution():
    splits = generate_splits(["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"], num_workers=4)
    assert all(len(s.files) == 2 for s in splits)


def test_uneven_distribution():
    splits = generate_splits(FILES, num_workers=4)  # 9 files, 4 workers
    sizes = [len(s.files) for s in splits]
    assert max(sizes) - min(sizes) <= 1


def test_no_file_in_two_splits():
    splits = generate_splits(FILES, num_workers=4)
    all_files = [f for s in splits for f in s.files]
    assert sorted(all_files) == sorted(FILES)


def test_correct_number_of_splits():
    splits = generate_splits(FILES, num_workers=4)
    assert len(splits) == 4
    assert [s.id for s in splits] == [0, 1, 2, 3]


def test_single_worker():
    splits = generate_splits(FILES, num_workers=1)
    assert len(splits) == 1
    assert splits[0].files == FILES


def test_empty_files():
    splits = generate_splits([], num_workers=4)
    assert all(s.files == [] for s in splits)


# --- generate_splits: shuffle ---

def test_no_shuffle_preserves_order():
    splits = generate_splits(FILES, num_workers=1, shuffle=False)
    assert splits[0].files == FILES


def test_shuffle_reproducible_same_epoch():
    s1 = generate_splits(FILES, num_workers=4, shuffle=True, seed=42, epoch=0)
    s2 = generate_splits(FILES, num_workers=4, shuffle=True, seed=42, epoch=0)
    assert [s.files for s in s1] == [s.files for s in s2]


def test_shuffle_differs_across_epochs():
    s0 = generate_splits(FILES, num_workers=1, shuffle=True, seed=42, epoch=0)
    s1 = generate_splits(FILES, num_workers=1, shuffle=True, seed=42, epoch=1)
    assert s0[0].files != s1[0].files


def test_shuffle_does_not_mutate_input():
    original = FILES.copy()
    generate_splits(FILES, num_workers=4, shuffle=True, seed=42)
    assert FILES == original
