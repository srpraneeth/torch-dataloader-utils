from benchmarks.scenarios import (
    s1_throughput,
    s2_unequal,
    s3_single_large,
    s4_rank_sharding,
    s5_column_projection,
    s6_predicate_pushdown,
    s7_startup_latency,
    s8_format_comparison,
)

# Maps scenario ID → (module, uses_root_dir)
# uses_root_dir=True  → scenario receives the parent data dir (contains dataset subdirs)
# uses_root_dir=False → scenario receives the single-dataset subdirectory directly
ALL_SCENARIOS: dict[str, tuple] = {
    "S1": (s1_throughput,         False),
    "S2": (s2_unequal,            False),
    "S3": (s3_single_large,       False),
    "S4": (s4_rank_sharding,      False),
    "S5": (s5_column_projection,  False),
    "S6": (s6_predicate_pushdown, True),
    "S7": (s7_startup_latency,    True),
    "S8": (s8_format_comparison,  True),
}
