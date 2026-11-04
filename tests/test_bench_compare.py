from scripts.bench_compare import median_run


def test_median_run_picks_the_middle_throughput():
    runs = [
        {"tokens_per_second": 30.0},
        {"tokens_per_second": 50.0},
        {"tokens_per_second": 40.0},
    ]
    assert median_run(runs)["tokens_per_second"] == 40.0
