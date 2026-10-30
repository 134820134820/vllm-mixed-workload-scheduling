from bench.monitor import parse


def test_parse_drops_buckets_and_constant_labels():
    text = (
        '# HELP x\nvllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
        'vllm:request_queue_time_seconds_bucket{engine="0",le="0.3",model_name="m"} 7.0\n'
        'vllm:request_queue_time_seconds_sum{engine="0",model_name="m"} 1.5\n'
        'vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 9.0\n'
    )
    assert parse(text) == {
        "vllm:num_requests_running": 3.0,
        "vllm:request_queue_time_seconds_sum": 1.5,
        'vllm:request_success_total{finished_reason="length"}': 9.0,
    }
