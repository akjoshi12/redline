from redline.config import Config


def test_defaults():
    c = Config()
    assert c.concurrency_cap == 4
    assert c.pop_size == 2
    assert c.gen_count == 1
    assert c.sweep_concurrency_start == 1
    assert c.sweep_concurrency_end == 4
    assert c.sweep_context_start == 512
    assert c.sweep_context_end == 2048
    assert c.sweep_context_step == 512
    assert c.soak_seconds == 30
    assert c.base_url == "http://localhost:1234"
    assert c.poll_interval == 2.0


def test_override():
    c = Config(concurrency_cap=10, base_url="http://remote:5678")
    assert c.concurrency_cap == 10
    assert c.base_url == "http://remote:5678"
    assert c.pop_size == 2
