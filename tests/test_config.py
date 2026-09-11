from ragforge.config import Settings


def test_defaults_are_sane():
    s = Settings(_env_file=None)
    assert s.top_k_rerank <= s.top_k_dense
    assert 0.0 < s.grounding_threshold < 1.0
    assert s.embed_dim == 384  # must match VECTOR(384) in docker/init.sql


def test_backend_is_one_of_the_two_supported():
    assert Settings(_env_file=None).llm_backend in {"ollama", "anthropic"}
