import pytest

from app.llm import factory


def test_legacy_openai_configuration(monkeypatch):
    for name in ('CHAT_PROVIDER', 'CHAT_MODEL', 'CHAT_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_MODEL', 'legacy-model')
    monkeypatch.setattr(factory, 'ChatOpenAI', lambda **kwargs: kwargs)
    model = factory.get_chat_model()
    assert model['model'] == 'legacy-model'
    assert model['api_key'] == 'test-key'


def test_chat_and_embeddings_use_independent_endpoints(monkeypatch):
    for prefix in ('CHAT', 'EMBEDDING'):
        monkeypatch.setenv(f'{prefix}_PROVIDER', 'openai_compatible')
        monkeypatch.setenv(f'{prefix}_MODEL', prefix.lower())
        monkeypatch.setenv(f'{prefix}_API_KEY', prefix.lower() + '-key')
        monkeypatch.setenv(f'{prefix}_BASE_URL', 'https://' + prefix.lower() + '.example/v1')
    monkeypatch.setattr(factory, 'ChatOpenAI', lambda **kwargs: kwargs)
    monkeypatch.setattr(factory, 'OpenAIEmbeddings', lambda **kwargs: kwargs)
    assert factory.get_chat_model()['api_key'] == 'chat-key'
    embeddings = factory.get_embeddings()
    assert embeddings['api_key'] == 'embedding-key'
    assert embeddings['check_embedding_ctx_length'] is False
    assert factory.embedding_identity().startswith('openai_compatible:')


def test_invalid_provider_and_missing_endpoint_fail_early(monkeypatch):
    monkeypatch.setenv('CHAT_PROVIDER', 'unknown')
    with pytest.raises(ValueError, match='Unsupported'):
        factory.get_chat_model()
    monkeypatch.setenv('CHAT_PROVIDER', 'openai_compatible')
    monkeypatch.setenv('CHAT_API_KEY', 'test')
    monkeypatch.delenv('CHAT_BASE_URL', raising=False)
    with pytest.raises(ValueError, match='CHAT_BASE_URL'):
        factory.get_chat_model()


def test_embedding_output_dimensions_are_sent_when_configured(monkeypatch):
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'openai')
    monkeypatch.setenv('EMBEDDING_API_KEY', 'test')
    monkeypatch.setenv('EMBEDDING_OUTPUT_DIMENSIONS', '1536')
    monkeypatch.setattr(factory, 'OpenAIEmbeddings', lambda **kwargs: kwargs)
    assert factory.get_embeddings()['dimensions'] == 1536
    monkeypatch.setenv('EMBEDDING_OUTPUT_DIMENSIONS', '0')
    with pytest.raises(ValueError, match='positive'):
        factory.get_embeddings()
