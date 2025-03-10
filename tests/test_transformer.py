from transformer_from_scratch.transformer import SimpleTransformer, get_vocab_size
from pathlib import Path
from transformer_from_scratch.transformer import tokenize, detokenize


def test_tokenize_detokenize():
    text = "Hello, world!"
    tokens = tokenize(text)
    assert len(tokens) > 0
    detokenized = detokenize(tokens)
    assert detokenized == text


def test_detokenize_tokenize():
    tokens = [1, 2, 10_000]
    text = detokenize(tokens)
    assert tokenize(text) == tokens


def test_transformer():
    max_length = 128
    model = SimpleTransformer(vocab_size=get_vocab_size(), embed_dim=128, num_heads=16, num_layers=10, max_length=max_length)
    assert model is not None

    text = "Hello, world!"
    max_generated_tokens = 40
    generated_text = model.generate(text, max_tokens=max_generated_tokens)
    assert len(generated_text) > 0
    assert generated_text.startswith(text)
    generated_tokens = tokenize(generated_text)
    assert len(generated_tokens) == max_generated_tokens + len(tokenize(text))

