from transformer_from_scratch.tokenizer import SimpleTokenizer
from transformer_from_scratch.transformer import (
    SimpleTransformer,
    get_shakespeare_text,
    simple_training_loop,
)
from sortedcontainers import SortedSet
import torch

unk_token_str = "<UNK>"


def test_tokenizer():
    vocab = ["alpha", "beta", "delta", "epsilon"]
    unk_token = len(vocab)
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.encode(f"{vocab[0]} {vocab[2]} unknown") == [0, 2, unk_token]
    assert tokenizer.decode([0, 2, 4]) == f"{vocab[0]} {vocab[2]} {unk_token_str}"
    assert tokenizer.n_vocab == len(vocab) + 1
    assert len(tokenizer.token_to_id) == len(vocab) + 1
    assert len(tokenizer.id_to_token) == len(vocab) + 1


def test_tokenizer_on_shakespeare():
    vocab = get_shakespeare_text().split()
    unk_token = len(set(vocab))
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.n_vocab == len(set(vocab)) + 1
    for shakespeare_word in ["king", "sir", "thou"]:
        assert unk_token not in tokenizer.encode(shakespeare_word)
    for unknown_word in ["internet", "pizza", "hamburger"]:
        assert tokenizer.encode(unknown_word) == [unk_token]
        assert tokenizer.decode([unk_token]) == unk_token_str


def test_tokenizer_on_simple_transformer():
    vocab = ["alpha", "beta", "delta", "epsilon"]
    tokenizer = SimpleTokenizer(vocab)
    # vocab_size, embed_dim, num_heads, num_layers,
    transformer = SimpleTransformer(
        tokenizer=tokenizer,
        embed_dim=4,
        num_heads=1,
        num_layers=1,
    )

    text = "alpha epsilon epsilon beta alpha beta delta epsilon"
    data = torch.tensor([tokenizer.encode(text)])
    simple_training_loop(transformer, data, num_epochs=1, batch_size=1)
    assert "epsilon" in transformer.generate(text, max_tokens=10)
