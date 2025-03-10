from transformer_from_scratch.transformer import train
from transformer_from_scratch.transformer import SimpleTransformer
from transformer_from_scratch.transformer import get_vocab_size


def test_training():
    model = SimpleTransformer(
        vocab_size=get_vocab_size(),
        embed_dim=32,
        num_heads=16,
        num_layers=10,
        max_length=128,
    )
    assert model is not None
    # before training, the model should not generate the correct text
    assert not model.generate("Hello", max_tokens=10).startswith("Hello, world!")

    data = ["Hello, world!"] * 100
    train(model, data, epochs=4, lr=0.01)
    assert model.generate("Hello", max_tokens=10).startswith("Hello, world!")