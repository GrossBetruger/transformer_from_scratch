from transformer_from_scratch.tokenizer import SimpleTokenizer
from transformer_from_scratch.transformer import simple_training_loop
from transformer_from_scratch.transformer import SimpleTransformer

import torch


def test_training():
    tokenizer = SimpleTokenizer(vocab=["hello", "world"])
    model = SimpleTransformer(
        tokenizer=tokenizer,
        embed_dim=24,
        num_heads=4,
        num_layers=2,
        max_length=128,
    )
    assert model is not None
    print(tokenizer.token_to_id, tokenizer.id_to_token)
    # before training, the model should not generate the correct text
    assert not model.generate("Hello,", max_tokens=10).startswith("hello world")

    data = ["hello world"] * 100
    data = torch.tensor([tokenizer.encode(x) for x in data])
    simple_training_loop(model, data, num_epochs=10, batch_size=1)
    assert model.generate("hello", max_tokens=3).startswith("hello world")
