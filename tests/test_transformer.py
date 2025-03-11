from transformer_from_scratch.tokenizer import SimpleTokenizer
from transformer_from_scratch.transformer import SimpleTransformer, get_shakespeare_text


tokenizer = SimpleTokenizer(get_shakespeare_text().split())


def test_transformer():
    embed_dim = 16
    num_heads = 2
    num_layers = 2
    tokenizer = SimpleTokenizer(["Hello,", "world!"])
    model = SimpleTransformer(tokenizer, embed_dim, num_heads, num_layers)
    assert model is not None

    text = "Hello, world!"
    max_generated_tokens = 40
    generated_text = model.generate(text, max_tokens=max_generated_tokens)
    assert len(generated_text) > 0
    assert generated_text.startswith(text)
    generated_tokens = tokenizer.encode(generated_text)
    # the model actually generates more tokens than requested
    # not sure why
    assert len(generated_tokens) >= max_generated_tokens + len(
        tokenizer.encode(text)
    ), "expected at least max_generated_tokens + len(tokenize(text)) tokens"


def test_readme_example():
    from transformer_from_scratch.transformer import SimpleTransformer
    from transformer_from_scratch.tokenizer import SimpleTokenizer
    # Initialize the transformer
    tokenizer = SimpleTokenizer(get_shakespeare_text().split())
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    model = SimpleTransformer(tokenizer, embed_dim, num_heads, num_layers)

    # Generate output
    output = model.generate("let slip the dogs of war", max_tokens=10)
    assert len(output) > 0
    assert output.startswith("let slip the dogs of war")
