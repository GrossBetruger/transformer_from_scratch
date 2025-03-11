# Transformer From Scratch

A Python implementation of the Transformer architecture from scratch, following the architecture described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). This implementation aims to provide a clear, educational view of the Transformer architecture's core components.

## Features

- Self-attention mechanism implementation
- Multi-head attention
- Position-wise feed-forward networks
- Positional encoding
- Layer normalization
- Full encoder-decoder architecture

## Requirements

- Python 3.10+
- Poetry for dependency management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GrossBetruger/transformer-from-scratch.git
cd transformer-from-scratch
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Project Structure

```
transformer-from-scratch/
├── transformer_from_scratch/
│   └── transformer.py      # Core transformer implementation
├── tests/
│   └── test_transformer.py # Unit tests
├── pyproject.toml         # Project dependencies and metadata
└── README.md             # Project documentation
```

## Usage

Here's a basic example of how to use the transformer:

```python
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
```

## Components

The implementation includes the following key components:

- **MultiHeadAttention**: Allows the model to jointly attend to information from different representation subspaces
- **PositionalEncoding**: Adds information about the relative or absolute position of tokens
- **TransformerEncoder**: Processes the input sequence
- **TransformerDecoder**: Generates the output sequence
- **PositionwiseFeedForward**: Applies two linear transformations with a ReLU activation

## Development

To run tests:
```bash
poetry run pytest
```

To run specific test file:
```bash
poetry run pytest tests/test_transformer.py
```

## References

1. ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) - The original transformer paper
2. [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Helpful tutorial on transformer implementation

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Thanks to the authors of the "Attention Is All You Need" paper for their groundbreaking work
- The PyTorch team for their excellent deep learning framework

