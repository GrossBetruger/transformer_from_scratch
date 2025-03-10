from pathlib import Path
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

def chunker(seq, size):
    chunks = [seq[pos:pos + size] for pos in range(0, len(seq), size)]
    return [" ".join(c) for c in chunks]


shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
req = requests.get(shakespeare_url)
with open("input.txt", "w") as f:
    f.write(req.text)

shakespeare = open("input.txt", "r").read()
print(f"Loaded Shakespeare text with length: {len(shakespeare)}")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simple Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

# Simple Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_length=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_length, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
        x = self.token_embedding(x) + self.positional_embedding(positions)
        x = x.permute(1, 0, 2)
        
        for layer in self.layers:
            x = layer(x)

        x = x.permute(1, 0, 2)
        logits = self.fc_out(x)
        return logits

    def generate(self, prompt, max_tokens=20):
        self.eval()
        tokens = tokenize(prompt)
        for _ in range(max_tokens):
            tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
            logits = self.forward(tokens_tensor)
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            tokens.append(next_token)
        return detokenize(tokens)

# Tokenizer using tiktoken
enc = tiktoken.get_encoding("gpt2")

def tokenize(text):
    return enc.encode(text)

def detokenize(tokens):
    return enc.decode(tokens)

def get_vocab_size():
    return enc.n_vocab

# Training Function
def train(model, data, epochs=10, lr=0.001):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for text in data:
            tokens = tokenize(text)
            input_tensor = torch.tensor(tokens[:-1]).unsqueeze(0).to(device)
            target_tensor = torch.tensor(tokens[1:]).unsqueeze(0).to(device)

            optimizer.zero_grad()
            logits = model(input_tensor)
            loss = criterion(logits.view(-1, logits.size(-1)), target_tensor.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data)}")


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))

if __name__ == "__main__":
    vocab_size = get_vocab_size()
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "model.pth"
    if model_path.exists():
        print(f"Loading model from: {model_path}")
        model = SimpleTransformer(vocab_size, embed_dim, num_heads, num_layers)
        load_model(model, model_path)
    else:
        model = SimpleTransformer(vocab_size, embed_dim, num_heads, num_layers)

    sample_data = ["".join(x) for x in chunker(shakespeare.split()[:1000], 4)]
    train(model, sample_data, epochs=10, lr=0.001)
    save_model(model, model_path)
    generated_text = model.generate("Hello")
    print("Generated text:", generated_text)
