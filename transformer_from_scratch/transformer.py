from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

import pandas as pd


def chunker(seq, size):
    chunks = [seq[pos : pos + size] for pos in range(0, len(seq), size)]
    return [" ".join(c) for c in chunks]


def get_shakespeare_text():
    import requests

    shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = requests.get(shakespeare_url)
    return req.text


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
            nn.Linear(4 * embed_dim, embed_dim),
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
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
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

    def generate(self, prompt, max_tokens=20, temperature=1.0):
        self.eval()
        tokens = tokenize(prompt)
        for _ in range(max_tokens):
            tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
            logits = self.forward(tokens_tensor)[:, -1, :]  # shape: (1, vocab_size)

            # Optionally adjust logits by temperature
            logits = logits / temperature

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()

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


def train(model, data, epochs=10, lr=0.001, warmup_steps=500, total_steps=10000):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Learning Rate Scheduler
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    global_step = 0
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
            scheduler.step()

            total_loss += loss.item()
            global_step += 1
            if global_step >= total_steps:
                break
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data)}")
        if global_step >= total_steps:
            break


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def curriculum_learning_step(
    model, data, chunk_size: int, data_size: int, epochs=10, lr=0.001, step=1
):
    data = [
        "".join(x)
        for x in pd.Series(chunker(shakespeare.split(), chunk_size)).sample(data_size)
    ]
    total_steps = len(data) * epochs
    warmup_steps = total_steps * 0.1
    print(
        f"curriculum learning {step}: (learning rate: {lr}, chunk size: {chunk_size}, epochs: {epochs}, dataset size: {len(data)})"
    )
    train(
        model,
        data,
        epochs=epochs,
        lr=lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    save_model(model, model_path)
    generated_text = model.generate("Hello", temperature=1)
    print()
    print("Generated text:", generated_text)
    print()


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

    shakespeare = get_shakespeare_text()

    num_epochs = 12
    # 1.
    curriculum_learning_step(model, shakespeare, 2, 100, num_epochs, 0.001, 1)
    # 2.
    curriculum_learning_step(model, shakespeare, 3, 500, num_epochs, 0.001, 2)
    # 3.
    curriculum_learning_step(model, shakespeare, 4, 1000, num_epochs, 0.001, 3)
    # 4.
    curriculum_learning_step(model, shakespeare, 5, 1200, num_epochs, 0.001, 4)
    # 5.
    curriculum_learning_step(model, shakespeare, 6, 1500, num_epochs, 0.001, 5)
    # 6.
    curriculum_learning_step(model, shakespeare, 7, 2000, num_epochs, 0.001, 6)
    # 7.
    curriculum_learning_step(model, shakespeare, 8, 2500, num_epochs, 0.001, 7)

    save_model(model, model_path)
    generated_text = model.generate("Let slip", temperature=1.1)
    print()
    print("Generated text:", generated_text)
    print()
