from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

import pandas as pd

from transformer_from_scratch.tokenizer import SimpleTokenizer


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
    def __init__(self, tokenizer, embed_dim, num_heads, num_layers, max_length=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.n_vocab
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_length, embed_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_dim, self.vocab_size)

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
        tokens = self.tokenizer.encode(prompt)
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
        return self.tokenizer.decode(tokens)


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
            tokens = model.tokenizer.encode(text)
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
    print()
    for temperature in [0.9, 1, 1.1, 1.2]:
        generated_text = model.generate("thou", temperature=temperature)
        print(f"Generated text (temp={temperature}):", generated_text)
    print()


def simple_training_loop(
    model: SimpleTransformer,
    data: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
):

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        # Iterate through the dataset in mini-batches
        for i in range(0, data.size(0), batch_size):
            batch = data[i : i + batch_size].to(device)
            # Prepare inputs and targets by shifting the sequence by one token
            inputs = batch[:, :-1]  # all tokens except the last one
            targets = batch[:, 1:]  # all tokens except the first one

            optimizer.zero_grad()
            # Forward pass: logits shape -> (batch_size, seq_len-1, vocab_size)
            logits = model(inputs)
            # Reshape logits and targets to compute the loss over all tokens
            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


if __name__ == "__main__":

    # Tokenizer using tiktoken
    # enc = tiktoken.get_encoding("gpt2")

    embed_dim = 16  # most be divisible by num_heads
    num_heads = 2
    num_layers = 2
    raw_data = get_shakespeare_text().split()
    enc = SimpleTokenizer(raw_data)
    # enc = tiktoken.get_encoding("gpt2")
    model = SimpleTransformer(enc, embed_dim, num_heads, num_layers)
    sequence_length = 4
    sample_raw_data = pd.Series(chunker(raw_data, sequence_length)).sample(1500)
    data = torch.tensor(([enc.encode(x) for x in sample_raw_data]))
    for _ in range(10):
        generated_text = model.generate("thou", temperature=1)
        print("(no training):", generated_text)
        print()
    simple_training_loop(model, data, num_epochs=1000, batch_size=10)
    for _ in range(10):
        generated_text = model.generate("thou", temperature=1)
        print("(after training):", generated_text)
        print()
    quit()

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "model.pth"
    if model_path.exists():
        print(f"Loading model from: {model_path}")
        model = SimpleTransformer(enc, embed_dim, num_heads, num_layers)
        model.to(device)
        load_model(model, model_path)
    else:
        model = SimpleTransformer(enc, embed_dim, num_heads, num_layers)
        model.to(device)
    # generate before training
    for _ in range(10):
        generated_text = model.generate("thou", temperature=1)
        print("Generated text (no training):", generated_text)
        print()

    shakespeare = get_shakespeare_text()

    num_epochs = 250
    # 1.
    lr = 0.0004
    curriculum_learning_step(model, shakespeare, 4, 500, num_epochs, lr, 1)
    curriculum_learning_step(model, shakespeare, 4, 10_000, num_epochs, lr * 0.75, 2)
    # curriculum_learning_step(model, shakespeare, 4, 50, num_epochs, lr * 0.5, 3)
    # curriculum_learning_step(model, shakespeare, 4, 50, num_epochs, lr * 0.25, 4)
    # curriculum_learning_step(model, shakespeare, 8, 50, num_epochs, lr * 0.125, 5)
    # curriculum_learning_step(model, shakespeare, 8, 50, num_epochs, lr * 0.0625, 6)
    # curriculum_learning_step(model, shakespeare, 8, 50, num_epochs, lr * 0.03125, 7)
    # curriculum_learning_step(model, shakespeare, 8, 50, num_epochs, lr * 0.015625, 8)

    # 2.
    # curriculum_learning_step(model, shakespeare, 3, 50_000, num_epochs, 0.001, 2)
    # 3.
    # curriculum_learning_step(model, shakespeare, 4, 10000, num_epochs, 0.001, 3)
    # # 4.
    # curriculum_learning_step(model, shakespeare, 5, 12000, num_epochs, 0.001, 4)
    # # 5.
    # curriculum_learning_step(model, shakespeare, 6, 15000, num_epochs, 0.001, 5)
    # # 6.
    # curriculum_learning_step(model, shakespeare, 7, 20000, num_epochs, 0.001, 6)
    # # 7.
    # curriculum_learning_step(model, shakespeare, 8, 25000, num_epochs, 0.001, 7)

    save_model(model, model_path)
    generated_text = model.generate("Let slip", temperature=1.1)
    print()
    print("Generated text:", generated_text)
    print()
