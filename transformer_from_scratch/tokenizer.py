from sortedcontainers import SortedSet


class SimpleTokenizer:
    def __init__(self, vocab, unk_token="<UNK>"):
        """
        Initialize the tokenizer with a given vocabulary.
        Args:
            vocab (list of str): List of tokens to be used as vocabulary.
            unk_token (str): Token to represent unknown words.
        """
        self.vocab = list(SortedSet(vocab))
        self.unk_token = unk_token

        # Create mappings from token to id and id to token.
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

        # Optionally add the unknown token if not already in vocab.
        if unk_token not in self.token_to_id:
            unk_id = len(self.token_to_id)
            self.token_to_id[unk_token] = unk_id
            self.id_to_token[unk_id] = unk_token

        self.n_vocab = len(self.token_to_id)

    def encode(self, text):
        """
        Encodes a string into a list of token ids.
        Splits text on whitespace and maps tokens to their corresponding id.
        Tokens not found in the vocabulary are replaced with the unknown token id.

        Args:
            text (str): The text to encode.
        Returns:
            List[int]: A list of token ids.
        """
        tokens = text.split()  # Naively split by whitespace
        token_ids = []
        for token in tokens:
            # If token is in vocab, use its id; otherwise use the id for unk_token.
            token_ids.append(
                self.token_to_id.get(token, self.token_to_id[self.unk_token])
            )
        return token_ids

    def decode(self, token_ids):
        """
        Decodes a list of token ids back into a string.
        Args:
            token_ids (List[int]): A list of token ids.
        Returns:
            str: The decoded string.
        """
        tokens = [
            self.id_to_token.get(token_id, self.unk_token) for token_id in token_ids
        ]
        return " ".join(tokens)


# Example usage:
if __name__ == "__main__":
    vocab = ["king", "queen", "boy", "girl"]
    tokenizer = SimpleTokenizer(vocab)

    # Test encoding
    text = "king boy unknown"
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)  # Output will be something like: [0, 2, <unk_id>]

    # Test decoding
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)  # Output: "king boy <UNK>"
