"""
Shakespeare Dataset Module

This module handles loading and preparing the Tiny Shakespeare dataset.
We use character-level tokenization for simplicity.

Key Concepts:
1. Tokenization - Converting characters to numbers
2. Train/Validation Split - Separating data for training and evaluation
3. Batch Generation - Creating input-target pairs for training
"""

import os
import urllib.request
import torch

# URL for the Tiny Shakespeare dataset (forked from Andrej Karpathy)
SHAKESPEARE_URL = "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/refs/heads/main/input.txt"
DATA_PATH = "data/shakespeare.txt"


def download_shakespeare():
    """Download the Shakespeare dataset if not already present."""
    if os.path.exists(DATA_PATH):
        print(f"Dataset already exists at {DATA_PATH}")
        return

    print("Downloading Shakespeare dataset...")
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
    print(f"Downloaded to {DATA_PATH}")


class CharacterTokenizer:
    """
    Character-level Tokenizer

    Converts text to numbers and back.
    Each unique character gets a unique ID.

    Example:
        'hello' -> [7, 4, 11, 11, 14]
        [7, 4, 11, 11, 14] -> 'hello'

    -------------------------------------------------------------------------
    WHY CHARACTER-LEVEL TOKENIZATION?
    -------------------------------------------------------------------------
    We use character-level tokenization for educational simplicity.
    Industry models (GPT, BERT) use subword tokenization (BPE, WordPiece)
    which is more efficient but adds complexity.

    IMPORTANT: The integer IDs we assign are ARBITRARY - they're just indices!
    The neural network doesn't "see" these numbers directly. Instead:

        Character → Integer ID → Embedding Layer → Learned Vector
           'h'    →     7      →   lookup[7]    → [0.23, -0.45, ...]

    The embedding layer (nn.Embedding) is a LEARNABLE lookup table.
    During training, backpropagation updates these vectors so that
    characters appearing in similar contexts get similar representations.

    This is the same principle as Word2Vec: the actual "meaning" is not in
    the arbitrary ID, but in the learned embedding vector.
    -------------------------------------------------------------------------
    """

    def __init__(self, text: str):
        # 1. Get all unique characters and sort them
        self.characters = sorted(list(set(text)))

        # 2. Store vocabulary size
        self.vocab_size = len(self.characters)

        # 3. Create character to ID mapping
        self.char_to_id = {}
        for index, char in enumerate(self.characters):
            self.char_to_id[char] = index

        # 4. Create ID to character mapping
        self.id_to_char = {}
        for index, char in enumerate(self.characters):
            self.id_to_char[index] = char

        # 5. Print vocabulary info
        print(f"Vocabulary size: {self.vocab_size} characters")
        print(f"Characters: {repr(''.join(self.characters[:50]))}...")

    def encode(self, text: str) -> list:
        """Convert text to list of integer IDs."""
        ids = []
        for char in text:
            ids.append(self.char_to_id[char])
        return ids

    def decode(self, ids: list) -> str:
        """Convert list of integer IDs back to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        characters = []
        for id in ids:
            characters.append(self.id_to_char[id])
        return "".join(characters)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    """
    Generate a random batch of training data.

    -------------------------------------------------------------------------
    HOW LANGUAGE MODEL TRAINING WORKS
    -------------------------------------------------------------------------
    We create input-target pairs where TARGET = INPUT shifted by 1 position.
    The model learns to predict the NEXT character at each position.

    Example with block_size=5:

        Text: "To be or not to be"

        1. Pick random starting position, grab (block_size + 1) characters:
           chunk = ['T', 'o', ' ', 'b', 'e', ' ']   (6 chars)

        2. Split into input (x) and target (y):
           x = ['T', 'o', ' ', 'b', 'e']     (first 5 chars)
           y = ['o', ' ', 'b', 'e', ' ']     (last 5 chars = shifted by 1)

        3. The model learns to predict:
           Given 'T'       → predict 'o'
           Given 'To'      → predict ' '
           Given 'To '     → predict 'b'
           Given 'To b'    → predict 'e'
           Given 'To be'   → predict ' '

    This is done for multiple random positions (batch_size) at once.
    -------------------------------------------------------------------------

    Args:
        data: Tensor of all token IDs
        block_size: Length of each sequence
        batch_size: Number of sequences per batch

    Returns:
        x: Input sequences, shape (batch_size, block_size)
        y: Target sequences, shape (batch_size, block_size)
    """
    # 1. Pick random starting positions
    max_start = len(data) - block_size - 1
    positions = torch.randint(max_start, (batch_size,))

    # 2. Extract input (x) and target (y) for each position
    x_list = []
    y_list = []

    for pos in positions:
        x_list.append(data[pos : pos + block_size])  # Input: chars 0 to n-1
        y_list.append(
            data[pos + 1 : pos + block_size + 1]
        )  # Target: chars 1 to n (shifted by 1)

    # 3. Stack into batch tensors: (batch_size, block_size)
    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x, y


def load_data(block_size: int = 256, train_split: float = 0.9):
    """
    Load and prepare the Shakespeare dataset.

    Args:
        block_size: Length of each sequence
        train_split: Fraction of data for training (rest is validation)

    Returns:
        train_data: Training token IDs as tensor
        val_data: Validation token IDs as tensor
        tokenizer: The character tokenizer
    """
    # 1. Download the dataset
    download_shakespeare()

    # 2. Load the text file
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    print(f"\nDataset size: {len(text):,} characters")
    print(f"Sample text:\n{text[:200]}")
    print("..." + "-" * 50)

    # 3. Create the tokenizer
    tokenizer = CharacterTokenizer(text)

    # 4. Encode the entire text to token IDs
    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids, dtype=torch.long)

    # 5. Split into training and validation sets
    split_index = int(train_split * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    print(f"\nTrain size: {len(train_data):,} tokens")
    print(f"Val size: {len(val_data):,} tokens")

    return train_data, val_data, tokenizer


# Test the dataset
if __name__ == "__main__":
    # 1. Load data
    train_data, val_data, tokenizer = load_data(block_size=128)

    # 2. Get a sample batch
    x, y = get_batch(train_data, block_size=128, batch_size=4)

    print(f"\nSample batch:")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")

    # 3. Show the input-target relationship
    print(f"\n--- Demonstrating input-target pairs ---")
    print(f"Input (x[0]):  {tokenizer.decode(x[0][:20])}...")
    print(f"Target (y[0]): {tokenizer.decode(y[0][:20])}...")
    print(f"Notice: target is input shifted by 1 character!")
