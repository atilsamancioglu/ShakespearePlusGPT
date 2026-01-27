"""
Text Generation Script for Shakespeare GPT

This script loads a trained model and generates Shakespeare-like text.

Key Concepts:
1. Loading a trained model from checkpoint
2. Autoregressive generation - predicting one character at a time
3. Temperature - controls randomness (higher = more creative, lower = more predictable)
"""

import torch
from model import GPT
from dataset import download_shakespeare, CharacterTokenizer, DATA_PATH


# ==============================================================================
# Load Model
# ==============================================================================

def load_model(checkpoint_path: str = "checkpoints/model.pt"):
    """
    Load a trained GPT model from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint file

    Returns:
        model: The trained GPT model (ready for generation)
        tokenizer: The character tokenizer (to convert text <-> numbers)
        device: The device the model is on ('mps', 'cuda', or 'cpu')
    """
    # 1. Set up device (use GPU if available)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load the checkpoint file
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 3. Get the model configuration that was saved during training
    config = checkpoint['config']

    # 4. Create the tokenizer (we need the same vocabulary as training)
    download_shakespeare()
    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        text = file.read()
    tokenizer = CharacterTokenizer(text)

    # 5. Create the model with the saved configuration
    model = GPT(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        block_size=config['block_size'],
        dropout=0.0  # No dropout during generation
    )

    # 6. Load the trained weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # 7. Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # 8. Print info
    print(f"Model loaded!")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  Training iterations: {checkpoint['iteration']}")

    return model, tokenizer, device


# ==============================================================================
# Generate Text
# ==============================================================================

@torch.no_grad()
def generate(model, tokenizer, device, prompt: str, max_tokens: int = 500, temperature: float = 0.8):
    """
    Generate text given a starting prompt.

    -------------------------------------------------------------------------
    HOW TEXT GENERATION WORKS (Autoregressive)
    -------------------------------------------------------------------------
    The model generates one character at a time:

    1. Start with prompt:           "ROMEO:"
    2. Model predicts next char:    "ROMEO:" → 'O'
    3. Append and repeat:           "ROMEO:O" → ' '
    4. Continue:                    "ROMEO:O " → 'w'
    5. And so on...                 "ROMEO:O w" → 'h'

    This continues until we reach max_tokens.

    TEMPERATURE controls randomness:
    - Low (0.5):  Very predictable, "safe" choices
    - Medium (0.8): Balanced creativity and coherence (default)
    - High (1.5):  Very creative but may be nonsensical
    -------------------------------------------------------------------------
    """
    # 1. Convert prompt text to token IDs
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    input_ids = input_ids.unsqueeze(0)  # Add batch dimension: shape becomes (1, seq_len)

    # 2. Generate new tokens using the model's generate method
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature
    )

    # 3. Convert token IDs back to text
    generated_text = tokenizer.decode(output_ids[0])

    return generated_text


# ==============================================================================
# Main - Run Generation
# ==============================================================================

if __name__ == "__main__":

    # 1. Load the trained model
    print("=" * 60)
    print("Shakespeare GPT - Text Generation")
    print("=" * 60)

    model, tokenizer, device = load_model("checkpoints/model.pt")

    # 2. Define some example prompts to try
    prompts = [
        "ROMEO:",
        "To be, or not to be",
        "JULIET:\nO Romeo, ",
    ]

    # 3. Generate text for each prompt
    print("\n" + "=" * 60)
    print("Generating Text...")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n--- Prompt: {repr(prompt)} ---\n")

        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_tokens=300,
            temperature=0.8
        )

        print(generated_text)
        print("\n" + "-" * 60)

    # 4. Interactive: Let user try their own prompts
    print("\n" + "=" * 60)
    print("Try Your Own Prompts!")
    print("=" * 60)
    print("Enter a prompt to generate text, or 'quit' to exit.")
    print("Tip: Try character names like 'HAMLET:', 'KING:', 'First Citizen:'")

    while True:
        try:
            prompt = input("\nYour prompt: ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Farewell!")
                break

            if not prompt:
                continue

            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_tokens=500,
                temperature=0.8
            )

            print("\n" + generated_text)

        except KeyboardInterrupt:
            print("\n\nFarewell!")
            break
