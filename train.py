"""
Training Script for Shakespeare GPT

This script trains our GPT model on Shakespeare text.
Training takes approximately 15-30 minutes on a MacBook.

Key Concepts:
1. Training Loop - Forward pass, loss calculation, backward pass, optimizer step
2. Learning Rate Warmup - Gradually increase LR at the start for stable training
3. Evaluation - Periodic validation to check if model is learning
"""

import os
import time
import torch
from model import GPT
from dataset import load_data, get_batch


# ==============================================================================
# Hyperparameters (Settings)
# ==============================================================================

# Model architecture
EMBEDDING_DIM = 384    # Size of embeddings (how big each vector is)
NUM_HEADS = 6          # Number of attention heads
NUM_LAYERS = 6         # Number of transformer blocks
BLOCK_SIZE = 256       # Maximum sequence length
DROPOUT = 0.1          # Dropout rate for regularization

# Training settings
BATCH_SIZE = 64        # Number of sequences per batch
MAX_ITERS = 5000       # Total training iterations
EVAL_INTERVAL = 500    # Evaluate every N iterations
LEARNING_RATE = 3e-4   # Learning rate
WARMUP_ITERS = 100     # Warmup iterations (gradually increase LR)

# System
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/model.pt"


# ==============================================================================
# Learning Rate Schedule (with Warmup)
# ==============================================================================

def get_learning_rate(iteration):
    """
    Learning rate with warmup.

    Why warmup?
    At the start of training, the model weights are random and gradients can be
    large/unstable. Starting with a small LR and gradually increasing helps
    stabilize early training.

    Schedule:
        Iteration 0-100:   LR increases from 0 → 0.0003 (warmup)
        Iteration 100+:    LR stays at 0.0003 (constant)
    """
    if iteration < WARMUP_ITERS:
        # Warmup: linearly increase from 0 to LEARNING_RATE
        return LEARNING_RATE * (iteration / WARMUP_ITERS)
    else:
        # After warmup: use constant learning rate
        return LEARNING_RATE


# ==============================================================================
# Evaluation Function
# ==============================================================================

@torch.no_grad()
def evaluate(model, train_data, val_data):
    """Calculate average loss on training and validation data."""
    model.eval()
    
    results = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total_loss = 0.0
        
        # Run 100 batches to get a good estimate
        for _ in range(100):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            total_loss += loss.item()
        
        results[name] = total_loss / 100
    
    model.train()
    return results


# ==============================================================================
# Sample Generation (to see progress)
# ==============================================================================

@torch.no_grad()
def generate_sample(model, tokenizer):
    """Generate a text sample to see how the model is learning."""
    model.eval()
    
    prompt = "ROMEO:"
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    output_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.8)
    
    model.train()
    return tokenizer.decode(output_ids[0])


# ==============================================================================
# Main Training Function
# ==============================================================================

def train():
    print("=" * 60)
    print("Shakespeare GPT Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    train_data, val_data, tokenizer = load_data(block_size=BLOCK_SIZE)
    vocab_size = tokenizer.vocab_size

    # -------------------------------------------------------------------------
    # 2. Create Model
    # -------------------------------------------------------------------------
    print("\nCreating model...")
    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT
    )
    model = model.to(DEVICE)

    # -------------------------------------------------------------------------
    # 3. Create Optimizer
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\nStarting training...")
    print(f"Max iterations: {MAX_ITERS}")
    print(f"Eval interval: {EVAL_INTERVAL}\n")

    os.makedirs("checkpoints", exist_ok=True)
    start_time = time.time()

    for iteration in range(MAX_ITERS):

        # 4.1 Update learning rate (warmup schedule)
        lr = get_learning_rate(iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 4.2 Get a batch of training data
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
        x, y = x.to(DEVICE), y.to(DEVICE)

        # 4.3 Forward pass - get predictions and loss
        logits, loss = model(x, y)

        # 4.4 Backward pass - compute gradients
        optimizer.zero_grad()
        loss.backward()

        # 4.5 Gradient clipping - prevent exploding gradients
        # In deep networks, gradients can become very large during backprop.
        # This clips them to a maximum norm of 1.0 for training stability.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 4.6 Update weights
        optimizer.step()

        # 4.7 Evaluate periodically
        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            losses = evaluate(model, train_data, val_data)
            elapsed = time.time() - start_time
            
            print(f"Iter {iteration:5d} | "
                  f"Train Loss: {losses['train']:.4f} | "
                  f"Val Loss: {losses['val']:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {elapsed:.0f}s")

            # Show a sample generation
            if iteration > 0:
                print("\n--- Sample ---")
                print(generate_sample(model, tokenizer)[:400])
                print("--------------\n")

    # -------------------------------------------------------------------------
    # 5. Save Model
    # -------------------------------------------------------------------------
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'iteration': MAX_ITERS,
        'val_loss': losses['val'],
        'config': {
            'vocab_size': vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'block_size': BLOCK_SIZE,
        }
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    # -------------------------------------------------------------------------
    # 6. Done!
    # -------------------------------------------------------------------------
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Final val loss: {losses['val']:.4f}")
    print(f"Model saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
