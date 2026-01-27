# Shakespeare GPT

A simple, educational implementation of GPT (Generative Pre-trained Transformer) using PyTorch. This project trains a character-level language model on Shakespeare's works to generate Shakespeare-like text.

**Built for learning** - The code is intentionally simple and heavily commented to help students understand how GPT and transformers work.

## What You'll Learn

- How **transformers** work (attention mechanism, embeddings, feed-forward layers)
- How **GPT** generates text (autoregressive generation)
- How **training loops** work in PyTorch
- Key concepts: tokenization, embeddings, causal masking, temperature sampling

## Project Structure

```
ShakespearePlusGPT/
├── model.py      # GPT model architecture (TransformerBlock, GPT class)
├── dataset.py    # Data loading and character-level tokenization
├── train.py      # Training loop
├── generate.py   # Text generation script
└── requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- A Mac with Apple Silicon (MPS) or a machine with CUDA GPU (or CPU, but slower)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ShakespearePlusGPT.git
cd ShakespearePlusGPT

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python train.py
```

This will:
- Download the Tiny Shakespeare dataset (~1MB)
- Train the GPT model for 5000 iterations
- Save the model to `checkpoints/model.pt`
- Show sample generations during training

**Training time:** ~15-30 minutes on MacBook with M1/M2

### 2. Generate Text

```bash
python generate.py
```

This will load the trained model and let you enter prompts interactively.

**Example prompts to try:**
```
ROMEO:
JULIET:
To be, or not to be
HAMLET:
O Romeo, Romeo,
```

## Model Architecture

```
Input: "To be or" (characters)
         ↓
┌─────────────────────────────┐
│   Token Embedding           │  Characters → Vectors
│   Position Embedding        │  Add position information
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│   Transformer Block (×6)    │
│   ├── Multi-Head Attention  │  Which characters to focus on?
│   ├── Feed-Forward Network  │  Process the information
│   └── Residual Connections  │  Help gradients flow
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│   Output Projection         │  Vectors → Character probabilities
└─────────────────────────────┘
         ↓
Output: " not" (next character prediction)
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embedding_dim` | 384 | Size of embedding vectors |
| `num_heads` | 6 | Number of attention heads |
| `num_layers` | 6 | Number of transformer blocks |
| `block_size` | 256 | Maximum sequence length |
| `batch_size` | 64 | Sequences per training batch |
| `learning_rate` | 3e-4 | Learning rate |
| `max_iters` | 5000 | Total training iterations |

**Total parameters:** ~10.6 million

## Key Concepts Explained

### Character-Level Tokenization
We use simple character-level tokenization for educational clarity. Each character becomes an integer ID, which is then looked up in a learned embedding table.

```python
"hello" → [7, 4, 11, 11, 14] → [[0.2, -0.1, ...], [0.5, 0.3, ...], ...]
```

*Note: Production models like GPT-4 use subword tokenization (BPE) which is more efficient but harder to understand.*

### Causal Masking
The model can only "see" previous characters when predicting the next one. This is enforced with a causal mask in the attention mechanism.

```
Predicting position 3:  [CAN SEE] [CAN SEE] [CAN SEE] [PREDICTING] [MASKED] [MASKED]
```

### Temperature
Controls randomness in generation:
- **Low (0.5):** More predictable, "safer" choices
- **Medium (0.8):** Balanced (default)
- **High (1.5):** More creative but potentially nonsensical

## Sample Output

After training, the model generates text like:

```
ROMEO:
O, she doth teach the torches to burn bright!
It seems she hangs upon the cheek of night
Like a rich jewel in an Ethiope's ear;
Beauty too rich for use, for earth too dear!
```

*(Actual output will vary based on training and temperature)*

## References

This implementation is based on concepts from:

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Original Transformer paper
   - https://arxiv.org/abs/1706.03762

2. **"Improving Language Understanding by Generative Pre-Training"** (Radford et al., 2018)
   - Original GPT paper
   - https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

3. **Andrej Karpathy's nanoGPT**
   - Inspiration for educational implementation
   - https://github.com/karpathy/nanoGPT

## License

MIT License - Feel free to use this for learning and teaching!

## Acknowledgments

- Shakespeare dataset from [Andrej Karpathy's char-rnn](https://github.com/karpathy/char-rnn)
- Inspired by the amazing educational content from Andrej Karpathy
