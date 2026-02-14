"""
GPT Model for Shakespeare Text Generation

This module implements a simplified GPT (Generative Pre-trained Transformer) model.
GPT is a decoder-only transformer that predicts the next token in a sequence.

Architecture Overview:
1. Token Embedding - Converts characters to vectors
2. Position Embedding - Adds position information
3. Transformer Blocks - Self-attention + Feed-forward layers
4. Output Layer - Predicts next character

This implementation uses PyTorch's built-in MultiheadAttention for clarity.

-----------------------------------------------------------------------------
GPT vs "Attention Is All You Need" (Original Transformer)
-----------------------------------------------------------------------------
The original Transformer (Vaswani et al., 2017) uses an ENCODER-DECODER
architecture for translation tasks (e.g., English → French).
It has 3 types of attention:
  1. Encoder self-attention   (bidirectional - sees all tokens)
  2. Decoder masked attention (causal - can't see future tokens)
  3. Cross-attention          (decoder attends to encoder output)

GPT simplifies this by using ONLY the decoder part:
  - No encoder         → removed
  - No cross-attention → removed (no encoder to attend to)
  - Only masked self-attention + feed-forward layers remain

Original Transformer Block (Decoder):     GPT Block (this model):
┌─────────────────────────┐               ┌─────────────────────────┐
│ Masked Self-Attention   │  KEEP         │ Masked Self-Attention   │
├─────────────────────────┤               ├─────────────────────────┤
│ Cross-Attention         │  REMOVE       │                         │
├─────────────────────────┤               ├─────────────────────────┤
│ Feed-Forward            │  KEEP         │ Feed-Forward            │
└─────────────────────────┘               └─────────────────────────┘

Why does decoder-only work?
GPT's insight: just train a decoder to predict the next token on massive
amounts of text, and it learns language well enough for many tasks.

Common architectures:
  Encoder-Decoder:  Original Transformer, T5  (translation, summarization)
  Decoder-only:     GPT, LLaMA, ChatGPT       (text generation, chatbots)
  Encoder-only:     BERT                       (text understanding)
-----------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    Transformer Decoder Block

    Each block contains:
    1. Multi-Head Self-Attention (with causal mask)
    2. Feed-Forward Network (MLP)

    Both sublayers use:
    - Layer Normalization (applied before each sublayer - "Pre-LN")
    - Residual Connections (x + sublayer(x))
    """

    # 1. Initialize the class with hyperparameters
    # -------------------------------------------------------------------------
    # Default values (384, 6) are chosen for this educational project:
    # - Small enough to train quickly (~10M total params)
    # - Large enough to learn meaningful patterns
    #
    # For reference, GPT-2 Small uses embedding_dim=768, num_heads=12
    # The key relationship: head_size = embedding_dim / num_heads = 64
    # This head_size=64 is consistent across most GPT models.
    # -------------------------------------------------------------------------
    def __init__(self,
                 embedding_dim: int = 384,
                 num_heads: int = 6,
                 dropout: float = 0.1):
        super().__init__()

        # 2. Create the first Layer Normalization
        self.ln1 = nn.LayerNorm(embedding_dim)

        # 3. Create Multi-Head Self-Attention using PyTorch's built-in module
        # This handles Q, K, V projections and attention computation internally
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input shape: (batch, sequence, embedding)
        )

        # 4. Create the second Layer Normalization
        self.ln2 = nn.LayerNorm(embedding_dim)

        # 5. Create the Feed-Forward Network (MLP)
        # The MLP expands to 4x the size, then projects back:
        #   384 → 1536 → 384
        #
        # Why 4x? From "Attention Is All You Need" paper, Section 3.3:
        # "The dimensionality of input and output is d_model=512,
        #  and the inner-layer has dimensionality d_ff=2048"
        # 2048 / 512 = 4x. This ratio is used in most transformers since.
        #
        # The expansion gives the network more capacity to learn complex
        # patterns, then compresses back to the original size.
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),  # Expand: 384 → 1536
            nn.GELU(),                                     # Activation function
            nn.Linear(4 * embedding_dim, embedding_dim),  # Project back: 1536 → 384
            nn.Dropout(dropout)                            # Regularization
        )

    # 6. Create the forward method
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            causal_mask: Attention mask to prevent looking at future tokens

        Returns:
            Output tensor of same shape as input
        """
        # 7. Self-Attention with residual connection
        # Pre-LN: normalize first, then apply attention
        x_norm = self.ln1(x)
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask,
            is_causal=False  # We provide our own mask
        )
        x = x + attn_output  # Residual connection

        # 8. Feed-Forward with residual connection
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer)

    A decoder-only transformer for character-level text generation.
    Given a sequence of characters, it predicts the next character.

    Architecture:
    - Token Embedding: Maps character IDs to vectors
    - Position Embedding: Adds position information
    - N x Transformer Blocks: Process the sequence
    - Output Projection: Predicts next character probabilities
    """

    # 1. Initialize the class with hyperparameters
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 384,
                 num_heads: int = 6,
                 num_layers: int = 6,
                 block_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        # 2. Store block_size for generation
        self.block_size = block_size

        # 3. Create Token Embedding layer
        # Maps each character ID to a learnable vector of size embedding_dim
        #
        # -----------------------------------------------------------------------
        # nn.Embedding vs nn.Parameter (for ViT students)
        # -----------------------------------------------------------------------
        # In ViT, you used nn.Parameter to create learnable embeddings:
        #   self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        #   self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        #
        # nn.Embedding does the SAME thing - it's just a convenience wrapper.
        # Under the hood, it creates a learnable weight matrix (nn.Parameter)
        # with a built-in lookup-by-index operation:
        #
        #   nn.Parameter approach:
        #       self.token_emb = nn.Parameter(torch.randn(65, 384))
        #       emb = self.token_emb[token_ids]         # manual indexing
        #
        #   nn.Embedding approach (what we use):
        #       self.token_emb = nn.Embedding(65, 384)
        #       emb = self.token_emb(token_ids)          # built-in lookup
        #
        # Both are learnable, both are updated by backpropagation.
        # nn.Embedding is preferred here because token IDs change every batch
        # (different characters each time), so lookup-by-index is cleaner.
        #
        # What does it learn?
        # - Token embedding: learns what each character "means"
        #   (characters in similar contexts get similar vectors)
        # - Position embedding: learns what each position "means"
        #   (e.g., start of sentence behaves differently from middle)
        # -----------------------------------------------------------------------
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        # 4. Create Position Embedding layer
        # Each position (0, 1, 2, ..., block_size-1) gets its own learnable vector
        # Same as nn.Parameter(torch.randn(block_size, embedding_dim)) in ViT
        self.position_embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim
        )

        # 5. Create Embedding Dropout
        self.dropout = nn.Dropout(dropout)

        # 6. Create stack of Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 7. Create Final Layer Normalization
        self.ln_final = nn.LayerNorm(embedding_dim)

        # 8. Create Output Projection layer
        # Maps from embedding_dim to vocab_size (one score per character)
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

        # 8.5 Create Loss Function
        self.loss_fn = nn.CrossEntropyLoss()

        # 9. Create Causal Mask (to prevent attending to future tokens)
        #
        # -----------------------------------------------------------------------
        # WHY DO WE NEED A CAUSAL MASK?
        # -----------------------------------------------------------------------
        # During training, we feed the ENTIRE sequence at once for efficiency.
        # But the model should only predict based on PREVIOUS tokens, not future ones.
        #
        # Without mask (CHEATING):
        #   To predict what comes after "To", the model could peek at "be" → unfair!
        #
        # With mask (CORRECT):
        #   Each position can only "see" itself and previous positions.
        #
        # Example for sequence "To be" (4 tokens):
        #
        #              Attending TO position:
        #              0    1    2    3
        #            ['T', 'o', ' ', 'b']
        #
        # Pos 0 'T':  [ ok  MASK MASK MASK ]  ← 'T' can only see itself
        # Pos 1 'o':  [ ok   ok  MASK MASK ]  ← 'o' can see 'T', 'o'
        # Pos 2 ' ':  [ ok   ok   ok  MASK ]  ← ' ' can see 'T', 'o', ' '
        # Pos 3 'b':  [ ok   ok   ok   ok  ]  ← 'b' can see all previous
        #
        # -----------------------------------------------------------------------
        # REFERENCE: "Attention Is All You Need" (Vaswani et al., 2017)
        # Section 3.1 - "Masked Multi-Head Attention"
        # Paper: https://arxiv.org/abs/1706.03762
        #
        # Quote: "We need to prevent leftward information flow in the decoder
        #         to preserve the auto-regressive property."
        #
        # Note: The original paper uses an encoder-decoder architecture.
        #       GPT simplifies this by using ONLY the decoder with causal masking.
        # -----------------------------------------------------------------------
        #
        # torch.triu creates an upper triangular matrix of True values.
        # True = masked (blocked), False = allowed to attend
        causal_mask = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool),
            diagonal=1
        )
        # register_buffer: saves tensor with model & moves it to GPU with model,
        # but it's NOT a learnable parameter (optimizer won't update it)
        self.register_buffer("causal_mask", causal_mask)

        # 10. Initialize weights
        self.apply(self._init_weights)

        # 11. Print model size
        total_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model created with {total_params:,} parameters")

    def _init_weights(self, module):
        """
        Initialize weights with small random values (std=0.02).

        Why not use PyTorch defaults?
        - PyTorch uses larger initial values (std=1.0 for Embedding, Kaiming for Linear)
        - Small weights (std=0.02) help training stability in deep transformer networks

        Reference: "Improving Language Understanding by Generative Pre-Training"
        (Radford et al., 2018) - The original GPT paper
        Paper: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
        Quote: "Model weights were initialized to N(0, 0.02)"
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 12. Create the forward method
    def forward(self,
                input_ids: torch.Tensor,
                targets: torch.Tensor = None) -> tuple:
        """
        Forward pass of the GPT model.

        Args:
            input_ids: Input token IDs, shape (batch_size, sequence_length)
            targets: Target token IDs for loss calculation (optional)

        Returns:
            logits: Predicted scores, shape (batch_size, sequence_length, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 13. Get Token Embeddings
        token_emb = self.token_embedding(input_ids)

        # 14. Get Position Embeddings
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.position_embedding(positions)

        # 15. Combine Token and Position Embeddings
        # token_emb = "what" each character is (semantic meaning)
        # pos_emb   = "where" each character is (position 0, 1, 2, ...)
        # Adding them gives the model both pieces of information.
        #
        # Dropout randomly zeros ~10% of values during training for regularization.
        # This prevents overfitting by forcing the model to not rely on any single value.
        # Reference: "Attention Is All You Need", Section 5.4
        # "We apply dropout to the sums of the embeddings and the positional encodings"
        x = self.dropout(token_emb + pos_emb)

        # 16. Get the causal mask for current sequence length
        # The full mask is (block_size x block_size), but our sequence might be shorter.
        # We slice to get only the (seq_len x seq_len) portion we need.
        mask = self.causal_mask[:seq_len, :seq_len]

        # 17. Pass through Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)

        # 18. Apply Final Layer Normalization
        x = self.ln_final(x)

        # 19. Project to vocabulary size
        logits = self.output_proj(x)

        # 20. Calculate loss if targets provided
        # - During TRAINING: targets provided → calculate loss to learn from mistakes
        # - During GENERATION: no targets → just return predictions (loss=None)
        loss = None
        if targets is not None:
            # CrossEntropyLoss expects:
            #   predictions: (N, num_classes)  → 2D
            #   targets:     (N,)              → 1D
            #
            # But our shapes are:
            #   logits:  (batch_size, seq_len, vocab_size) → 3D
            #   targets: (batch_size, seq_len)             → 2D
            #
            # So we flatten batch & sequence into a single dimension:
            #   logits:  (batch_size * seq_len, vocab_size)
            #   targets: (batch_size * seq_len,)
            batch_size, seq_len, vocab_size = logits.shape

            logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
            targets_flat = targets.reshape(batch_size * seq_len)

            loss = self.loss_fn(logits_flat, targets_flat)

        return logits, loss

    # 21. Generate method for text generation
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """
        Generate new tokens one at a time (autoregressive generation).

        How it works:
            "ROMEO:" → predict 'O' → "ROMEO:O" → predict ' ' → "ROMEO:O " → ...

        Args:
            input_ids: Starting tokens, shape (batch_size, sequence_length)
            max_new_tokens: How many new characters to generate
            temperature: Controls randomness (0.5=predictable, 1.0=normal, 1.5=creative)
        """
        for _ in range(max_new_tokens):

            # 22. If sequence is longer than block_size, crop to last block_size tokens
            # .size(1) gets the sequence length (dim 0=batch, dim 1=sequence)
            # -self.block_size uses negative indexing to take the LAST 256 tokens
            # e.g., 350 tokens with block_size=256 → [:, -256:] → last 256 tokens
            # This is needed because position embeddings only go up to block_size
            if input_ids.size(1) <= self.block_size:
                current_input = input_ids
            else:
                current_input = input_ids[:, -self.block_size:]

            # 23. Get model predictions
            logits, _ = self.forward(current_input)

            # 24. Take only the last position's predictions (what comes next?)
            last_logits = logits[:, -1, :]

            # 25. Apply temperature (divide to control randomness)
            # Higher temp → more uniform probs → more random
            # Lower temp → sharper probs → more predictable
            last_logits = last_logits / temperature

            # 26. Convert logits to probabilities
            probs = F.softmax(last_logits, dim=-1)

            # 27. Sample one token from the probability distribution
            # Why multinomial (random sampling) instead of argmax?
            # - In vision: argmax picks ONE correct class (cat is always cat)
            # - In text generation: there are many valid next characters
            #   argmax would always pick the same char → "the the the the..."
            #   multinomial samples randomly based on probabilities → natural text
            next_token = torch.multinomial(probs, num_samples=1)

            # 28. Append to sequence and continue
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# Test the model
if __name__ == "__main__":
    # 1. Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create model
    model = GPT(
        vocab_size=65,
        embedding_dim=384,
        num_heads=6,
        num_layers=6,
        block_size=256
    ).to(device)

    # 3. Create dummy input
    batch_size = 4
    seq_len = 64
    dummy_input = torch.randint(0, 65, (batch_size, seq_len)).to(device)
    dummy_targets = torch.randint(0, 65, (batch_size, seq_len)).to(device)

    # 4. Test forward pass
    logits, loss = model(dummy_input, dummy_targets)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # 5. Test generation
    generated = model.generate(dummy_input[:1, :10], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
