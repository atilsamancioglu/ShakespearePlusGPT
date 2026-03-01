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

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.rule import Rule

console = Console()


# ==============================================================================
# Hyperparameters (Settings)
# ==============================================================================

# Model architecture
EMBEDDING_DIM = 384  # Size of embeddings (how big each vector is)
NUM_HEADS = 6  # Number of attention heads
NUM_LAYERS = 6  # Number of transformer blocks
BLOCK_SIZE = 256  # Maximum sequence length
DROPOUT = 0.1  # Dropout rate for regularization

# Training settings
BATCH_SIZE = 64  # Number of sequences per batch
MAX_ITERS = 5000  # Total training iterations
EVAL_INTERVAL = 500  # Evaluate every N iterations
LEARNING_RATE = 3e-4  # Learning rate
WARMUP_ITERS = 100  # Warmup iterations (gradually increase LR)

# System
DEVICE = "cuda" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/model.pt"


# ==============================================================================
# Learning Rate Schedule (with Warmup)
# ==============================================================================


def get_learning_rate(iteration):
    """
    Learning rate with warmup.

    -------------------------------------------------------------------------
    WHY START WITH A LOW LEARNING RATE? (Warmup)
    -------------------------------------------------------------------------
    At iteration 0, weights are RANDOM → gradients point in unreliable
    directions. A large LR would take big steps in wrong directions,
    potentially destabilizing training permanently.

    Warmup lets the model "get its bearings" first:
      - Start slow → gradients are unreliable, take small careful steps
      - Speed up   → gradients become meaningful, learn efficiently

    Analogy: Like driving in an unfamiliar city - go slow at first
    to read the signs, then speed up once you know where you're going.

    Full training LR strategy (used in GPT papers):
      Start:  low → high  (warmup - unreliable gradients)
      Middle: constant     (learn efficiently)
      End:    high → low   (fine-tune, don't overshoot)

    We implement the first two stages for simplicity.
    -------------------------------------------------------------------------

    Schedule:
        Iteration 0-100:   LR increases from 0 → 0.0003 (warmup)
        Iteration 100+:    LR stays at 0.0003 (constant)
    """
    if iteration < WARMUP_ITERS:
        return LEARNING_RATE * (iteration / WARMUP_ITERS)
    else:
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
# Rich UI Helpers
# ==============================================================================


def format_time(seconds):
    """Format seconds into human-readable HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_stats_table(history):
    """Build a rich table showing training history."""
    table = Table(
        title="📊 Training History",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold cyan",
        show_lines=True,
        expand=False,
    )
    table.add_column("Iteration", style="bold white", justify="right", width=10)
    table.add_column("Train Loss", style="green", justify="center", width=12)
    table.add_column("Val Loss", style="yellow", justify="center", width=12)
    table.add_column("LR", style="magenta", justify="center", width=10)
    table.add_column("Elapsed", style="cyan", justify="center", width=10)

    for row in history:
        # Color val loss: green if improving, red if worsening
        val_color = "bright_green"
        if len(history) > 1:
            prev_val = (
                history[-2]["val_loss"] if history[-2] != row else row["val_loss"]
            )
            val_color = "bright_green" if row["val_loss"] <= prev_val else "bright_red"

        table.add_row(
            f"{row['iter']:,}",
            f"{row['train_loss']:.4f}",
            f"[{val_color}]{row['val_loss']:.4f}[/{val_color}]",
            f"{row['lr']:.2e}",
            format_time(row["elapsed"]),
        )

    return table


def build_info_panel(iteration, loss, lr, elapsed, iters_per_sec):
    """Build a live status panel."""
    remaining_iters = MAX_ITERS - iteration
    eta = remaining_iters / iters_per_sec if iters_per_sec > 0 else 0
    progress_pct = 100 * iteration / MAX_ITERS

    info = (
        f"[bold cyan]Iter:[/bold cyan]        {iteration:,} / {MAX_ITERS:,}\n"
        f"[bold cyan]Progress:[/bold cyan]    {progress_pct:.1f}%\n"
        f"[bold cyan]Loss:[/bold cyan]        [bold yellow]{loss:.4f}[/bold yellow]\n"
        f"[bold cyan]LR:[/bold cyan]          [magenta]{lr:.2e}[/magenta]\n"
        f"[bold cyan]Speed:[/bold cyan]       {iters_per_sec:.1f} iter/s\n"
        f"[bold cyan]Elapsed:[/bold cyan]     [green]{format_time(elapsed)}[/green]\n"
        f"[bold cyan]ETA:[/bold cyan]         [red]{format_time(eta)}[/red]"
    )
    return Panel(
        info, title="⚡ Live Stats", border_style="bright_yellow", expand=False
    )


# ==============================================================================
# Main Training Function
# ==============================================================================


def train():
    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel.fit(
            "[bold magenta]🎭  Shakespeare GPT — Training[/bold magenta]\n"
            f"[dim]Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}  |"
            f"  Iters: {MAX_ITERS}  |  Heads: {NUM_HEADS}  |  Layers: {NUM_LAYERS}[/dim]",
            border_style="magenta",
            box=box.DOUBLE_EDGE,
        )
    )
    console.print()

    # ── Load Data ─────────────────────────────────────────────────────────────
    with console.status("[bold green]Loading dataset...[/bold green]", spinner="dots"):
        train_data, val_data, tokenizer = load_data(block_size=BLOCK_SIZE)
        vocab_size = tokenizer.vocab_size
    console.print(f"  ✅  Data loaded — vocab size: [bold]{vocab_size:,}[/bold]")

    # ── Create Model ──────────────────────────────────────────────────────────
    with console.status("[bold green]Building model...[/bold green]", spinner="dots"):
        model = GPT(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            block_size=BLOCK_SIZE,
            dropout=DROPOUT,
        )
        model = model.to(DEVICE)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6

    console.print(f"  ✅  Model ready — [bold]{param_count:.2f}M[/bold] parameters")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ── Training Loop ─────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold blue]Training[/bold blue]", style="blue"))
    console.print()

    os.makedirs("checkpoints", exist_ok=True)

    history = []  # list of eval snapshots
    start_time = time.time()
    iter_times = []  # rolling window for speed estimate
    current_loss = float("inf")

    with Progress(
        SpinnerColumn(style="bold magenta"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, style="magenta", complete_style="bold green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=5,
        transient=False,
    ) as progress:
        task = progress.add_task("[bold cyan]Training…[/bold cyan]", total=MAX_ITERS)

        for iteration in range(MAX_ITERS):
            iter_start = time.time()

            # ── LR warmup ──────────────────────────────────────────────────
            lr = get_learning_rate(iteration)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # ── Forward / backward ─────────────────────────────────────────
            x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits, loss = model(x, y)
            current_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ── Speed tracking ─────────────────────────────────────────────
            iter_times.append(time.time() - iter_start)
            if len(iter_times) > 50:
                iter_times.pop(0)
            iters_per_sec = 1.0 / (sum(iter_times) / len(iter_times))

            # ── Update progress bar label ──────────────────────────────────
            phase = (
                "[yellow]warmup[/yellow]"
                if iteration < WARMUP_ITERS
                else "[green]train[/green]"
            )
            progress.update(
                task,
                advance=1,
                description=(
                    f"[bold cyan]{phase}[/bold cyan]  "
                    f"loss=[bold yellow]{current_loss:.4f}[/bold yellow]  "
                    f"lr=[magenta]{lr:.1e}[/magenta]  "
                    f"[dim]{iters_per_sec:.1f} it/s[/dim]"
                ),
            )

            # ── Periodic evaluation ────────────────────────────────────────
            if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
                elapsed = time.time() - start_time
                losses = evaluate(model, train_data, val_data)

                history.append(
                    {
                        "iter": iteration,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "lr": lr,
                        "elapsed": elapsed,
                    }
                )

                # Print a snapshot below the bar
                progress.console.print(
                    f"  [dim]│[/dim] [bold white]iter {iteration:5,}[/bold white]  "
                    f"train=[green]{losses['train']:.4f}[/green]  "
                    f"val=[yellow]{losses['val']:.4f}[/yellow]  "
                    f"elapsed=[cyan]{format_time(elapsed)}[/cyan]  "
                    f"ETA=[red]{format_time((MAX_ITERS - iteration) / iters_per_sec)}[/red]"
                )

                # Save checkpoint
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "iteration": iteration,
                    "val_loss": losses["val"],
                    "config": {
                        "vocab_size": vocab_size,
                        "embedding_dim": EMBEDDING_DIM,
                        "num_heads": NUM_HEADS,
                        "num_layers": NUM_LAYERS,
                        "block_size": BLOCK_SIZE,
                    },
                }
                torch.save(checkpoint, CHECKPOINT_PATH)

                # Show text sample (skip iteration 0)
                if iteration > 0:
                    sample = generate_sample(model, tokenizer)[:400]
                    progress.console.print(
                        Panel(
                            f"[italic]{sample}[/italic]",
                            title="🎭 Generated Sample",
                            border_style="dim blue",
                            expand=False,
                        )
                    )

    # ── History Table ─────────────────────────────────────────────────────────
    console.print()
    console.print(build_stats_table(history))

    # ── Final Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    console.print()
    console.print(
        Panel(
            f"[bold green]✅  Training complete![/bold green]\n\n"
            f"  Total time  : [cyan]{format_time(total_time)}[/cyan]  "
            f"([dim]{total_time / 60:.1f} min[/dim])\n"
            f"  Final val loss : [bold yellow]{history[-1]['val_loss']:.4f}[/bold yellow]\n"
            f"  Checkpoint  : [dim]{CHECKPOINT_PATH}[/dim]",
            border_style="bright_green",
            box=box.DOUBLE_EDGE,
        )
    )


if __name__ == "__main__":
    train()
