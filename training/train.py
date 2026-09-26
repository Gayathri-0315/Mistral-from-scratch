import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config

from training.dataset import TextDataset
from model.mistral import MistralModel


# -----------------------------
# Dataset
# -----------------------------

full_dataset = TextDataset(
    config.TRAIN_FILE,
    config.TOKENIZER_FILE,
    seq_len=config.MAX_SEQ_LEN
)

# Split into training and validation data
validation_size = int(
    len(full_dataset) * config.VALIDATION_SPLIT
)

training_size = len(full_dataset) - validation_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Model
# -----------------------------

model = MistralModel(
    vocab_size=config.VOCAB_SIZE,
    hidden_size=config.HIDDEN_SIZE,
    num_heads=config.NUM_HEADS,
    intermediate_size=config.INTERMEDIATE_SIZE,
    num_layers=config.NUM_LAYERS,
    max_seq_len=config.MAX_SEQ_LEN
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE
)


# -----------------------------
# Training
# -----------------------------

best_val_loss = float("inf")

print("Starting training...")
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

for epoch in range(config.EPOCHS):

    total_loss = 0.0

    for inputs, targets in train_loader:

        optimizer.zero_grad()

        logits = model(inputs)

        loss = F.cross_entropy(
            logits.reshape(-1, config.VOCAB_SIZE),
            targets.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1:02d}/{config.EPOCHS} "
        f"| Loss: {average_loss:.4f}"
    )


print("Training complete!")


average_loss = total_loss / len(train_loader)

# Validation
model.eval()

validation_loss = 0.0

with torch.no_grad():

    for inputs, targets in val_loader:

        logits = model(inputs)

        loss = F.cross_entropy(
            logits.reshape(-1, config.VOCAB_SIZE),
            targets.reshape(-1)
        )

        validation_loss += loss.item()

validation_loss /= len(val_loader)
if validation_loss < best_val_loss:
    best_val_loss = validation_loss

    torch.save(
        model.state_dict(),
        config.CHECKPOINT_FILE
    )

    print("✓ Best model saved!")

model.train()

# -----------------------------
# Save model
# -----------------------------

torch.save(
    model.state_dict(),
    config.CHECKPOINT_FILE
)

print(
    f"Epoch {epoch + 1:02d}/{config.EPOCHS} "
    f"| Train Loss: {average_loss:.4f} "
    f"| Val Loss: {validation_loss:.4f}"
)