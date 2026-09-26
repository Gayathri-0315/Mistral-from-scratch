import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import config
from tokenizers import Tokenizer

from model.mistral import MistralModel


# -----------------------------
# Configuration
# -----------------------------

VOCAB_SIZE = 649
SEQ_LEN = 8

HIDDEN_SIZE = 64
NUM_HEADS = 4
INTERMEDIATE_SIZE = 256
NUM_LAYERS = 2


# -----------------------------
# Load tokenizer
# -----------------------------

tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)




# -----------------------------
# Load model
# -----------------------------
tokenizer = Tokenizer.from_file(config.TOKENIZER_FILE)

model = MistralModel(
    vocab_size=config.VOCAB_SIZE,
    hidden_size=config.HIDDEN_SIZE,
    num_heads=config.NUM_HEADS,
    intermediate_size=config.INTERMEDIATE_SIZE,
    num_layers=config.NUM_LAYERS,
    max_seq_len=config.MAX_SEQ_LEN
)

model.load_state_dict(
    torch.load(
        config.CHECKPOINT_FILE,
        map_location="cpu"
    )
)

model.eval()


# -----------------------------
# Generate text
# -----------------------------
def generate(
    prompt,
    max_new_tokens=config.MAX_NEW_TOKENS,
    temperature=config.TEMPERATURE
):
    generated_ids = tokenizer.encode(prompt).ids
    eos_id = tokenizer.token_to_id("<eos>")

    for _ in range(max_new_tokens):

        input_ids = torch.tensor(
            [generated_ids[-config.MAX_SEQ_LEN:]],
            dtype=torch.long
        )

        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[:, -1, :].squeeze(0)

        # Repetition penalty
        next_token_logits = apply_repetition_penalty(
            next_token_logits,
            generated_ids,
            penalty=config.REPETITION_PENALTY
        )

        # Temperature
        next_token_logits = next_token_logits / temperature

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        generated_ids.append(next_token)
        if next_token == eos_id:
             break

    return tokenizer.decode(generated_ids)



def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    for token_id in set(generated_ids):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty

    return logits

def top_k_filter(logits, k=config.TOP_K):
    values, indices = torch.topk(logits, k)

    filtered_logits = torch.full_like(
        logits,
        float("-inf")
    )

    filtered_logits[indices] = values

    return filtered_logits

# -----------------------------
# Test
# -----------------------------

prompt = "Artificial intelligence"

result = generate(
    prompt,
    max_new_tokens=20,
    temperature=0.8
)

if __name__ == "__main__":
    print("Tiny Mistral Chat")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ")

        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        output = generate(prompt)

        print("Model:", output)
        print()
