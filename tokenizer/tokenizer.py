from tokenizers import Tokenizer
from tokenizers import pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder



def create_tokenizer(text_file, vocab_size=1000):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=True
)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<unk>",
            "<pad>",
            "<bos>",
            "<eos>"
        ]
    )

    tokenizer.train([text_file], trainer)

    return tokenizer