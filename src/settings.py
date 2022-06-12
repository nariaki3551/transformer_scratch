PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIALS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
]
UNK_ID = 1
PAD_ID = 0

assert SPECIALS[UNK_ID] == UNK_TOKEN
assert SPECIALS[PAD_ID] == PAD_TOKEN

VOCAB_CORPUS_FILE_JA = "../corpus/small_parallel_enja/test.ja"
VOCAB_CORPUS_FILE_EN = "../corpus/small_parallel_enja/test.en"
TRAIN_CORPUS_FILE_JA = "../corpus/small_parallel_enja/test.ja"
TRAIN_CORPUS_FILE_EN = "../corpus/small_parallel_enja/test.en"
TEST_CORPUS_FILE_JA = "../corpus/small_parallel_enja/test.ja"
TEST_CORPUS_FILE_EN = "../corpus/small_parallel_enja/test.en"

GPU = False
