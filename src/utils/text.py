from typing import List, Optional, Callable, Iterator

import torchtext

from settings import SPECIALS, UNK_ID, VOCAB_CORPUS_FILE_JA, VOCAB_CORPUS_FILE_EN
from _logging import getLogger

logger = getLogger(__name__)


def tokenizer_ja(text: str) -> List[str]:
    return text.strip().split()


def tokenizer_en(text: str) -> List[str]:
    return text.strip().lower().split()


def text_iterator(
    io_stream, tokenizer: Callable[[str], List[str]]
) -> Iterator[List[str]]:
    for line in io_stream:
        yield tokenizer(line)


def load_corpus_ja(filepath: str) -> List[List[str]]:
    with open(filepath, "r") as f:
        corpus_ja = list(text_iterator(f, tokenizer_ja))
    return corpus_ja


def load_corpus_en(filepath: str) -> List[List[str]]:
    with open(filepath, "r") as f:
        corpus_ja = list(text_iterator(f, tokenizer_en))
    return corpus_ja


def text_encoder(
    tokens: List[str],
    vocab: torchtext.vocab.Vocab,
    sentence_length: Optional[int] = None,
    bos: Optional[bool] = False,
    eos: Optional[bool] = False,
) -> List[int]:
    """tokens(単語列)をvocabを用いて, id列に変換する
    Args:
        tokens: 単語リスト
        vocab: 語彙manager
        sentence_length: もしNoneでなければ, 返り値リストの長さがsentence_lengthになるように
                          <pad>でパディングする
        bos: もしTrueなら、先頭に<bos>(begin of sentence)を追加する
        eos: もしTrueなら、末尾に<eos>(end of sentence)を追加する

    Returns:
        indices: tokensに対応するid列
    """
    indices = vocab.lookup_indices(tokens)

    if bos:
        indices = [vocab["<bos>"]] + indices
    if eos:
        indices += [vocab["<eos>"]]
    if sentence_length is not None:
        assert len(indices) <= sentence_length
        indices += [vocab["<pad>"]] * (sentence_length - len(indices))

    return indices


def text_decoder(indices: List[int], vocab: torchtext.vocab.Vocab) -> List[str]:
    """id列を単語列に変換する"""
    tokens = vocab.lookup_tokens(indices)
    return tokens


def create_vocabs(max_tokens):
    with open(VOCAB_CORPUS_FILE_JA, "r") as f:
        vocab_ja = torchtext.vocab.build_vocab_from_iterator(
            iterator=text_iterator(f, tokenizer_ja),
            max_tokens=max_tokens,
            specials=SPECIALS,
            special_first=True,
        )
        vocab_ja.set_default_index(UNK_ID)

    with open(VOCAB_CORPUS_FILE_EN, "r") as f:
        vocab_en = torchtext.vocab.build_vocab_from_iterator(
            iterator=text_iterator(f, tokenizer_en),
            max_tokens=max_tokens,
            specials=SPECIALS,
            special_first=True,
        )
        vocab_en.set_default_index(UNK_ID)

    logger.info(f"size of vocab_ja {len(vocab_ja)}")
    logger.info(f"size of vocab_en {len(vocab_en)}")

    return vocab_ja, vocab_en
