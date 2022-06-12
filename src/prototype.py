import time
import argparse
import statistics

from utils.np import *
from torchtext.data.metrics import bleu_score
import tqdm

import settings
import nn.layers
import nn.optimizers
import nn.transformers
import utils.data as data_utils
import utils.text as text_utils
from _logging import getLogger, setLevel

logger = getLogger(__name__)


def main():
    max_tokens = args.max_tokens if args.max_tokens > 0 else None
    sentence_length = args.sentence_length

    # create vocaburary and dataloader
    vocab_ja, vocab_en = text_utils.create_vocabs(max_tokens)
    data_loader, test_corpus = create_data_loader(args.batch_size, vocab_ja, vocab_en)

    # create transformer (encoder and decoder)
    transformer = nn.transformers.Transformer(
        vocab_ja,
        vocab_en,
        args.embed_dim,
        args.feed_forward_dim,
        args.num_heads,
        sentence_length,
    )

    # create loss function and optimizer
    loss_fn = nn.layers.CrossEntropyLoss()
    optimizer = nn.optimizers.Adam(transformer.params, transformer.grads)

    def padding(indices, vocab):
        return np.array(
            indices + [vocab[settings.PAD_TOKEN]] * (sentence_length - len(indices)),
            dtype=np.int64,
        )

    # define progressbar
    if args.batch_progress:
        iter_wrapper = lambda x: tqdm.tqdm(x, total=data_loader.num_batches())
    else:
        iter_wrapper = lambda x: x

    # main train loop
    start_time = time.time()
    for _ in range(args.epoch):

        loss_in_epoch = list()

        for batch in iter_wrapper(data_loader):
            batch_size = len(batch)
            sentence_ja_array = np.zeros((batch_size, sentence_length), dtype=np.int64)
            sentence_en_array = np.zeros((batch_size, sentence_length), dtype=np.int64)
            target_word_array = np.zeros((batch_size, sentence_length), dtype=np.int64)

            for i in range(batch_size):
                sentence_ja_array[i] = padding(batch[i].ja[:-1], vocab_ja)
                sentence_en_array[i] = padding(batch[i].en, vocab_en)
                target_word_array[i] = padding(batch[i].ja[1:], vocab_ja)

            output = transformer.forward(sentence_en_array, sentence_ja_array)
            loss = loss_fn.forward(
                output.reshape(
                    -1, len(vocab_ja)
                ),  # (batch_size * sentence_length, vocab_size)
                target_word_array.reshape(
                    -1,
                ),  # (batch_size * sentence_length, )
                ignore_index=settings.PAD_ID,
            )

            dout = loss_fn.backward()
            dout = dout.reshape(batch_size, sentence_length, len(vocab_ja))
            transformer.backward(dout)

            optimizer.update()
            optimizer.zero_grad()

            loss_in_epoch.append(loss.item())

        print(
            f"epoch {_:5d}, "
            f"loss {statistics.mean(loss_in_epoch):.5f}, "
            f"time {time.time()-start_time:.2f}s"
        )

        if (_ + 1) % args.translate_test_interval == 0:
            # translation test using training model
            bleu_scores = list()
            for sentence_ja, sentence_en in zip(*test_corpus):
                bleu_score = translater_test(
                    sentence_ja,
                    sentence_en,
                    transformer,
                    vocab_ja,
                    vocab_en,
                    sentence_length,
                )
                bleu_scores.append(bleu_score)
            print(
                f"epoch {_:5d}, "
                f"bleu {statistics.mean(bleu_scores):.5f}, "
                f"time {time.time()-start_time:.2f}s"
            )
            optimizer.zero_grad()


def create_data_loader(batch_size, vocab_ja, vocab_en):

    # load corpus for training
    train_corpus_ja = text_utils.load_corpus_ja(settings.TRAIN_CORPUS_FILE_JA)
    train_corpus_en = text_utils.load_corpus_en(settings.TRAIN_CORPUS_FILE_EN)

    # load corpus for test
    if settings.TRAIN_CORPUS_FILE_JA == settings.TEST_CORPUS_FILE_JA:
        num_test_sentences = 10
        test_corpus_ja = train_corpus_ja[:num_test_sentences]
        test_corpus_en = train_corpus_en[:num_test_sentences]
        train_corpus_ja = train_corpus_ja[num_test_sentences:]
        train_corpus_en = train_corpus_en[num_test_sentences:]
    else:
        test_corpus_ja = text_utils.load_corpus_ja(settings.TEST_CORPUS_FILE_JA)
        test_corpus_en = text_utils.load_corpus_en(settings.TEST_CORPUS_FILE_EN)

    logger.info(
        f"max length of japanes train sentense {max(map(len, train_corpus_ja))}"
    )
    logger.info(
        f"max length of english train sentense {max(map(len, train_corpus_en))}"
    )
    logger.info(f"max length of japanes test sentense {max(map(len, test_corpus_ja))}")
    logger.info(f"max length of english test sentense {max(map(len, test_corpus_en))}")

    # create Dataset/DataLoader
    dataset_corpus_ja = list()
    dataset_corpus_en = list()

    for corpus_ja, corpus_en in zip(train_corpus_ja, train_corpus_en):
        encoded_corpus_ja = text_utils.text_encoder(
            corpus_ja, vocab_ja, bos=True, eos=True
        )
        encoded_corpus_en = text_utils.text_encoder(corpus_en, vocab_en)
        dataset_corpus_ja.append(encoded_corpus_ja)
        dataset_corpus_en.append(encoded_corpus_en)

    dataset = data_utils.Dataset(
        dataset_corpus_ja,
        dataset_corpus_en,
    )

    data_loader = data_utils.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    return data_loader, (test_corpus_ja, test_corpus_en)


def translater_test(
    sentence_ja, sentence_en, transformer, vocab_ja, vocab_en, sentence_length
):
    if not args.quiet:
        covered_sentence_ja = text_utils.text_decoder(
            text_utils.text_encoder(sentence_ja, vocab_ja), vocab_ja
        )
        covered_sentence_en = text_utils.text_decoder(
            text_utils.text_encoder(sentence_en, vocab_en), vocab_en
        )
        print("[ja]", "".join(covered_sentence_ja))
        print("[en]", " ".join(covered_sentence_en))

    sentence_en_array = np.array(
        [text_utils.text_encoder(sentence_en, vocab_en, sentence_length)],
        dtype=np.int64,
    )

    sentence = [settings.BOS_TOKEN]
    index = 0

    while len(sentence) < sentence_length:
        sentence_jp_array = np.array(
            [text_utils.text_encoder(sentence, vocab_ja, sentence_length)],
            dtype=np.int64,
        )
        output = transformer.forward(
            sentence_en_array, sentence_jp_array
        )  # (1, sentence_length, vocab_size)
        word_id = output[0, index].argmax()
        word = vocab_ja.lookup_token(word_id)
        sentence.append(word)
        if word == settings.EOS_TOKEN:
            break
        index += 1

    if not args.quiet:
        print("translate", sentence, "\n")
    candidate_sentence = [word for word in sentence if word not in settings.SPECIALS]
    references_sentence = [
        word for word in sentence_en if word not in settings.SPECIALS
    ]
    return bleu_score([sentence], [[references_sentence]])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="number of epoch",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="size of batch",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="max size of tokens in vocabulary",
    )
    parser.add_argument(
        "--sentence_length",
        type=int,
        default=18,
        help="length of sentence",
    )
    parser.add_argument(
        "--translate_test_interval",
        type=int,
        default=10,
        help="execute translatioi test of trained model per interval epoch",
    )
    parser.add_argument(
        "--batch_progress",
        action="store_true",
        help="display progress in epoch",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="quiet for testing sentence",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=16,
        help="dimension of embedding layer",
    )
    parser.add_argument(
        "--feed_forward_dim",
        type=int,
        default=32,
        help="dimension of hidden layer of feed forward",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="number of heads of MultiHeadAttention",
    )

    args = parser.parse_args()
    print("args", args)

    setLevel(10)
    main()
