import math
import random
import collections

Data = collections.namedtuple("Data", "ja en")


class Dataset:
    def __init__(self, corpus_ja, corpus_en):
        assert len(corpus_ja) == len(corpus_en)
        self.corpus_ja = corpus_ja
        self.corpus_en = corpus_en

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        ja = self.corpus_ja[index]
        en = self.corpus_en[index]
        return Data(ja, en)

    def __len__(self):
        return len(self.corpus_ja)


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def num_batches(self):
        if self.drop_last:
            return math.floor(len(self) / self.batch_size)
        else:
            return math.ceil(len(self) / self.batch_size)

    def __iter__(self):
        iterator_data = list(self.dataset)
        if self.shuffle:
            random.shuffle(iterator_data)

        for i in range(0, len(iterator_data), self.batch_size):
            yield iterator_data[i : i + self.batch_size]

        if not self.drop_last and i < len(iterator_data) - 1:
            yield iterator_data[i:]

    def __len__(self):
        return len(self.dataset)
