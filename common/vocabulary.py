"""
A Vocabulary maintains a mapping between words and corresponding unique integers, holds special
integers (tokens) for indicating start and end of sequence, and offers functionality to map
out-of-vocabulary words to the corresponding token.
"""

import json
import os
from typing import List, Union

class Vocabulary(object):

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<S>"
    EOS_TOKEN = "</S>"
    UNK_TOKEN = "<UNK>"

    UNK_INDEX = 0
    PAD_INDEX = 1
    SOS_INDEX = 2
    EOS_INDEX = 3

    def __init__(self, word_counts: Union[str,dict], min_count: int = 5):
        super().__init__()
        if type(word_counts) == str:
            if not os.path.exists(word_counts):
                raise FileNotFoundError(f"Word couts fo not exist at {word_counts}")

            with open(word_counts,"r") as word_count_file:
                word_counts = json.load(word_count_file)

        # form a list of (word, count) tuples and apply min_count threshold
        word_counts = [
            (word, count) for word,count in word_counts.items()
            if count >= min_count
        ]

        # sort in descending order of word counts
        word_counts = sorted(word_counts, key=lambda  wc: -wc[1])
        words  = [w[0] for w in word_counts]

        self.word2index = {}
        self.word2index[self.UNK_TOKEN] = self.UNK_INDEX
        self.word2index[self.PAD_TOKEN] = self.PAD_INDEX
        self.word2index[self.SOS_TOKEN] = self.SOS_INDEX
        self.word2index[self.EOS_TOKEN] = self.EOS_INDEX

        for index, word in enumerate(words):
            self.word2index[word] = index + 4
        self.index2word = {index: word for word,index in self.word2index.items()}

    def to_indices(self, words: List[str]) -> List[int]:
        return [self.word2index.get(word, self.UNK_INDEX) for word in words]

    def to_word(self, indices: List[int]) -> List[str]:
        return [self.index2word.get(index, self.UNK_TOKEN) for index in indices]

    def save(self, save_vocabulary_path: str) -> None:
        # save self.word2index to json file
        with open(save_vocabulary_path,"w") as save_vocabulary_file:
            json.dump(self.word2index, save_vocabulary_file)

    def __len__(self):
        return len(self.index2word)