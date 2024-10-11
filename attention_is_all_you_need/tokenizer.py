import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

class BPETokenzer:
    def __init__(self):
        self._merges = {}
        self._decode_vocab = {}
        self._vocab = {}

    def get_stats(self, token_ids):
        pair_counts = defaultdict(int)
        for pair in zip(token_ids, token_ids[1:]):
            pair_counts[pair] += 1
        return pair_counts

    def encode(self, document):
        tokens = list(document.encode('utf-8'))
        while True:
            pair_counts = self.get_stats(tokens)
            pair = min(pair_counts, key=lambda k: self._merges.get(k, float('inf')))
            if pair in self._merges:
                tokens = self.merge(tokens, pair, self._merges[pair])
            else:
                break
        return tokens

    def decode(self, ids):
        tokens = b''.join(self._vocab[idx] for idx in ids)
        return tokens.decode('utf-8', errors='replace')

    def merge(self, tokens, pair, token_id):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(token_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, train_corpus, num_merges):
        corpus = deepcopy(train_corpus)
        last_token_id = 255
        for _ in tqdm(range(num_merges)):
            pair_count = defaultdict(int)
            word_idx = defaultdict(set)
            for idx, word in enumerate(corpus):
                for i in range(1, len(word)):
                    pair = (word[i-1], word[i])
                    pair_count[pair] += 1
                    word_idx[pair].add(idx)
            if pair_count:
                top_pair = max(pair_count, key=pair_count.get)
                merged_token_id = last_token_id + 1
                last_token_id = merged_token_id
                self._merges[top_pair] = merged_token_id
                for idx in range(len(corpus)):
                    if idx in word_idx[top_pair]:
                        word = corpus[idx]
                        corpus[idx] = self.merge(word, top_pair, merged_token_id)
            else:
                break
        vocab = {key: bytes([key]) for key in range(256)}
        for (a,b), idx in self._merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        self._vocab = vocab

def build_corpus(sentences):
    corpus = []
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            word = word.lower() + ' '
            word = word.encode('utf-8')
            word = list(map(int, word))
            corpus.append(word)
    return corpus

def df_to_corpus(df):
    english_sentences = list(df['english'].values)
    hindi_sentences = list(df['hindi'].values)
    e_cor = build_corpus(english_sentences)
    h_cor = build_corpus(hindi_sentences)
    return e_cor, h_cor

df = pd.read_csv('data/train.csv')
eng_corpus, hindi_corpus = df_to_corpus(df)
en_tokenizer = BPETokenzer()
en_tokenizer.train(eng_corpus, 10)
print('done')



"""
# for character based pairing and encoding, not used in BPE.
def save_base_vocab(name, corpus):
    base_vocab = set()
    for word in corpus:
        base_vocab = base_vocab.union(set(word))
    with open(name, 'w') as f:
        for ch in base_vocab:
            f.write(ch)

def load_base_vocab(name):
    base_vocab = set()
    with open(name, 'r') as f:
        for ch in f.read():
            base_vocab.add(ch)
    return base_vocab

def save_merges(name, merges):
    with open(name, 'w') as f:
        for item in merges.items():
            f.write(str(item)+'\n')

def load_merges(name):
    merges = {}
    with open('merges.txt', 'r') as f:
        for line in f.readlines():
            item = eval(line)
            key = item[0]
            val = item[1]
            merges[key] = val
    return merges
"""
