import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
import pickle
import re

class BPETokenizer:
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

    def get_corpus_stats(self, corpus):
        stats = defaultdict(int)
        for word in corpus:
            word = list(word.encode('utf-8'))
            word = tuple(word)
            stats[word] += 1
        return stats

    def train(self, train_corpus, num_merges):
        last_token_id = 255
        cstats = self.get_corpus_stats(train_corpus)
        for _ in tqdm(range(num_merges)):
            pair_count = defaultdict(int)
            for key, val in cstats.items():
                for i in range(1, len(key)):
                    pair = (key[i-1], key[i])
                    pair_count[pair] += val
            if pair_count:
                top_pair = max(pair_count, key=pair_count.get)
                merged_token_id = last_token_id  + 1
                last_token_id = merged_token_id
                self._merges[top_pair] = merged_token_id
                new_cstats = {}
                for key, val in cstats.items():
                    key = self.merge(key, top_pair, merged_token_id)
                    new_cstats[tuple(key)] = val
                cstats = new_cstats
            else:
                break
        vocab = {key: bytes([key]) for key in range(256)}
        for (a,b), idx in self._merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        self._vocab = vocab

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def build_corpus(sents):
    corpus = ' '.join(sents).lower()
    punctuation_marks = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    corpus = re.sub(f"[{re.escape(punctuation_marks)}]", '', corpus)
    words = [' '+word for word in corpus.split()]
    return words

def df_to_corpus(df):
    english_sentences = list(df['english'].values)
    hindi_sentences = list(df['hindi'].values)
    e_cor = build_corpus(english_sentences)
    h_cor = build_corpus(hindi_sentences)
    return e_cor, h_cor

if __name__ == "__main__":
    data_dir = '../data'
    save_tokenizer_dir = './saved'
    df = pd.read_csv(f'{data_dir}/train.csv')
    eng_corpus, hindi_corpus = df_to_corpus(df)
    en_tokenizer = BPETokenizer()
    print("Training EN tokenizer...")
    en_tokenizer.train(eng_corpus, num_merges=5000)
    print("EN tokenizer training done. Saving tokenizer on disk...")
    en_tokenizer.save(f'{save_tokenizer_dir}/en_tokenizer.pkl')
    print(f"Tokenizer saved at {save_tokenizer_dir}/en_tokenizer.pkl\n")

    hi_tokenizer = BPETokenizer()
    print("Training HI tokenizer...")
    hi_tokenizer.train(hindi_corpus, num_merges=10000)
    print("HI tokenizer training done. Saving tokenizer on disk...")
    hi_tokenizer.save(f'{save_tokenizer_dir}/hi_tokenizer.pkl')
    print(f"Tokenizer saved at {save_tokenizer_dir}/hi_tokenizer.pkl\n")


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
