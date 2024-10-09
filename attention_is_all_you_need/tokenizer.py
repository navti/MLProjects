import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

def build_corpus(sentences):
    corpus = []
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            word = word.lower() + ' '
            word = list(word)
            corpus.append(word)
    return corpus

def df_to_corpus(df):
    english_sentences = list(df['english'].values)
    hindi_sentences = list(df['hindi'].values)
    e_cor = build_corpus(english_sentences)
    h_cor = build_corpus(hindi_sentences)
    return e_cor, h_cor

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

def build_vocab(corpus, base_vocab, num_merges):
    merges = {}
    vocab = deepcopy(base_vocab)
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
            merged_token = ''.join(top_pair)
            merges[top_pair] = merged_token
            vocab.add(merged_token)
            for idx in range(len(corpus)):
                if idx in word_idx[top_pair]:    
                    word = corpus[idx]
                    new_word = []
                    i = 0
                    while i < len(word):
                        if i < len(word)-1 and (word[i], word[i+1]) == top_pair:
                            new_word.append(merged_token)
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    corpus[idx] = new_word
        else:
            break
    return merges, vocab

df = pd.read_csv('data/train.csv')
eng_corpus, hindi_corpus = df_to_corpus(df)
# save_base_vocab('base_vocab_en.txt', eng_corpus)
# save_base_vocab('base_vocab_hi.txt', hindi_corpus)
# merges, vocab = build_vocab(eng_corpus, 10000)
base_vocab_en = load_base_vocab('base_vocab_en.txt')
merges, vocab = build_vocab(eng_corpus, base_vocab_en, 10)
print('done')