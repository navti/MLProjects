import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
import pickle
import re

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer
    """
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.special_keys = ["[PAD]", "[SOS]", "[EOS]", "[CLS]", "[SEP]", "[MASK]"]
        self.special_tokens = {}

    def get_stats(self, token_ids):
        """
        Get frequency of token pairs
        :param token_ids: token_ids[list]
        :return:
            pair_counts: dict containing token pair keys and their frequencies
        """
        pair_counts = defaultdict(int)
        for pair in zip(token_ids, token_ids[1:]):
            pair_counts[pair] += 1
        return pair_counts

    def encode(self, document):
        """
        Encode the document into token ids. Split document into sub documents
        on the special tokens. Then encode those sub documents into tokens and
        return final joined token list.
        :param document: type string
        :return:
            result: list of token ids.
        """
        # store start and end indices of special tokens in a dict st
        st = {}
        for k in self.special_keys:
            it = re.finditer(re.escape(k), document)
            for m in it:
                st[(m.start(), m.end())] = k
        # sort by start index
        st = dict(sorted(st.items()))
        # if no special tokens, encode the original document directly
        if len(st.keys()) == 0:
            return self._encode(document)

        # form splits with items being either string docs or special token id
        splits = []
        head = 0
        for k in st:
            if k[0] > head:
                splits.append(document[head:k[0]])
            splits.append(self.special_tokens[st[k]])
            head = k[1]
        if head <= len(document)-1:
            splits.append(document[head:])

        # iterate over splits items, encode item if it is str type, else include as is if it's int type
        result = []
        for item in splits:
            if isinstance(item, int):
                result += [item]
            else:
                result += self._encode(item)
        return result

    def _encode(self, document):
        """
        Encode document into token ids. Convert to utf-8 encoding and merge bytes
        using the tokenizer.
        :param document: type str
        :return:
            tokens: list containing merged token ids
        """
        tokens = list(document.encode('utf-8'))
        while True:
            pair_counts = self.get_stats(tokens)
            if len(pair_counts) == 0:
                break
            # get the pair with minimum corresponding value in merges, as that pair must be merged first
            pair = min(pair_counts, key=lambda k: self.merges.get(k, float('inf')))
            if pair in self.merges:
                tokens = self.merge(tokens, pair, self.merges[pair])
            else:
                break
        return tokens

    def decode_raw(self, ids):
        """
        Decode given token ids into raw text (including special tokens)
        :param ids: token ids
        :return: type str, decoded string
        """
        tokens = b''.join(self.vocab[idx] for idx in ids)
        return tokens.decode('utf-8', errors='replace')

    def decode_nice(self, tokens):
        """
        Decode given token ids into text (with special tokens removed)
        :param tokens: token ids
        :return: type str, decoded string
        """
        special_tokens = set(self.special_tokens.values())
        nice_tokens = []
        for token in tokens:
            if token not in special_tokens:
                nice_tokens.append(token)
        return self.decode_raw(nice_tokens)

    def merge(self, tokens, pair, token_id):
        """
        Merge a token pair and return new tokens
        :param tokens: list, token ids
        :param pair: the token id pair that should be merged
        :param token_id: token id that the pair should be merged into
        :return:
            new_tokens: list of new token ids after performng merges
        """
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
        """
        Get frequency of words in the corpus
        :param corpus: list of words
        :return:
            stats: dict containing word frequencies
        """
        stats = defaultdict(int)
        for word in corpus:
            word = list(word.encode('utf-8'))
            word = tuple(word)
            stats[word] += 1
        return stats

    def train(self, train_corpus, num_merges):
        """
        Train the tokenizer
        :param train_corpus: list of words in the corpus
        :param num_merges: no. of merges to be done
        :return: None
        """
        # since it is byte encoding, the vocabulary already has 255 tokens
        last_token_id = 255
        # get the corpus stats, this will be used for efficient merge updates
        cstats = self.get_corpus_stats(train_corpus)
        for _ in tqdm(range(num_merges)):
            pair_count = defaultdict(int)
            # build pair_count using cstats, don't have to go over repeated words
            for key, val in cstats.items():
                for i in range(1, len(key)):
                    pair = (key[i-1], key[i])
                    pair_count[pair] += val
            if pair_count:
                # find pair with max frequency
                top_pair = max(pair_count, key=pair_count.get)
                # new merge token id
                merged_token_id = last_token_id  + 1
                last_token_id = merged_token_id
                # register this new toke in the merges dict
                self.merges[top_pair] = merged_token_id
                new_cstats = {}
                # iterate over cstats keys and perform merges
                for key, val in cstats.items():
                    key = self.merge(key, top_pair, merged_token_id)
                    new_cstats[tuple(key)] = val
                # update cstats with new cstats
                cstats = new_cstats
            else:
                break
        # build vocabulary
        vocab = {key: bytes([key]) for key in range(256)}
        for (a,b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        self.vocab = vocab
        # add special tokens at the end of the vocabulary
        for i,token in enumerate(self.special_keys):
            self.vocab[last_token_id+1+i] = bytes(list(token.encode('utf-8')))
            self.special_tokens[token] = last_token_id + 1 + i

    def save(self, filename):
        """
        Save tokenizer to disk in .pkl format
        :param filename: save file path
        :return: None
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

def load_tokenizer(filename):
    """
    Load tokenizer from disk in .pkl format
    :param filename: tokenizer pkl file path
    :return: tokenizer
    """
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def build_corpus(sents):
    """
    Build corpus from given sentences
    :param sents: list of sentences
    :return:
        words: list of words
    """
    # join sentences and convert case to lower
    corpus = ' '.join(sents).lower()
    # remove punctuation, they will be separate tokens
    punctuation_marks = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    corpus = re.sub(f"[{re.escape(punctuation_marks)}]", '', corpus)
    words = [' '+word for word in corpus.split()]
    return words

def df_to_corpus(df):
    """
    Build corpus from given pandas dataframe
    :param df: pandas dataframe with "english" and "hindi" columns containing sentences
    :return:
        e_cor: English corpus, list of words
        h_cor: Hindi corpus, list of words
    """
    english_sentences = list(df['english'].values)
    hindi_sentences = list(df['hindi'].values)
    e_cor = build_corpus(english_sentences)
    h_cor = build_corpus(hindi_sentences)
    return e_cor, h_cor

# test
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
