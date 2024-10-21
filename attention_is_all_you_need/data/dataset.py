import torch
from torch.utils.data import Dataset
import pandas as pd

class TranslateSet(Dataset):
    def __init__(self, csv_file_path, context_len, en_tokenizer, hi_tokenizer):
        super(TranslateSet, self).__init__()
        df = pd.read_csv(csv_file_path)
        self.en_sentences = list(df['english'].values)
        self.hi_sentences = list(df['hindi'].values)
        self.en_tokenizer = en_tokenizer
        self.hi_tokenizer = hi_tokenizer
        self.context_len = context_len

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        input_sent = self.en_sentences[idx]
        target_sent = self.hi_sentences[idx]
        input_tokens, _ = self._preprocess_sentence(input_sent, self.context_len, self.en_tokenizer)
        target_tokens, target_pad_len = self._preprocess_sentence(target_sent, self.context_len + 1, self.hi_tokenizer)
        return input_tokens, target_tokens, target_pad_len

    def _preprocess_sentence(self, sent, context_len, tokenizer):
        tokens = tokenizer.encode(sent)
        sos = [tokenizer.special_tokens["[SOS]"]]
        eos = [tokenizer.special_tokens["[EOS]"]]
        pad = [tokenizer.special_tokens["[PAD]"]]
        pad_len = 0
        if len(tokens) >= context_len-2:
            tokens = sos + tokens[:context_len-2] + eos
        else:
            pad_len = context_len - len(tokens) - 2
            tokens = sos + tokens + eos + pad*pad_len
        pad_mask = torch.ones(context_len)
        if pad_len > 0:
            pad_mask[-pad_len:] = torch.zeros(pad_len)
        return torch.tensor(tokens), pad_mask
