import math
import torch

def ngrams_stats(words, n):
    stats = {}
    for i in range(len(words)-n+1):
        key = tuple(words[i:i+n])
        stats[key] = stats.get(key,0) + 1
    return stats

def clipped_precision(out_words, ref_words, n):
    eps = 1e-10
    out_stats = ngrams_stats(out_words, n)
    ref_stats = ngrams_stats(ref_words, n)
    cp_sum = 0
    for key in out_stats:
        cp_sum += min(ref_stats.get(key, 0), out_stats[key])
    precision = cp_sum/len(out_stats)
    return precision + eps

def brevity_penalty(out, ref):
    r, c = len(ref), len(out)
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-r/c)
    return bp    

def bleu_score(out, ref, n=4):
    out_words = out.split()
    ref_words = ref.split()
    n = min(n, len(out_words))
    if n == 0:
        return 0
    weight = 1/n
    cp = 0
    for i in range(1,n+1):
        cp += math.log(clipped_precision(out_words, ref_words, i))
    precision = math.exp(weight * cp)
    bp = brevity_penalty(out_words, ref_words)
    score = bp * precision
    return score

def get_batch_bleu_score(out_tokens, targets, tokenizer):
    batch_size = out_tokens.shape[0]
    bscore = 0
    for out, ref in zip(out_tokens, targets):
        out = tokenizer.decode_nice(out.tolist())
        ref = tokenizer.decode_nice(ref.tolist())
        bscore += bleu_score(out, ref)
    return bscore / batch_size

def process_sentence(sent, context_len, tokenizer):
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
