import math

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
    weight = 1/n
    cp = 0
    for i in range(1,n+1):
        cp += math.log(clipped_precision(out_words, ref_words, i))
    precision = math.exp(weight * cp)
    bp = brevity_penalty(out_words, ref_words)
    score = bp * precision
    return score
