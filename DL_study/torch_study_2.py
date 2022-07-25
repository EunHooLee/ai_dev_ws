
"""
s1 = '나는 책상 위에 사과를 먹었다'
s2 = '알고 보니 그 사과는 Jason 것이었다'
s3 = '그래서 Jason에게 사과를 했다.'

print(s1.split())
print(s2.split())
print(s3.split())

token2idx = {}
index = 0

for sentesnce in [s1,s2,s3]:
    tokens = sentesnce.split()
    for token in tokens:
        if token2idx.get(token) == None:
            token2idx[token] = index
            index +=1
print(token2idx)

def indexed_sentence(sentence):
    return [token2idx[token] for token in sentence]

s1_i = indexed_sentence(s1.split())
s2_i = indexed_sentence(s2.split())
s3_i = indexed_sentence(s3.split())

# print(s1_i)
# print(s2_i)
# print(s3_i)

s4 = '나는 책상 위에 배를 먹었다'
#s4_i = indexed_sentence(s4.split())
print("items: ",token2idx.items())
token2idx = {t : i+1 for t, i in token2idx.items()}
token2idx['<unk>'] = 0
print(token2idx)
def indexed_sentence_unk(sentence):
    return [token2idx.get(token, token2idx['<unk>']) for token in sentence]


s4_i = indexed_sentence_unk(s4.split())
print(s4_i)
"""

# ===============================================================================
"""
idx2char = {0 : '<pad>', 1 : '<unk>'}

srt_idx = len(idx2char)
for x in range(32,127):
    idx2char.update({srt_idx: chr(x)})
    srt_idx +=1

for x in range(int('0x3131', 16), int('0x3163',16) +1):
    idx2char.update({srt_idx : chr(x)})
    srt_idx +=1

for x in range(int('0xAC00',16),int('0xD7A3',16)+1):
    idx2char.update({srt_idx : chr(x)})
    srt_idx +=1

char2idx = {v : k for k, v in idx2char.items()}
print([char2idx.get(c,0) for c in '그래서 Jason에게 사과를 했다'])
print([char2idx.get(c,0) for c in 'ㅇㅋㅇㅋ! ㅋㅋㅋㅋㅋ'])
"""
# ===============================================================================
# Algoritm 1 : BPE

import re, collections

from torch import pairwise_distance

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] +=freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\\S)'+ bigram + r'(?!\\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair),word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w </w>':5, 'l o w e r </w>' :2, 'n e w e s t </w>':6, 'w i d e s t </w>':3}

num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best  = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f'Step {i + 1}')
    print(best)
    print(vocab)
    print('\n')