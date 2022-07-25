import torch as th

from torchtext import data
from torchtext import datasets


# TEXT = data.Field(lower=True, batch_first=True)
# LABEL = data.Field(sequential=False)

# train, test = datasets.IMDB.splits(TEXT, LABEL)

s1 = 'I always love my parents'
print(s1.split())