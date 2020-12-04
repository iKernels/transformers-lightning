# Language modeling

Language modeling is the art of modifying an input sentence to automatically generate tamperings and corresponding labels. Thos modification are then used to learn language models about languages structure. It is very similar to ho human learn languages by doing exercises where they predict missing words or tries to find errors in sentences.

All language modeling techniques accept two special arguments: `weights` and `whole_word_masking/swapping`. The first allow the user to provide a weights vectors with the same size of the vocabulary to weight probabilities accordingly. The second parameter enforces modifications at the `word` level instead of the `token` one. 

## Masked Language Modeling

Base language model used by most SOTA transformers-based models. It consists in selecting a random 15% of the input tokens and applying the following modifications:

- substitute the token with a mask `[MASK]` token the 80% of times.
- randomly swap a token with another one the 10% of times.
- leave the token as it is in the remaining 10% of times.

The labels are created such that the model should predict the original token id of masked tokens.

Usage example:

```python
>>> import torch
>>> from transformers import BertTokenizer

>>> tok = BertTokenizer.from_pretrained('bert-base-cased')
>>> mlm = MaskedLanguageModeling(tok, whole_word_masking=True)

>>> input_ids = torch.tensor([tok.encode("test sentence")])
>>> input_ids
... tensor([[ 101, 2774, 5650, 102]])

>>> masked, labels = mlm(input_ids)
>>> masked
... tensor([[ 101, 103, 5650, 102]]) # 103 mask token id
>>> labels
... tensor([[-100, 2774, -100, -100]]) # -100 = IGNORE_IDX
```
