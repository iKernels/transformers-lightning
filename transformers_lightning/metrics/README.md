# Metrics

Metrics are compatible with the `pytorch_lightning.metrics.Metric` class, that is, they automatically manage an internal state to progressively recompute the output value when new data are added. This is a useful behaviour to increasingly update the metric value while doing step after step in the training phase.

At the moment, only metrics for information retrieval tasks are given since most common metrics such as `Accuracy` and `F1 score` are already available in the `pytorch_lightning` repository.

For the general structure and behaviour of a `pytorch_lightning.metrics.Metric`, check out the [original documentation](https://pytorch-lightning.readthedocs.io/en/stable/metrics.html).


## Information Retrieval

Since information retrieval metric must work on many predictions at a time, we introduce a new way of grouping predictions and labels by providing an additional tensor containing the index of the document. Basically, predictions belonging to the same document should have the same `idx`. `Indexes` are not inevitably continuous because they will be only used to group predictions about the same document together. The `indexes` are the first argument of each of the following metrics.

### MeanReciprocalRank

Compute the MRR score over a set of documents.

Usage example:
```python
>>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
>>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
>>> target = torch.tensor([False, False, True, False, True, False, False])

>>> mrr = MeanReciprocalRank()
>>> mrr(indexes, preds, target)
>>> mrr.compute()
... 0.75
```

### MeanAveragePrecision

Compute the MAP score over a set of documents.

Usage example:
```python
>>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
>>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
>>> target = torch.tensor([False, False, True, False, True, False, False])

>>> map = MeanAveragePrecision()
>>> map(indexes, preds, target)
>>> map.compute()
... 0.75
```


### PrecisionAtK

Compute the Precision @ K over a set of documents.

Usage example:
```python
>>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
>>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
>>> target = torch.tensor([False, False, True, False, True, False, False])

>>> p_k = PrecitionAtK(k=1)
>>> p_k(indexes, preds, target)
>>> p_k.compute()
... 0.5
```


### RecallAtK

Compute the Recall @ K over a set of documents.

Usage example:
```python
>>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
>>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
>>> target = torch.tensor([False, False, True, False, True, False, False])

>>> r_k = RecallAtK(k=1)
>>> r_k(indexes, preds, target)
>>> r_k.compute()
... 0.5
```



### HitRateAtK

Compute the Hit Rate @ K over a set of documents.

Usage example:
```python
>>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
>>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
>>> target = torch.tensor([False, False, True, False, True, False, False])

>>> hr_k = HitRateAtK(k=1)
>>> hr_k(indexes, preds, target)
>>> hr_k.compute()
... 0.5
```


