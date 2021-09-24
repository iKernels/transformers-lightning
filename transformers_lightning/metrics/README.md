# Metrics

All metrics were integrated successfully in the [torchmetrics](https://github.com/PyTorchLightning/metrics/tree/master/torchmetrics/retrieval) library.
Now this package contains only the new `HitRate@K` metric, which will be likely moved to `torchmetrics` in the next release. `HitRate@K` is the fraction of queries for which at least a positive candidate was found in the first `K` positions.
