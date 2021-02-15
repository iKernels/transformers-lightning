import torch


def make_ngrams_labels(labels, n, ignore_idx=None):

    if ignore_idx is None:
        grams = labels.repeat(1, n).view(labels.size(0), n, -1)
    else:
        grams = labels[:, 1:].repeat(1, n).view(labels.size(0), n, -1)

    for i in range(1, grams.size(1)):
        grams[:, i] |= torch.roll(grams[:, i - 1], 1)

        if ignore_idx is not None:
            grams[:, i][grams[:, i] < 0] = ignore_idx

    for i in range(1, grams.size(1)):
        grams[:, i, :i] = ignore_idx

    if ignore_idx is not None:
        v = (torch.zeros(grams.size(0) * n, dtype=torch.long) + ignore_idx).view(grams.size(0), n, 1).to(labels.device)
        grams = torch.cat([v, grams], dim=2)

    return grams.permute(0, 2, 1).contiguous()
