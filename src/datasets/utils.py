from torch.utils.data import Dataset, WeightedRandomSampler


def stratified_sampler(dataset: Dataset):
    weights = len(dataset) / dataset.label_counts
    weights = weights[dataset.labels.long()]

    sampler = WeightedRandomSampler(
        weights,
        len(dataset),
        replacement=True,
    )
    return sampler
