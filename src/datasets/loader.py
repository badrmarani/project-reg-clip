from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def stratified_sampling(dataset: Dataset, on_labels: bool):
    if on_labels:
        weights = len(dataset) / dataset.label_counts
        weights = weights[dataset.labels.long()]
    else:
        weights = len(dataset) / dataset.group_counts
        weights = weights[dataset.groups.long()]

    sampler = WeightedRandomSampler(
        weights,
        len(dataset),
        replacement=True,
    )
    return sampler


def get_loaders(
    dataset: Dataset,
    use_stratified_sampling: bool = True,
    labels: bool = True,
    **kwargs
):
    if dataset.split == 0:
        if use_stratified_sampling:
            sampler = stratified_sampling(dataset, on_labels=labels)
            shuffle = False
        else:
            sampler = None
            shuffle = True
    else:
        sampler = None
        shuffle = False

    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        shuffle=shuffle,
        **kwargs,
    )
