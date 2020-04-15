from torch.utils.data import Dataset


def set_transform(dataset, transform):
    if hasattr(dataset, "dataset"):
        set_transform(dataset.dataset, transform)
    elif isinstance(dataset, Dataset):
        if hasattr(dataset, "transform"):
            dataset.transform = transform
        elif hasattr(dataset, "datasets"):
            for dtst in dataset.datasets:
                set_transform(dtst, transform)
