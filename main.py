import prior
from tqdm import tqdm


def load_dataset() -> prior.DatasetDict:
    data = {}
    for split, size in [("train", 1000), ("val", 100), ("test", 100)]:
        data[split] = prior.Dataset(
            [i for i in tqdm(range(size))], dataset="test", split="test"
        )
    return prior.DatasetDict(**data)
