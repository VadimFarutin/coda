from torch.utils.data.dataset import Dataset


class DataWithLabelsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(X)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.X[item], self.y[item]
