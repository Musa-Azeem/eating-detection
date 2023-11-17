from torch.utils.data import Dataset

class AccRawDataset(Dataset):
    def __init__(self, X, winsize):
        super().__init__()
        self.winsize = winsize
        self.X = X
    
    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError("Index Out of Range")

        return self.X[i:i+self.winsize].T.flatten()
    
    def __len__(self):
        return len(self.X) - self.winsize