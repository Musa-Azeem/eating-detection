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
    
class AccRawDatasetPartitioned(Dataset):
    def __init__(self, X, winsize):
        super().__init__()
        self.winsize = winsize
        self.X = X
        if len(X) % winsize != 0:
            raise ValueError("Winsize must be a factor of the dataset length.")
        self.X = X.view(-1, winsize, 3)

    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError("Index Out of Range")
        
        return self.X[i].T.flatten()
    
    def __len__(self):
        return len(self.X)
    
class AccAndLabelsDataset(Dataset):
    def __init__(self, X, y, winsize):
        super().__init__()
        self.winsize = winsize
        self.X = X
        self.y = y
    
    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError("Index Out of Range")

        return (self.X[i:i+self.winsize].T.flatten(), self.y[i])
    
    def __len__(self):
        return len(self.y)
    
class FiveClassDataset(Dataset):
    def __init__(self, X, y, winsize):
        super().__init__()
        self.winsize = winsize
        self.X = X
        self.y = y

    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError("Index Out of Range")

        return (self.X[i:i+self.winsize].T.flatten(), self.y[i])
    
    def __len__(self):
        return len(self.y)