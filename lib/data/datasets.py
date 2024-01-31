from torch.utils.data import Dataset
import torch

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

class AccRawDatasetStrided(Dataset):
    def __init__(self, X, winsize, stride):
        super().__init__()
        self.winsize = winsize
        self.stride = stride
        self.X = X
    
    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError("Index Out of Range")

        return self.X[i*self.stride:i*self.stride+self.winsize].T.flatten()

    def __len__(self):
        return (len(self.X) - self.winsize) // self.stride + 1
    
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
    
class MultiClassDataset(Dataset):
    def __init__(self, X, y, winsize, window=True):
        super().__init__()
        self.window = window
        self.winsize = winsize
        self.X = X
        self.y = y

        if window:
            self.len = len(self.y)
        else:
            if len(X) % winsize != 0:
                raise ValueError("Winsize must be a factor of the dataset length.")
            self.X = X.view(-1, winsize, 3)
            self.len = len(self.X)

    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError("Index Out of Range")

        if self.window:
            return (self.X[i:i+self.winsize].T.flatten(), self.y[i])
        else:
            return (self.X[i].T.flatten(), torch.mode(self.y[i*self.winsize:(i+1)*self.winsize])[0])
    
    def __len__(self):
        return self.len

DATA_DIR = f'{os.path.expanduser("~")}/.delta/nursing_pt'
class WindowedDatasetWithStrideAndModeOfLabel(torch.utils.data.Dataset):
    def __init__(self, nurse, windowsize=1, stride=1):
        self.windowsize = windowsize
        self.stride = stride
        self.channels = 3
        self.X,self.y = torch.load(f'{DATA_DIR}/{nurse}.pt')
        self.len = math.ceil(len(self.X)/self.stride)
        
        self.X = torch.cat([self.X,torch.zeros(self.windowsize-1,3)])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (
            self.X[(idx*self.stride):(idx*self.stride)+self.windowsize].transpose(0,1),
            self.y[(idx*self.stride):(idx*self.stride)+self.windowsize].mode().values
        )