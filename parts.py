import torchvision
import torch
from torch.utils.data import Dataset

def recycle(iterable): #stolen from Google Brain Big Transfer
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i
      
def run_eval(args, data_loader):
  raise(NotImplementedError)

  
def project2set(Dataset):
  def __init__(self, args, preprocess):
    raise(NotImplementedError)
    self.imgs
    self.classes
    self.preprocess
    
  def __getitem__(self, item):
    return img, class

  def __len__(self):
    return len(self.imgs)
  
#train_loader, valid_loader, train_set, valid_set = parts.mktrainval(args, preprocess)

def mktrainval(args, preprocess):
  raise(NotImplementedError)
  
  .unsqueeze(0)
