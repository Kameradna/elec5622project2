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
  
#train_loader, valid_loader, train_set, valid_set = parts.mktrainval(args, preprocess)

def mktrainval(args, preprocess):
  raise(NotImplementedError).unsqueeze(0)
