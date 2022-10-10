import torchvision
import torch

def recycle(iterable): #stolen from Google Brain Big Transfer
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i
      
def run_eval(args, data_loader):
  raise(NotImplementedError)
