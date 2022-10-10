import torchvision
import torch
from torch.utils.data import ImageFolder, SubsetRandomSampler, DataLoader

def recycle(iterable): #stolen from Google Brain Big Transfer
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i
      
def run_eval(args, model, step, stats, data_loader):
  raise(NotImplementedError)
  
#train_loader, valid_loader, test_loader, train_set, valid_set, test_set = parts.mktrainval(args, preprocess)

def mktrainval(args, preprocess):
  
  train_set = ImageFolder(pjoin(args.datadir,'training'),transform=preprocess)
  valid_set = ImageFolder(pjoin(args.datadir,'validation'),transform=preprocess)
  test_set = ImageFolder(pjoin(args.datadir,'test'),transform=preprocess)
  
  train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
  valid_loader = DataLoader(valid_set, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
  test_loader = DataLoader(valid_set, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
  
  raise(NotImplementedError)
