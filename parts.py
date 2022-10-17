import torch
from os import path
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import time
import numpy as np
from copy import deepcopy

def recycle(iterable): #stolen from Google Brain Big Transfer
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i

def accuracy(output, target, topk=(1,)): #from pytorch/references/classification/utils.py
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():#same as the old no_grad context manager
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
      
def run_eval(args, model, step, stats, device, data_loader, split): #also largely from Google Brain Team Big Transfer
  #setup
  all_c, all_top1, all_top2 = [], [], []
  nb_classes = len(data_loader.dataset.classes)
  confusion_matrix = torch.zeros(nb_classes, nb_classes)
  end = time.perf_counter()

  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      logits = model(x)
      c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)

      #get stats
      top1, top2 = accuracy(logits, y, topk=(1, 2))

      #group stats and move to cpu
      all_c.extend(c.cpu().numpy())  # Also ensures a sync point.
      all_top1.append(top1.cpu().numpy())
      all_top2.append(top2.cpu().numpy())

      #this section attributed to @ptrblck on pytorch forums
      _, preds = torch.max(logits, 1)
      for t, p in zip(y.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
  
  per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
  stats[f'{split}_mca'][step] = 100.0*per_class.mean().numpy()
  stats[f'{split}_acc'][step] = np.mean(all_top1)
  stats[f'{split}_loss'][step] = np.mean(all_c)

  #close
  time_taken = time.perf_counter() - end
  if args.verbose:
    print(f"{split.capitalize()}@{step}: loss {np.mean(all_c):.5f} : top1 {np.mean(all_top1):.2f}% : top2 {np.mean(all_top2):.2f}% : mca {100.0*per_class.mean().numpy():.2f}% : took {time_taken:.2f} seconds")
    print(f"Accuracies per class is {per_class.numpy()}")
  return stats
  
#train_loader, valid_loader, test_loader, train_set, valid_set, test_set = parts.mktrainval(args, preprocess)

def mktrainval(args, preprocess):

  if args.random_rotate:
    preprocess = T.Compose([
          deepcopy(preprocess),
          T.RandomRotation(degrees=180, interpolation=T.InterpolationMode.BILINEAR)
            ])
  
  if args.random_flip:
    preprocess = T.Compose([
          deepcopy(preprocess),
          T.RandomHorizontalFlip()
            ])
  if args.verbose:
    print(preprocess)
  
  train_set = ImageFolder(path.join(args.datadir,'training'),transform=preprocess)
  valid_set = ImageFolder(path.join(args.datadir,'validation'),transform=preprocess)
  test_set = ImageFolder(path.join(args.datadir,'test'),transform=preprocess)
  
  train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
  valid_loader = DataLoader(valid_set, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
  test_loader = DataLoader(valid_set, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
  
  return train_loader, valid_loader, test_loader, train_set, valid_set, test_set
