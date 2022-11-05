import torch
import os
from os import path
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision.datasets import ImageFolder, FashionMNIST
import torchvision.transforms as T
import time
import numpy as np
from copy import deepcopy
import argparse
import logging
import logging.config

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
      
def run_eval(args, model, step, stats, data_loader, split, logger, device): #also largely from Google Brain Team Big Transfer
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
    logger.info(f"{split.capitalize()}@{step}: loss {np.mean(all_c):.5f} : top1 {np.mean(all_top1):.2f}% : top2 {np.mean(all_top2):.2f}% : mca {100.0*per_class.mean().numpy():.2f}% : took {time_taken:.2f} seconds")
    logger.info(f"Accuracies per class is {per_class.numpy()}")
  return stats
  
def logstats(args, stats, logger, step, best_step, time_taken):
  if best_step is None: #if this is the first epoch
    logger.info(f"Stats@{step}: valid loss {stats['valid_loss'][step]:.5f}, \
valid accuracy {stats['valid_acc'][step]:.2f}% \
valid mca {stats[f'valid_mca'][step]:.2f}%")
    return

  if time_taken == 'end':#if this is the final testing step
    logger.info(f"Test stats@{step}: test loss {stats['test_loss'][step]:.5f}, \
test accuracy {stats['test_acc'][step]:.2f}% \
test per-class mean accuracy {stats[f'test_mca'][step]:.2f}%")
    return

  if args.training_stats:
    logger.info(f"Stats@{step}: train loss {stats['train_loss'][step]:.5f}, \
train accuracy {stats['train_acc'][step]:.2f}%, \
valid loss {stats['valid_loss'][step]:.5f}, \
valid accuracy {stats['valid_acc'][step]:.2f}% \
valid mca {stats[f'valid_mca'][step]:.2f}% \
learning rate {stats['lr'][step]:.8f} \
eff batch size {int(args.batch_split*stats['batch_size'][step])} \
time taken avg/step {time_taken:.2f}s \
{'(best)' if best_step == step else ''}"
                        )
  else:
    logger.info(f"Stats@{step}: train loss {stats['train_loss'][step]:.5f}, \
valid loss {stats['valid_loss'][step]:.5f}, \
valid accuracy {stats['valid_acc'][step]:.2f}% \
valid mca {stats[f'valid_mca'][step]:.2f}% \
learning rate {stats['lr'][step]:.8f} \
eff batch size {int(args.batch_split*stats['batch_size'][step])} \
time taken avg/step {time_taken:.2f}s \
{'(best)' if best_step == step else ''}"
                        )


def mktrainval(args, preprocess, logger):
  if args.dataset == 'hep2':
    valid_set = ImageFolder(path.join(args.datadir,'validation'),transform=preprocess)
    test_set = ImageFolder(path.join(args.datadir,'test'),transform=preprocess)
  elif args.dataset == 'fashion':
    preprocess = T.Compose([
          T.Grayscale(num_output_channels=3),
          deepcopy(preprocess)
            ])
    valid_set = FashionMNIST(path.join(args.datadir),train=True,transform=preprocess,download=True)
    test_set = None
    test_loader = None
  else:
    logger.critical(f"{args.dataset} not implemented, please check spelling or implement it yourself. Exiting...")
    exit()


  valid_loader = DataLoader(valid_set, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)
  test_loader = DataLoader(valid_set, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

  if args.aggressive:
    preprocess = T.Compose([
          T.AugMix(),
          deepcopy(preprocess)
            ])


  if args.gaussian:
    preprocess = T.Compose([
          T.GaussianBlur(kernel_size=3, sigma=(0.1,2)),
          deepcopy(preprocess)
          
            ])

  if args.random_rotate:
    preprocess = T.Compose([
          T.RandomRotation(degrees=180, interpolation=T.InterpolationMode.BILINEAR),
          deepcopy(preprocess)
            ])
  
  if args.random_flip:
    preprocess = T.Compose([
          T.RandomHorizontalFlip(),
          deepcopy(preprocess)
            ])
  if args.verbose:
    logger.debug(preprocess)
  
  if args.dataset == 'hep2':
    train_set = ImageFolder(path.join(args.datadir,'training'),transform=preprocess)
    train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
  else:
    train_set = FashionMNIST(path.join(args.datadir),train=True,transform=preprocess,download=True)
    train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

  return train_loader, valid_loader, test_loader, train_set, valid_set, test_set

def parseargs():
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", default='unnamed', required=False,
                    help="Name for this run, set by default by other args, but feel free to identify a specific run by name")
  parser.add_argument("--datadir", default="data", required=False,
                    help="Path to the data folder, preprocessed for torchvision.")
  parser.add_argument("--savedir", default="save", required=False,
                    help="Path to the save folder, for placement of csv and pth files.")
  parser.add_argument("--loaddir", default="save", required=False,
                    help="Path to the model you want to load for either continued training or some testing.")
  parser.add_argument("--logdir", default="log", required=False,
                    help="Path to the logs folder.")
  parser.add_argument("--savestats", default=True, action="store_true", required=False,
                    help="Save stats for every run?")
  parser.add_argument("--savepth", default=False, action="store_true", required=False,
                    help="Save pth for every run?")

  #architecture, dataset
  parser.add_argument("--dataset", default="hep2", required=False,
                    help="What dataset should we use?")
  parser.add_argument("--classes", default=6, type=int, required=False,
                    help="How many output classes are there?")
  parser.add_argument("--feature_extractor", default="alexnet", required=False,
                    help="What feature extractor should we use?")

  #dataloading
  parser.add_argument("--num_workers", type=int, required=False, default=4,
                    help="Number of workers for dataloading")

  #learning rate schedulers

  parser.add_argument("--step_lr", type=int, required=False, default=None,
                    help="Learning rate schedule step size, if you want a step size")
  parser.add_argument("--patience", type=int, required=False, default=2000,
                    help="Learning rate schedule patience before it decreases the lr")
  parser.add_argument("--lr_gamma", type=float, required=False, default=0.1,
                    help="Learning rate multiplier every step size")


  parser.add_argument("--early_stop_steps", type=int, required=False, default=5000,
                    help="Number of steps of no learning to terminate")


  #optimiser
  parser.add_argument("--base_lr", type=float, required=False, default=0.003,
                    help="Base learning rate")
  parser.add_argument("--weight_decay", type=float, required=False, default=0.0,
                    help="Weight decay")


  parser.add_argument("--batch_size", type=int, required=False, default=128,
                    help="Batch size for training")
  parser.add_argument("--batch_split", type=int, required=False, default=1,
                    help="Number of batches to accumulate grads initially")

  parser.add_argument("--epochs", type=int, required=False, default=100,
                  help="Run how many epochs before terminating?")

  parser.add_argument("--adabatch", required=False, action="store_true", default=False,
                    help="Adaptively increase the batch size when training stagnates to get better performance?")

  


  #run types and data aug

  parser.add_argument("--random_flip", default=True, action="store_true", required=False,
                    help="No random horizontal flips for data aug?")
  parser.add_argument("--random_rotate", default=True, action="store_false", required=False,
                    help="No random rotations for data aug?")
  parser.add_argument("--gaussian", default=False, action="store_true", required=False,
                    help="Random gaussian blurs for data aug?")
  parser.add_argument("--aggressive", default=False, action="store_true", required=False,
                    help="Random extra aggressive data aug?")                  

  #stats and reporting
  parser.add_argument("--verbose", required=False, action="store_true", default=False,
                    help="Log data all the time?")
  parser.add_argument("--training_stats", required=False, action="store_true", default=False,
                    help="Calculate training stats?")
  parser.add_argument("--grid_search", required=False, action="store_true", default=False,
                    help="Run a grid search across some common hyperparameters?")
  parser.add_argument("--repeats", type=int, required=False, default=1,
                    help="Repeat the run how many times?")
  parser.add_argument("--eval_every", type=int, required=False, default=50,
                    help="Eval_every so many steps")


  return parser.parse_args()

def setup_logger(args):
  """Creates and returns a fancy logger."""
  # return logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
  # Why is setting up proper logging so !@?#! ugly?
  os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
  logging.config.dictConfig({
      "version": 1,
      "disable_existing_loggers": False,
      "formatters": {
          "standard": {
              "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
          },
      },
      "handlers": {
          "stderr": {
              "level": "INFO",
              "formatter": "standard",
              "class": "logging.StreamHandler",
              "stream": "ext://sys.stderr",
          },
          "logfile": {
              "level": "INFO",
              "formatter": "standard",
              "class": "logging.FileHandler",
              "filename": os.path.join(args.logdir, args.name, "train.log"),
              "mode": "a",
          }
      },
      "loggers": {
          "": {
              "handlers": ["stderr", "logfile"],
              "level": "DEBUG",
              "propagate": True
          },
      }
  })
  logger = logging.getLogger(__name__)
  logger.flush = lambda: [h.flush() for h in logger.handlers]
  logger.info(args)
  return logger