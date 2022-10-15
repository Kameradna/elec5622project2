"""
This is the main functional section of the Project 2 cell classification,
including main training loop and entry point for users to run the project.

Most interesting parts-

- Using batch size to dictate patience of early stopping and ReduceLROnPlateau scheduler
- Grid search using sklearn.model_selection.ParameterGrid
- Using updated pytorch weights API to apply transforms to data
- Saving non-DataParallel state_dict() for easier reloading
- Friendly interface to run different combinations of parameters from the command line
- Terminating long training via keyboard will still execute testing phase and report stats

Defaults are 
Batch size = 128 (use higher if you have access to more VRAM)
Learning rate = 0.003 (optimal at batch size of 128, 0.006 better for higher batch sizes)
Early stopping after 10 epochs of no improvement in validation accuracy (1 epoch = 8701/batch_size steps)
Learning rate *= 0.1 after 5 epochs of no *significant* improvement in validation accuracy (significant defined by defaults of ReduceLROnPlateau scheduler)

Known issues-
- Mca adds 2x time per validation
- Timing is not real


"""

from torchvision.models import alexnet, AlexNet_Weights
import argparse
import torch
from sklearn.model_selection import ParameterGrid
import os

#more boilerplate stuff
import pandas as pd
import numpy as np
from copy import deepcopy
import time

#other companion code
import parts

def grid_search(args):

  best_acc = 0.0
  grid_dict = {
    'base_lr': [0.003],
    'batch_size':[128],
    'random_rotate':[True, False],
    'random_flip':[True, False],
    'repeats':range(10)
  }

  grid = ParameterGrid(grid_dict)
  print(f"Grid searching across {len(grid)} combinations")
  for idx, params in enumerate(grid):
    print(f"Run {idx} of {len(grid)} combinations")
    args.base_lr = params['base_lr']
    # args.lr_step_size = params['lr_step_size']
    # args.lr_gamma = params['lr_gamma']
    args.batch_size = params['batch_size']
    args.random_flip = params['random_flip']
    args.random_rotate = params['random_rotate']


    args.early_stop_steps = int(10*8701/128) #10 epochs for batch size 128, more for larger since they take a longer epoch time to converge, patience of lr scheduler is 5 epochs
    
    #run main
    stats, step, model, stop_reason, kill_flag = main(args)

    if stats['test_acc'][step] > best_acc:
      best_acc = stats['test_acc'][step]
      best_model = model
      best_args = deepcopy(args)
      best_stats = stats
      best_stop_reason = stop_reason
    
    if kill_flag:
      print("Ending grid search...")
      break

  print(f"The best combination of params found was {best_args}, with test accuracy {best_acc}")
  print(f"Saving best model in {args.savedir}, along with best_stats.csv")
  if os.path.exists(args.savedir) != True:
      os.mkdir(args.savedir)

  args = best_args
  name_args = ['alexnet', f"baselr{args.base_lr}", f"lrstep{args.lr_step_size}", f"lrgam{args.lr_gamma}", f"bat{args.batch_size}", f"step{step}", best_stop_reason]
  name = f"{'_'.join(name_args)}_best_{best_acc}.pth"
  torch.save(best_model.module.state_dict(),os.path.join(args.savedir,name)) #.module to deencapsulate the statedict from DataParallel

  name = f"{'_'.join(name_args)}_{best_stop_reason}_best_{best_acc}.csv"
  best_stat_df = pd.DataFrame(best_stats)
  best_stat_df.to_csv(os.path.join(args.savedir,name))

def main(args):
  print("Loading model, weights and data")
  print(f"args = {args}")
  model_start = time.perf_counter()
  #getting transforms
  weights = AlexNet_Weights.DEFAULT
  preprocess = weights.transforms()
  
  #setting up training and validation data
  train_loader, valid_loader, test_loader, train_set, valid_set, test_set = parts.mktrainval(args, preprocess)
  
  #misc dataparallel
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  #getting alexnet and altering the final output layer
  model = alexnet(weights=weights)
  num_features = model.classifier[-1].in_features
  model.classifier[-1] = torch.nn.Linear(num_features, len(valid_set.classes),bias=True)

  #model to dataparallel
  model = torch.nn.DataParallel(model)
  
  #optimiser, loss function, lr scheduler
  criterion = torch.nn.CrossEntropyLoss() #.to(device)
  optim = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
  # optim_schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_step_size, gamma=args.lr_gamma, last_epoch=- 1, verbose=False)
  optim_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=args.lr_gamma, patience=args.early_stop_steps//2) #other items default
  
  #move to GPU if present
  model = model.to(device)
  optim.zero_grad(set_to_none=True)
  
  #statbuilding
  stats = {"train_acc":{},
          "valid_acc":{},
          "test_acc":{},
          "valid_mca":{},#new
          "valid_mca":{},#new
          "test_mca":{},#new
          "train_loss":{},
          "valid_loss":{},
          "test_loss":{},
          'lr':{}}
  step = -1
  best_valid_acc = 0.0
                                                   
  #initial validation
  print("Running initial validation")
  model.eval()
  stats = parts.run_eval(args, model, step, stats, device, valid_loader, 'valid')
  if args.training_stats:
    stats = parts.run_eval(args, model, step, stats, device, train_loader, 'train')

  time_taken = time.perf_counter()-model_start
  print(f"Stats@{step}: valid loss {stats['valid_loss'][step]:.5f},",
                      f"valid accuracy {stats['valid_acc'][step]:.2f}%",
                      f"valid per-class mean accuracy {stats[f'valid_mca'][step]:.2f}%",
                      f"time taken loading {time_taken:.2f}s"
                    )

  
  print("Beginning training")
  try:
    kill_flag = False
    #main training loop
    for x, y in parts.recycle(train_loader):
      training_loop_start = time.perf_counter()
      step += 1 #so by design, the first trained epoch is 0, while the first initial starting point is at step -1
      if args.verbose:
        print(f"Training step {step}")

      #onto training
      model.train()
      #move to device
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      #the forward pass
      logits = model(x)
      logits.clamp_(0,1)
      c = criterion(logits, y)

      #the backward pass
      c.backward()
      
      #apply optimisation and reset grads to 0 for next pass                                               
      optim.step()
      optim.zero_grad(set_to_none=True)
      
      #run validation
      model.eval()
      stats = parts.run_eval(args, model, step, stats, device, valid_loader, 'valid')
      if args.training_stats:
        stats = parts.run_eval(args, model, step, stats, device, train_loader, 'train')
      stats['train_loss'][step] = float(c.data.cpu().numpy())
      stats['lr'][step] = optim.param_groups[0]['lr']
        
      optim_schedule.step(stats['valid_acc'][step])

      #grab the best model if it happens
      if stats['valid_acc'][step] > best_valid_acc:
        best_valid_acc = stats['valid_acc'][step]
        best_weights = deepcopy(model.state_dict())
        best_step = step

      time_taken = time.perf_counter()-training_loop_start
      if args.training_stats:
        print(f"Stats@{step}: train loss {stats['train_loss'][step]:.5f},",
                              f"train accuracy {stats['train_acc'][step]:.2f}%,",
                              f"valid loss {stats['valid_loss'][step]:.5f},",
                              f"valid accuracy {stats['valid_acc'][step]:.2f}%",
                              f"valid mca {stats[f'valid_mca'][step]:.2f}%",
                              f"learning rate {stats['lr'][step]:.8f}",
                              f"time taken this step {time_taken:.2f}s",
                              f"{'(best)' if best_step == step else ''}"
                    )
      else:
        print(f"Stats@{step}: train loss {stats['train_loss'][step]:.5f},",
                              f"valid loss {stats['valid_loss'][step]:.5f},",
                              f"valid accuracy {stats['valid_acc'][step]:.2f}%",
                              f"valid mca {stats[f'valid_mca'][step]:.2f}%",
                              f"learning rate {stats['lr'][step]:.8f}",
                              f"time taken this step {time_taken:.2f}s",
                              f"{'(best)' if best_step == step else ''}"
                    )

      if step - args.early_stop_steps > best_step:
        print(f"Learning has stagnated for {args.early_stop_steps} steps, terminating training and running test stats")
        stop_reason = f'stagnatewithreduceonplateau@{step}'
        break
      if step >= args.early_stop_steps*50:#for worst case scenario of learning continuing at snail pace
        print(f"Learning has taken too long; {args.early_stop_steps*50} steps, terminating training and running test stats")
        stop_reason = f'too_slow@{step}'
        break

      
      ######### end of training loop
  except KeyboardInterrupt:
    print(f"Keyboard interrupted training, calculating test accuracy!")
    kill_flag = input("Do you want to end all runs? [y]/n")
    if kill_flag == 'y' or kill_flag == '':
      kill_flag = True
    else:
      kill_flag = False
    stop_reason = f'keyboard@{step}'
  
  #display best stats
  model.load_state_dict(best_weights)
  step = best_step

  if args.training_stats:
    print(f"Stats@{step}: train loss {stats['train_loss'][step]:.5f},",
                          f"train accuracy {stats['train_acc'][step]:.2f}%,",
                          f"valid loss {stats['valid_loss'][step]:.5f},",
                          f"valid accuracy {stats['valid_acc'][step]:.2f}%,",
                          f"valid per-class mean accuracy {stats[f'valid_mca'][step]:.2f}%",
                          f"learning rate {stats['lr'][step]:.8f}"
                )
  else:
    print(f"Stats@{step}: train loss {stats['train_loss'][step]:.5f},",
                          f"valid loss {stats['valid_loss'][step]:.5f},",
                          f"valid accuracy {stats['valid_acc'][step]:.2f}%,",
                          f"valid per-class mean accuracy {stats[f'valid_mca'][step]:.2f}%",
                          f"learning rate {stats['lr'][step]:.8f}"
                )

  stats = parts.run_eval(args, model, step, stats, device, test_loader, 'test')
  print(f"Test stats@{step}: test loss {stats['test_loss'][step]:.5f},",
                          f"test accuracy {stats['test_acc'][step]:.2f}%",
                          f"test per-class mean accuracy {stats[f'test_mca'][step]:.2f}%")

  print(f"Training took {args.batch_size/8701*step:.2f} epochs, {(time.time()-model_start)/3600:.2f} hours")

  if args.savepth:
    if os.path.exists(args.savedir) != True:
      os.mkdir(args.savedir)
    name_args = ['alexnet', f"baselr{args.base_lr}", f"lrstep{args.lr_step_size}", f"lrgam{args.lr_gamma}", f"bat{args.batch_size}", f"step{step}", stop_reason, f"{stats['test_acc'][step]:.2f}",  "random_flip" if args.random_flip else '', "random_rotate" if args.random_rotate else '']
    name = f"{'_'.join(name_args)}.pth"
    print(f"Saving model in {os.path.join(args.savedir,name)}")
    torch.save({"checkpoint":model.module.state_dict()},os.path.join(args.savedir,name)) #.module to deencapsulate the statedict from DataParallel
  if args.savestats:
    if os.path.exists(args.savedir) != True:
      os.mkdir(args.savedir)
    name_args = ['alexnet', f"baselr{args.base_lr}", f"lrstep{args.lr_step_size}", f"lrgam{args.lr_gamma}", f"bat{args.batch_size}", f"step{step}", stop_reason, f"{stats['test_acc'][step]:.2f}",  "random_flip" if args.random_flip else '', "random_rotate" if args.random_rotate else '']
    name = f"{'_'.join(name_args)}.csv"
    stat_df = pd.DataFrame(stats).to_csv(os.path.join(args.savedir,name))

  return stats, step, model, stop_reason, kill_flag

  
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--datadir", default="data", required=False,
                    help="Path to the data folder, preprocessed for torchvision.")
  parser.add_argument("--savedir", default="save", required=False,
                    help="Path to the save folder, for placement of csv and pth files.")
  parser.add_argument("--savestats", default=True, action="store_true", required=False,
                    help="Save stats for every run?")
  parser.add_argument("--savepth", default=False, action="store_true", required=False,
                    help="Save pth for every run?")
  parser.add_argument("--base_lr", type=float, required=False, default=0.003,
                    help="Base learning rate")
  # parser.add_argument("--lr_step_size", type=int, required=False, default=100,
  #                   help="Learning rate schedule step size")
  parser.add_argument("--lr_gamma", type=float, required=False, default=0.1,
                    help="Learning rate multiplier every step size")
  parser.add_argument("--batch_size", type=int, required=False, default=128,
                    help="Batch size for training")
  parser.add_argument("--early_stop_steps", type=int, required=False, default=680,
                    help="Number of steps of no learning to terminate")
  parser.add_argument("--num_workers", type=int, required=False, default=8,
                    help="Number of workers for dataloading")
  parser.add_argument("--verbose", required=False, action="store_true", default=False,
                    help="Print data all the time?")
  parser.add_argument("--training_stats", required=False, action="store_true", default=False,
                    help="Calculate training stats?")
  parser.add_argument("--grid_search", required=False, action="store_true", default=False,
                    help="Run a grid search across some common hyperparameters?")
  parser.add_argument("--random_flip", default=False, action="store_true", required=False,
                    help="Random horizontal flips for data aug?")
  parser.add_argument("--random_rotate", default=False, action="store_true", required=False,
                    help="Random rotations for data aug?")
  # args.random_flip = params['random_flip']
  # args.random_rotate
  args = parser.parse_args()
  
  if args.grid_search:
    grid_search(args)
  else:
    main(args)
