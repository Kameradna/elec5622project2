"""
This is the main functional section of the Project 2 cell classification,
including main training loop and entry point for users to run the project

Not debugged at all, just written.

To do:
Implement the dataloading in parts.py
Move run_eval to parts.py
"""

from torchvision.models import alexnet, AlexNet_Weights
import argparse
import torch

#more boilerplate stuff
# import pandas
# import numpy as np
from copy import deepcopy

#other companion code
import parts

def main(args):
  print("Loading model, weights and data")
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
  optim_schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_step_size, gamma=args.lr_gamma, last_epoch=- 1, verbose=False)
  
  #move to GPU if present
  model = model.to(device)
  optim.zero_grad(set_to_none=True)
  
  #statbuilding
  stats = {"train_acc":{},
          "valid_acc":{},
          "test_acc":{},
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
  stats = parts.run_eval(args, model, step, stats, device, train_loader, 'train')
  
  print("Beginning training")
  try:
    #main training loop
    for x, y in parts.recycle(train_loader):
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
      stats['lr'][step] = optim_schedule.get_last_lr()[0]

      if args.training_stats:
        print(f"Stats@{step}: train loss {stats['train_loss'][step]:.5f},",
                              f"train accuracy {stats['train_acc'][step]:.2f}%,",
                              f"valid loss {stats['valid_loss'][step]:.5f},",
                              f"valid accuracy {stats['valid_acc'][step]:.2f}%",
                              f"learning rate {stats['lr'][step]}"
                    )
      else:
        print(f"Stats@{step}: train loss {stats['train_loss'][step]:.5f},",
                              f"valid loss {stats['valid_loss'][step]:.5f},",
                              f"valid accuracy {stats['valid_acc'][step]:.2f}%",
                              f"learning rate {stats['lr'][step]}"
                    )

      #grab the best model if it happens
      if stats['valid_acc'][step] > best_valid_acc:
        best_weights = deepcopy(model.state_dict())
        best_step = step

      if step % args.lr_step_size == 0:
        optim_schedule.step()
      
      ######### end of training loop
  except KeyboardInterrupt:
    print(f"Done training, calculating test accuracy!")
  
  #display best stats
  model.load_state_dict(best_weights)
  step = best_step

  if args.training_stats:
    print(f"Stats@{step}: train loss {stats['train_loss'][step]:.5f},",
                          f"train accuracy {stats['train_acc'][step]:.2f}%,",
                          f"valid loss {stats['valid_loss'][step]:.5f},",
                          f"valid accuracy {stats['valid_acc'][step]:.2f}%,",
                          f"learning rate {stats['lr'][step]}"
                )
  else:
    print(f"Stats@{step}: train loss {stats['train_loss'][step]:.5f},",
                          f"valid loss {stats['valid_loss'][step]:.5f},",
                          f"valid accuracy {stats['valid_acc'][step]:.2f}%,",
                          f"learning rate {stats['lr'][step]}"
                )

  stats = parts.run_eval(args, model, step, stats, device, test_loader, 'test')
  print(f"Test stats@{step}: test loss {stats['test_loss'][step]:.5f},",
                          f"test accuracy {stats['train_acc'][step]:.2f}%")

  
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--datadir", default="data", required=False,
                    help="Path to the data folder, preprocessed for torchvision.")
  parser.add_argument("--base_lr", type=float, required=False, default=0.003,
                    help="Learning rate")
  parser.add_argument("--lr_step_size", type=float, required=False, default=10,
                    help="Learning rate")
  parser.add_argument("--lr_gamma", type=float, required=False, default=0.1,
                    help="Learning rate")
  #args.batch_size, num_workers=args.num_workers
  parser.add_argument("--batch_size", type=int, required=False, default=128,
                    help="Batch size for training")
  parser.add_argument("--num_workers", type=int, required=False, default=8,
                    help="Number of workers for dataloading")
  parser.add_argument("--verbose", required=False, action="store_true", default=False,
                    help="Print data all the time?")
  parser.add_argument("--training_stats", required=False, action="store_true", default=False,
                    help="Calculate training stats?")
  main(parser.parse_args())
