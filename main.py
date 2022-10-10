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
import numpy as np

import parts

def main(args):
  #getting transforms
  weights = AlexNet_Weights.DEFAULT
  preprocess = weights.transforms()
  
  #setting up training and validation data
  train_loader, valid_loader, train_set, valid_set = parts.mktrainval(args, preprocess)
  
  #misc dataparallel
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  #getting alexnet and altering the final output layer
  model = alexnet(weights=weights)
  num_features = model.classifier.in_features
  model.classifier = nn.Linear(num_features, len(valid_set.classes),bias=True)
  
  #model to dataparallel
  model = torch.nn.DataParallel(model)
  
  #optimiser, loss function, lr scheduler
  criterion = torch.nn.MSELoss.to(device)
  optim = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
  optim_schedule = torch.optim.lr_scheduler.StepLR(optim=optim, step_size=args.lr_step_size, gamma=args.lr_gamma, last_epoch=- 1, verbose=False)
  
  #move to GPU if present
  model = model.to(device)
  optim.zero_grad(set_to_none=True)
  
  #statbuilding
  stats = {"train_acc":[],
          "valid_acc":[],
          "step":[]}
  step = 0
                                                   
  #initial validation
  model.eval()
  valid_acc = parts.run_eval(args, model, step, stats, valid_loader)
  
  #main training loop
  for x, y in parts.recycle(train_loader):
    
    #onto training
    model.train()
    #move to device
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
  
    logits = model(x)
    logits.clamp_(0,1)
    c = cri(logits, y)
    c_num = float(c.data.cpu().numpy())

    #the backward pass
    c.backward()
    
    #apply optimisation and reset grads to 0 for next pass                                               
    optim.step()
    optim.zero_grad(set_to_none=True)
    
    if step % args.lr_step_size == 0:
      optim_schedule.step()
    
    #run validation
    model.eval()
    train_acc = parts.run_eval(args, model, step, stats, train_loader)
    valid_acc = parts.run_eval(args, model, step, stats, valid_loader)
  
  print(f"Done training!")
  
  #probably publish the stats in some readable way
  #graphs, save to csv, etc
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--datadir data", required=True,
                    help="Path to the data folder, preprocessed for torchvision.")
  parser.add_argument("--lr", type=float, required=True,
                    help="Learning rate")
  parser.add_argument("--lr_step_size", type=float, required=False, default=10,
                    help="Learning rate")
  parser.add_argument("--lr_gamma", type=float, required=False, default=0.1,
                    help="Learning rate")
  main(parser.parse_args())
