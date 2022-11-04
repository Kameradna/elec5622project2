"""
This is the run functional section of the Project 2 cell classification,
including run training loop and entry point for users to run the project.

Most interesting parts-

- Patience of early stopping and ReduceLROnPlateau scheduler and adaptive gradient accumulation (adabatch analogue)
- Grid search using sklearn.model_selection.ParameterGrid
- Using updated pytorch weights API to apply transforms to data
- Saving non-DataParallel state_dict() for easier reloading
- Friendly interface to run different combinations of parameters from the command line
- Terminating long training via keyboard will still execute testing phase and report stats
- Amp implemented
- Logging implemented

Defaults are 
Batch size = 128 (use higher if you have access to more VRAM)
Learning rate = 0.003 (optimal at batch size of 128, 0.006 better for higher batch sizes)
Early stopping after 10 epochs of no improvement in validation accuracy (1 epoch = 8701/batch_size steps)
Learning rate *= 0.1 after 5 epochs of no *significant* improvement in validation accuracy (significant defined by defaults of ReduceLROnPlateau scheduler)

Known issues-
- Random rotate adds 2x time per validation

To run with escalating accumulating sizes for the grads

python run.py --early_stop_steps 1000 --lr_gamma 0.5 --adabatch


Todo:
Use HuggingFace Accelerate with Deepspeed, maybe fsdp see if there is a speedup on me vs old mate
Cosine Annealing with warm restarts
Learning rate warmup

Use nvFuser with a newer GPU architecture on more difficult problems to acheive way better performance
Additionally, using optimum gradient checkpointing for memory management.
"""

from torchvision.models import alexnet, AlexNet_Weights, resnet101, ResNet101_Weights
import torch
from sklearn.model_selection import ParameterGrid
import os

#more boilerplate stuff
import pandas as pd
from copy import deepcopy
import time

#other companion code
import parts

def grid_search(args, logger):

  best_acc = 0.0
  grid_dict = { #params to search through, will cover all combinations
    'random_rotate':[True, False],
    'random_flip':[True, False],
    'gaussian':[True, False],
    'aggressive':[True, False],
    'weight_decay':[0.0,0.0005],
    'repeats':range(3)
  }

  #for testing combos quickly to ensure no issues
  # args.eval_every = 1
  # args.early_stop_steps = 0

  grid = ParameterGrid(grid_dict)
  logger.info(f"Grid searching across {len(grid)} combinations")
  for idx, params in enumerate(grid):
    logger.info(f"Run {idx} of {len(grid)} combinations")
    args.random_flip = params['random_flip']
    args.random_rotate = params['random_rotate']
    args.gaussian = params['gaussian']
    args.aggressive = params['aggressive']
    args.weight_decay = params['weight_decay']

    #run run
    stats, step, model, stop_reason, kill_flag = run(args, logger)

    if stats['test_acc'][step] > best_acc:
      best_acc = stats['test_acc'][step]
      best_model = model
      best_args = deepcopy(args)
      best_stats = stats
      best_stop_reason = stop_reason
    
    if kill_flag:
      logger.info("Ending grid search...")
      break

  logger.info(f"The best combination of params found was {best_args}, with test accuracy {best_acc}")
  logger.info(f"Saving best model in {args.savedir}, along with best_stats.csv")
  if os.path.exists(args.savedir) != True:
      os.mkdir(args.savedir)

  args = best_args
  name_args = [args.feature_extractor, f"baselr{args.base_lr}", "", f"lrgam{args.lr_gamma}", f"bat{args.batch_size}", f"step{step}", best_stop_reason]
  name = f"{'_'.join(name_args)}_best_{best_acc}.pth"
  torch.save(best_model.module.state_dict(),os.path.join(args.savedir,name)) #.module to deencapsulate the statedict from DataParallel

  name = f"{'_'.join(name_args)}_{best_stop_reason}_best_{best_acc}.csv"
  best_stat_df = pd.DataFrame(best_stats)
  best_stat_df.to_csv(os.path.join(args.savedir,name))




def run(args, logger):

  logger.info("Loading model, weights and data")
  torch.backends.cuda.benchmarking = True

  model_start = time.perf_counter()

  #getting transforms
  if args.feature_extractor == 'alexnet':
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    #getting input sizes and altering the final output layer
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(num_features, args.classes,bias=True)
  elif args.feature_extractor == 'resnet101':
    weights = ResNet101_Weights.IMAGENET1K_V2
    model = resnet101(weights=weights)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, args.classes,bias=True)
  else:
    logger.info(f"{args.feature_extractor} not implemented yet. Please retype or implement here. Exiting...")

  preprocess = weights.transforms()
  

  #setting up training and validation data
  train_loader, valid_loader, test_loader, train_set, valid_set, test_set = parts.mktrainval(args, preprocess, logger)
  

  logger.debug(f"Classes are {valid_set.classes}")
  
  device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

  #optimiser, loss function, lr scheduler
  criterion = torch.nn.CrossEntropyLoss().to(device)
  optim = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)
  optim_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=args.lr_gamma, patience=args.patience) #other items default

  model = torch.nn.DataParallel(model) #move to gpus
  model.to(device)
  
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
          'lr':{},
          'batch_size':{},
          'batch_split':{}}
  
  step = -1
  accum_step = 0

                                                   
  #initial validation
  logger.info("Running initial validation")
  model.eval()
  stats = parts.run_eval(args, model, step, stats, valid_loader, 'valid', logger, device)
  if args.training_stats:
    stats = parts.run_eval(args, model, step, stats, train_loader, 'train', logger, device)

  time_taken = time.perf_counter()-model_start

  parts.logstats(args, stats, logger, step, None, None)



  best_valid_acc = stats['valid_acc'][step]
  best_weights = deepcopy(model.state_dict())
  best_step = step

  step += 1 #so by design, the first trained epoch is 0, while the first initial starting point is at step -1
  last_eval_time = time.perf_counter()
  


  logger.info("Beginning training")

  try:
    kill_flag = False
    training_loop_start = time.perf_counter()



    #run training loop
    for x, y in parts.recycle(train_loader):

      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)


      if args.verbose:
        logger.info(f"Training step {step}")

      #onto training
      model.train()

      #the forward pass
      logits = model(x)
      logits.clamp_(0,1) #actually forced nans in amp scenario
      c = criterion(logits, y)

      #the backward pass with gradient accumulation
      (c/args.batch_split).backward(c) #/args.batch_split
      accum_step += 1
      

      if accum_step % args.batch_split == 0:
        accum_step = 0
        step += 1
        optim.step()
        optim.zero_grad(set_to_none=True)
        stats['lr'][step] = optim.param_groups[0]['lr']

        optim_schedule.step(stats['valid_acc'][-1])

        stats['train_loss'][step] = float(c.data.cpu().numpy())
        stats['train_loss'][step] = float(c.data.cpu().numpy())
        
        stats['batch_size'][step] = args.batch_size
        stats['batch_split'][step] = args.batch_split



        #adaptive batching
        if stats['lr'][step] != optim.param_groups[0]['lr'] and args.adabatch: #if the scheduler changed the learning rate by gamma
          args.batch_split = int(args.batch_split*2)
          logger.info(f"Accumulating grads over {args.batch_split} steps")



        #run validation
        if step % args.eval_every == 0:
            
          model.eval()
          stats = parts.run_eval(args, model, step, stats, valid_loader, 'valid', logger, device)
          if args.training_stats:
            stats = parts.run_eval(args, model, step, stats, train_loader, 'train', logger, device)
          

          #grab the best model if it happens
          if stats['valid_acc'][step] > best_valid_acc:
            best_valid_acc = stats['valid_acc'][step]
            best_weights = deepcopy(model.state_dict())
            best_step = step

          time_taken = (time.perf_counter()-last_eval_time)/args.eval_every
          last_eval_time = time.perf_counter()
          
          parts.logstats(args, stats, logger, step, best_step, time_taken)

        if step - args.early_stop_steps > best_step:
          logger.info(f"Learning has stagnated for {args.early_stop_steps} steps, terminating training and running test stats")
          stop_reason = f'stagnate@{step}'
          break
        if step >= args.epochs*8701/args.batch_size:#for worst case scenario of learning continuing at snail pace
          logger.info(f"Learning has taken too long; {args.epochs*8701/args.batch_size} steps, terminating training and running test stats")
          stop_reason = f'timeout@{step}'
          break

        if args.verbose:
          logger.info(f"Training step took {time.perf_counter()-training_loop_start:.2f} seconds")
        ######### end of training loop




  except KeyboardInterrupt:
    logger.info(f"Keyboard interrupted training, calculating test accuracy!")
    kill_flag = input("Do you want to end all runs? [y]/n")
    if kill_flag == 'y' or kill_flag == '':
      kill_flag = True
    else:
      kill_flag = False
    stop_reason = f'keyboard@{step}'



  
  #display best stats
  model.load_state_dict(best_weights)
  step = best_step

  parts.logstats(args, stats, logger, step, best_step, time_taken)

  if test_loader is not None:
    stats = parts.run_eval(args, model, step, stats, test_loader, 'test', logger, device)
    parts.logstats(args, stats, logger, step, best_step, 'end')

  logger.info(f"Training took {args.batch_size/8701*step:.2f} epochs, {(time.perf_counter()-model_start)/3600:.2f} hours")

  name_args = [args.feature_extractor, 
      f"baselr{args.base_lr}", 
      f"lrgam{args.lr_gamma}", "",       #spaces to runtain interworking with process_output
      f"bat{args.batch_size}", 
      f"step{step}", stop_reason, 
      f"{stats['test_acc'][step]:.2f}",  
      "random_flip" if args.random_flip else '', 
      "random_rotate" if args.random_rotate else '',
      "gaussian" if args.gaussian else '',
      "aggressive" if args.aggressive else '',
      'weight_decay' if args.weight_decay is not None else ''
      ]

  if args.savepth:
    if os.path.exists(args.savedir) != True:
      os.mkdir(args.savedir)
    name = f"{'_'.join(name_args)}.pth"
    logger.info(f"Saving model in {os.path.join(args.savedir,name)}")
    torch.save({"checkpoint":model.module.state_dict()},os.path.join(args.savedir,name)) #.module to deencapsulate the statedict from DataParallel
  if args.savestats:
    if os.path.exists(args.savedir) != True:
      os.mkdir(args.savedir)
    name = f"{'_'.join(name_args)}.csv"
    stat_df = pd.DataFrame(stats).to_csv(os.path.join(args.savedir,name))

  return stats, step, model, stop_reason, kill_flag



def setup():

  args = parts.parseargs()
  logger = parts.setup_logger(args)
  return args, logger



if __name__ == "__main__":
  args, logger = setup()

  if args.grid_search:
    grid_search(args, logger)
  else:
    for idx in range(args.repeats):
      run(args, logger)
