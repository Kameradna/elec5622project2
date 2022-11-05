import parts
import torch
from torchvision.models import alexnet, AlexNet_Weights, resnet101, ResNet101_Weights
import time


def test(args, logger):
    #run eval
    logger.info("Loading model, weights and data")
    torch.backends.cuda.benchmarking = True
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

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

    #load model from loaddir
    module_model = torch.nn.DataParallel(model) #put the pretrained model in dataparallel as before so weights correspond


    print(f"Loading model will be attempted from '{args.loaddir}'")
    model.load_state_dict(torch.load(args.loaddir, map_location="cpu")["checkpoint"])
    print(f"Found saved model to resume from at '{args.loaddir}'")

    try:
        state_dict = module_model.module.state_dict()
    except AttributeError:
        raise(NotImplementedError, "There is some issue with the weights, implement a handler.")
    
    model.load_state_dict(state_dict)

    print(f"successfully loaded model to resume from '{args.loaddir}'")

    model = torch.nn.DataParallel(model)
    model.to(device)

    preprocess = weights.transforms()
    
    #setting up training and validation data
    _, _, test_loader, _, _, test_set = parts.mktrainval(args, preprocess, logger)
    
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
    logger.info("Running test")


    model.eval()
    stats = parts.run_eval(args, model, step, stats, test_loader, 'test', logger, device)

    parts.logstats(args, stats, logger, step, 1, 'end')

    time_taken = time.perf_counter()-model_start
    
    logger.info(f"Testing took {time_taken:.2f} seconds")

    return 0

def setup():
  args = parts.parseargs()
  logger = parts.setup_logger(args)
  return args, logger

if __name__ == "__main__":
  args, logger = setup()

  test(args, logger)