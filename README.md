# elec5622project2


## Datasets
```shell
gdown https://drive.google.com/file/d/1reoKyZ3SwNDcZlNmT_031VgvQSI4dkhU/view?usp=sharing --fuzzy
unzip "Three Datasets for Training, Validation and Test.zip" -d data
unzip data/test.zip -d data
unzip data/training.zip -d data
unzip data/validation.zip -d data
gdown https://drive.google.com/file/d/1yzqHghT9M0G4BHhz_gDTB0Ml7DXPM3BF/view?usp=sharing --fuzzy -O data/data.csv
rm -v "Three Datasets for Training, Validation and Test.zip" data/test.zip data/training.zip data/validation.zip
```
## Prerequisites

Install a conda env or your choice of sandboxed environment, yml files will be included.
We just need python=3.x, pip, pytorch, torchvision, maybe some numpy and pandas.

## How to use

```shell
python dataset_format.py #only need to run once
python main.py --savestats #recommended, other params visible using -h, defaults should be functional for normal use case
python process_output.py
```
Dataset_format puts the images in the training, validation and test sets into or use by the torchvision class ImageFolder.

main.py runs either a single run with selectable parameters or a grid search (when using --grid_search, see -h) based on the values of grid_dict inside the function grid_search in main.py. parts.py is function definitions. main.py optionally saves pth for each run, csv for each run, and other functions see below.

```shell
usage: main.py [-h] [--datadir DATADIR] [--savedir SAVEDIR] [--savestats] [--savepth] [--base_lr BASE_LR] [--lr_step_size LR_STEP_SIZE] [--lr_gamma LR_GAMMA]
               [--batch_size BATCH_SIZE] [--early_stop_steps EARLY_STOP_STEPS] [--num_workers NUM_WORKERS] [--verbose] [--training_stats] [--grid_search]

optional arguments:
  -h, --help            show this help message and exit
  --datadir DATADIR     Path to the data folder, preprocessed for torchvision.
  --savedir SAVEDIR     Path to the save folder, for placement of csv and pth files.
  --savestats           Save stats for every run?
  --savepth             Save pth for every run?
  --base_lr BASE_LR     Base learning rate
  --lr_step_size LR_STEP_SIZE
                        Learning rate schedule step size
  --lr_gamma LR_GAMMA   Learning rate multiplier every step size
  --batch_size BATCH_SIZE
                        Batch size for training
  --early_stop_steps EARLY_STOP_STEPS
                        Number of epochs of no learning to terminate
  --num_workers NUM_WORKERS
                        Number of workers for dataloading
  --verbose             Print data all the time?
  --training_stats      Calculate training stats?
  --grid_search         Run a grid search across some common hyperparameters?
```

process_output.py takes the output csv files and reports the aggregated results in a single csv file grid_search_results.csv.



## Actual code
Absolutely standard Alexnet from pytorch. The code was completed in less than a day.

We restructure the data for passing to the ImageFolder class of Dataset in torchvision.

We allow for learning rate scheduling, changing batch size etc.

![image](https://user-images.githubusercontent.com/48018617/194888313-a586faca-3ff0-4423-9b87-aba254cec9ba.png)

At the initial first working copy, no hyperparamter tuning, we get 82.1% accuracy on the validation set. Acceptable for the first day.

## Results

Grid search is running for the hyperparameters batch size, learning rate, and some learning rate scheduling parameters. Early stopping is used to ensure models that begin to overfit are not pursued further than 30 steps, which may need to be increased if this proves too constraining for promising models.

![Test acc vs base lr](https://user-images.githubusercontent.com/48018617/195095185-99de1f3e-cf74-497a-aa75-ee0290504c08.png)
![Test acc vs base lr bs64](https://user-images.githubusercontent.com/48018617/195095194-b60583b7-f674-4dc5-bb17-6be8a1071658.png)
![Test acc vs batch size](https://user-images.githubusercontent.com/48018617/195095195-dfe7805c-5fb1-4713-a15f-ea77a65938dc.png)

With an initial understanding of the landscape of the learning rate and batch size selections, we move on to more specific grid searches of longer duration, extending early stopping to 100 steps. 

![Test acc vs base lr longer](https://user-images.githubusercontent.com/48018617/195110084-08b14ab6-14b6-4fc9-bf19-20721f59f84d.png)
![Test acc vs base lr longer128](https://user-images.githubusercontent.com/48018617/195110094-ec3cc6e6-a853-4092-b760-ffa26082996b.png)

We get quite promising results from this regime, for example this run with near-identical hyperparameters to our happy guess at the beginning; getting 85.33% validation accuracy. The only difference is that this run uses much more relaxed learning rate scheduling; a step size of 200 and a gamma of 0.5. Further on we can experiment with less/no learning rate scheduling.

![Training and validation loss](https://user-images.githubusercontent.com/48018617/195115413-7b780b47-7f02-4129-a7c5-e113ce92da9c.png)
![Validation accuracy vs step](https://user-images.githubusercontent.com/48018617/195116133-eb920351-95aa-467c-82c4-2f8fbb32c4db.png)

We could consider this a somewhat over-regularised scheme, with training loss staying well above validation loss.


100 steps still only represents just 0.735 or 1.47 epochs respectively for batch sizes 64 and 128 (based on the 8707 image training set). We can standardise the testing to account for this, making the early stopping a function of batch size, and early stopping if there is no learning for 3 epochs (408 and 204 steps each for batch sizes 64 and 128). Let's explore some of the most promising hyperparameter options; learning rate 0.003, batch sizes 64 and 128, with a scheduler step size of 100, gamma 0.1.

A teammates computer was able to run the model with a batch size of 256, and we found the higher learning rate of 0.006, no scheduling, to be a viable candidate for best training scheme.

![loss vs step 256 0 006](https://user-images.githubusercontent.com/48018617/195250701-66f90036-ee57-41be-8c2f-409a3f0d5dea.png)
![valid acc vs step 256 0 006](https://user-images.githubusercontent.com/48018617/195250707-3f97dfdb-d6ca-4748-8897-17e88535afa8.png)

Based on this positive result of acheiving 96.44% accuracy with the same early stopping rules (3 epochs = 102 steps) as above, of stopping after 3 epochs, we can try batches of 512 if we can run them, as well as automated learning rate scheduling when learning stagnates. Also noteworthy is the lack of overfitting, so longer runs should be possible with no consequence.

To summarise; the up to this point best model training scheme is batch size 256, learning rate 0.006 with no scheduling, and seemingly we should let the training go for longer with a more relaxed early stopping guideline of >3 epochs (>102 steps). 

## Final model

| Left-aligned | Center-aligned | Right-aligned |
| :----         |     :---:      |          ---: |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |
