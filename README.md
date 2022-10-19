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

Hyperparamter grid search was run for batch size, learning rate, and some learning rate scheduling parameters. Early stopping is used to ensure models that do not learn well are not pursued further than 30 steps, which may need to be increased if this proves too constraining for promising models. We run each model around 5 times, taking the maximum test accuracy acheived for each set of hyperparameters as a good starting point for viability of those hyperparameters.

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


100 steps still only represents just 0.735 or 1.47 epochs respectively for batch sizes 64 and 128 (based on the 8707 image training set). We can standardise the testing to account for this, making the early stopping a function of batch size, and early stopping if there is no learning for 3 epochs (408 and 204 steps each for batch sizes 64 and 128). Let's explore some of the most promising hyperparameter options; learning rate 0.003, batch sizes 64 and 128, with a scheduler step size of 100, gamma 0.1. We use the longer early stopping regime of 3 epochs to better see the differences between these more viable options.

A teammates computer was able to run the model with a batch size of 256, and we found the higher learning rate of 0.006, no scheduling, to be a viable candidate for best training scheme.

![loss vs step 256 0 006](https://user-images.githubusercontent.com/48018617/195250701-66f90036-ee57-41be-8c2f-409a3f0d5dea.png)
![valid acc vs step 256 0 006](https://user-images.githubusercontent.com/48018617/195250707-3f97dfdb-d6ca-4748-8897-17e88535afa8.png)

Based on this positive result of acheiving 96.44% accuracy with the same early stopping rules (3 epochs = 102 steps) as above, of stopping after 3 epochs, we can try batches of 512 if we can run them, as well as automated learning rate scheduling when learning stagnates. Also noteworthy is the lack of overfitting, so longer runs should be possible with no consequence.

To summarise; the up to this point best model training scheme is batch size 256, learning rate 0.006 with no scheduling, and seemingly we should let the training go for longer with a more relaxed early stopping guideline of >3 epochs (>102 steps). 

Luping CVPR paper uses the equivalent to ReduceLROnPlateau. Therefore we implemented it.

When we add on ReduceLROnPlateau sheduling, ie lowering the learning rate by gamma when the validation accuracy stagnates for 5 epochs, and extending learning time with longer ten epoch early stopping, we get great results at batch size 128. We expect this trend to continue for final runs at batch size 256 or 512.

![ReduceLROnPlateau](https://user-images.githubusercontent.com/48018617/195511198-9e5c017a-e0e0-400e-9ac0-740db8cde399.png)
![ReduceLROnPlateauloss](https://user-images.githubusercontent.com/48018617/195511216-86b02053-24ef-4fb5-9982-ac9a40649da5.png)
![ReduceLROnPlateaulr](https://user-images.githubusercontent.com/48018617/195512514-277a81ce-dbe6-40f6-87d1-2811d465b569.png)

This method of reducing learning rate at epoch landmarks, as well as early stopping as a function of epochs and not steps makes the learning regime less discriminative to batch size, as we see batches of 128 performing excellently.

According to the Luping CVPR paper, rotation is a key data augmentation that can be useful. We implemented it. The Luping paper sees MCA gains from MCA 88.58%, ACA 89.04% with no augmentation, to MCA 96.76%, ACA 97.24% with small rotational steps. We do not see such radical gains. In the best of 10 runs each, randomly rotating and flipping improves scores by less than 0.3% at most. The variance of all schemes is high, with all data augmentation schemes having runs with failed learning, getting caught in local minima at <93%.

![Data aug](https://user-images.githubusercontent.com/48018617/196410142-2bb8f1ef-34ad-4525-a79b-0ed0b9f9d75e.png)

The caveat here is that random rotation doubles the step time to ~4s per iteration, which doubles training time to 4 hours per run.

We thought we should report mean class accuracy as well as average classification accuracy, so we implemented this into the statistics collection.

![ACAMCA](https://user-images.githubusercontent.com/48018617/196411174-f1e09262-2a7e-4946-bf75-023913002888.png)

We see very little difference between the metrics, but notice in our implemented verbose output of per class accuracies reported for each class that the Golgi cell class seems the most problematic to train for. The class accuracy of Golgi generally lags behind the other classes, and for a majority of the training is is much less than 20% even while the other classes have accuracies above 95%.

The results at batch sizes of 256 and 512 is not more impressive, and we think that this is because of slower convergence at larger batch sizes. To remedy this we can try altering batch sizes on the fly to speed up convergence. https://arxiv.org/abs/1712.02029v2

When we implemented this via accumulating gradients in the original paper, we reached 98% accuracy and will report the final accuracy once the run concludes. Another attempt was run in parallel when we got access to more computation at a starting batch size of 2048.

As an extension, we used Big Transfer; a version of a ResNet-50 with included Group Normalisation trained on ImageNet21k and the BiT hyperule of batch size 512, 500 steps with learning rate warmup to 0.003 and then specific cooldown; achieved similar performance to our best every run in around 2 mins.


2016 sota was ~84.63% (MCA?) https://qixianbiao.github.io/HEp2Cell/

https://github.com/KaikaiZhao/HEp-2_cell_classification has strong VGG performance of 98%




We can take inspiration from KaikaiZhao for simplifying the 4096-4096 fc layers in the classifier, potentially have a drop+relu-9216-6 or a drop+relu-9216-4096-6

```shell
(classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
```
could become
```shell
(classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=6, bias=True)
  )
```
or if that is too limiting
```shell
(classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
```
with different sizes of the hidden layer.

## Final model

We aggregate all our runs and show graphs of the best examples from each combination of hyperparameters.
https://tableconvert.com/csv-to-markdown
