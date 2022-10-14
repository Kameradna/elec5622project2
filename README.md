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

(report results of 128 vs 256 vs 512)

We thought we should report mean class accuracy as well as average classification accuracy, so we implemented this into the statistics collection.

(report correspondence between ACA (generic accuracy) vs MCA (mean per-class accuracy))


According to the Luping CVPR paper, rotation is a key data augmentation that can be useful. We implemented it. The Luping paper sees MCA gains from MCA 88.58%, ACA 89.04% with no augmentation, to MCA 96.76%, ACA 97.24% with small rotational steps.

(report results of random rotate, random horizontal flip)


https://arxiv.org/abs/1712.02029v2
We can try altering batch sizes on the fly to speed up convergence.

2016 sota was ~84.63% (MCA?) https://qixianbiao.github.io/HEp2Cell/

https://github.com/KaikaiZhao/HEp-2_cell_classification has strong VGG performance of 98%
![image](https://user-images.githubusercontent.com/48018617/195846522-23f8b264-0277-49ab-9821-2971ec255bdf.png)


## Final model

We aggregate all our runs and show graphs of the best examples from each combination of hyperparameters.
https://tableconvert.com/csv-to-markdown

| **run id** | **base_lr** | **batch_size** | **test_acc** | **stop_reason** | **step** |
|------------|-------------|----------------|--------------|-----------------|----------|
| 0          | 0.002       | 32             | 65.52        | stagnate        | 104      |
| 1          | 0.003       | 128            | 68.64        | stagnate        | 75       |
| 2          | 0.0075      | 64             | 49.43        | stagnate        | 67       |
| 3          | 0.003       | 64             | 69.75        | stagnate        | 633      |
| 4          | 0.01        | 128            | 65.66        | stagnate        | 55       |
| 5          | 0.01        | 64             | 54.49        | stagnate        | 42       |
| 6          | 0.006       | 128            | 87.4         | stagnate        | 903      |
| 7          | 0.002       | 32             | 73.34        | stagnate        | 96       |
| 8          | 0.005       | 128            | 70.94        | stagnate        | 125      |
| 9          | 0.003       | 128            | 85.28        | stagnate        | 561      |
| 10         | 0.005       | 128            | 80.14        | stagnate        | 291      |
| 11         | 0.01        | 128            | 51.72        | stagnate        | 51       |
| 12         | 0.0075      | 128            | 74.76        | stagnate        | 59       |
| 13         | 0.003       | 64             | 69.42        | stagnate        | 588      |
| 14         | 0.003       | 128            | 82.44        | stagnate        | 381      |
| 15         | 0.005       | 32             | 43.95        | stagnate        | 72       |
| 16         | 0.01        | 64             | 52.55        | stagnate        | 50       |
| 17         | 0.01        | 64             | 50.81        | stagnate        | 47       |
| 18         | 0.0075      | 32             | 58.57        | stagnate        | 63       |
| 19         | 0.01        | 64             | 45.8         | stagnate        | 56       |
| 20         | 0.0075      | 128            | 79.36        | stagnate        | 238      |
| 21         | 0.005       | 128            | 79.35        | stagnate        | 117      |
| 22         | 0.003       | 128            | 71.22        | stagnate        | 172      |
| 23         | 0.003       | 64             | 57.01        | stagnate        | 298      |
| 24         | 0.006       | 256            | 96.44        | stagnate        | 1084     |
| 25         | 0.003       | 64             | 67.77        | stagnate        | 609      |
| 26         | 0.01        | 32             | 39.22        | stagnate        | 42       |
| 27         | 0.003       | 128            | 71.72        | stagnate        | 715      |
| 28         | 0.003       | 32             | 71.96        | stagnate        | 76       |
| 29         | 0.003       | 64             | 80.55        | stagnate        | 744      |
| 30         | 0.002       | 64             | 81.98        | stagnate        | 207      |
| 31         | 0.006       | 128            | 94.07        | stagnate        | 2044     |
| 32         | 0.01        | 32             | 40.18        | stagnate        | 49       |
| 33         | 0.005       | 64             | 75.31        | stagnate        | 279      |
| 34         | 0.005       | 64             | 66.85        | stagnate        | 269      |
| 35         | 0.003       | 64             | 81.61        | stagnate        | 645      |
| 36         | 0.005       | 32             | 42.8         | stagnate        | 74       |
| 37         | 0.003       | 32             | 54.26        | stagnate        | 48       |
| 38         | 0.003       | 128            | 72.55        | stagnate        | 294      |
| 39         | 0.003       | 128            | 93.01        | stagnate        | 1409     |
| 40         | 0.003       | 64             | 78.03        | stagnate        | 578      |
| 41         | 0.005       | 64             | 81.02        | stagnate        | 392      |
| 42         | 0.003       | 64             | 78.62        | stagnate        | 599      |
| 43         | 0.003       | 64             | 76.97        | stagnate        | 567      |
| 44         | 0.005       | 64             | 71.77        | stagnate        | 293      |
| 45         | 0.005       | 64             | 73.15        | stagnate        | 93       |
| 46         | 0.003       | 64             | 77.7         | stagnate        | 703      |
| 47         | 0.006       | 128            | 86.8         | stagnate        | 701      |
| 48         | 0.005       | 128            | 81.7         | stagnate        | 280      |
| 49         | 0.01        | 64             | 48.51        | stagnate        | 43       |
| 50         | 0.0075      | 128            | 77.89        | stagnate        | 134      |
| 51         | 0.003       | 64             | 74.25        | stagnate        | 1379     |
| 52         | 0.1         | 128            | 39.91        | stagnate        | 7        |
| 53         | 0.003       | 128            | 73.15        | stagnate        | 82       |
| 54         | 0.003       | 64             | 82.39        | stagnate        | 243      |
| 55         | 0.003       | 64             | 72.6         | stagnate        | 276      |
| 56         | 0.003       | 128            | 82.34        | stagnate        | 307      |
| 57         | 0.005       | 64             | 73.47        | stagnate        | 135      |
| 58         | 0.0075      | 32             | 47.03        | stagnate        | 46       |
| 59         | 0.003       | 64             | 69.47        | stagnate        | 612      |
| 60         | 0.005       | 128            | 69.15        | stagnate        | 88       |
| 61         | 0.003       | 64             | 78.34        | stagnate        | 572      |
| 62         | 0.006       | 128            | 95.27        | stagnate        | 1686     |
| 63         | 0.003       | 64             | 78.48        | stagnate        | 684      |
| 64         | 0.003       | 64             | 42.25        | stagnate        | 462      |
| 65         | 0.01        | 32             | 18.16        | stagnate        | 35       |
| 66         | 0.006       | 128            | 86.67        | stagnate        | 1069     |
| 67         | 0.003       | 64             | 59.21        | stagnate        | 56       |
| 68         | 0.003       | 64             | 80.32        | stagnate        | 275      |
| 69         | 0.005       | 64             | 77.38        | stagnate        | 311      |
| 70         | 0.01        | 128            | 65.88        | stagnate        | 68       |
| 71         | 0.003       | 64             | 62.44        | stagnate        | 183      |
| 72         | 0.003       | 128            | 83.77        | stagnate        | 468      |
| 73         | 0.01        | 128            | 64.32        | stagnate        | 80       |
| 74         | 0.003       | 64             | 79.82        | stagnate        | 266      |
| 75         | 0.003       | 128            | 72.46        | stagnate        | 282      |
| 76         | 0.003       | 64             | 80.04        | stagnate        | 200      |
| 77         | 0.003       | 64             | 50.11        | stagnate        | 607      |
| 78         | 0.003       | 32             | 66.12        | stagnate        | 82       |
| 79         | 0.0075      | 64             | 53.61        | stagnate        | 39       |
| 80         | 0.003       | 64             | 66.9         | stagnate        | 587      |
| 81         | 0.003       | 64             | 62.39        | stagnate        | 1022     |
| 82         | 0.003       | 128            | 96.05        | stagnate        | 1756     |
| 83         | 0.01        | 64             | 53.33        | stagnate        | 60       |
| 84         | 0.0075      | 32             | 48.32        | stagnate        | 36       |
| 85         | 0.003       | 128            | 80.92        | stagnate        | 711      |
| 86         | 0.003       | 64             | 77.38        | stagnate        | 188      |
| 87         | 0.006       | 128            | 83.68        | stagnate        | 1120     |
| 88         | 0.005       | 64             | 73.98        | stagnate        | 107      |
| 89         | 0.003       | 64             | 58.81        | stagnate        | 285      |
| 90         | 0.003       | 64             | 42.81        | stagnate        | 480      |
| 91         | 0.01        | 64             | 48.74        | stagnate        | 34       |
| 92         | 0.1         | 128            | 39.04        | stagnate        | 2        |
| 93         | 0.003       | 64             | 79.86        | stagnate        | 651      |
| 94         | 0.003       | 128            | 94.39        | stagnate        | 1625     |
| 95         | 0.0075      | 128            | 73.61        | stagnate        | 65       |
| 96         | 0.003       | 128            | 87.72        | stagnate        | 948      |
| 97         | 0.005       | 64             | 78.71        | stagnate        | 288      |
| 98         | 0.003       | 64             | 80.09        | stagnate        | 701      |
| 99         | 0.003       | 128            | 94.44        | stagnate        | 1109     |
| 100        | 0.005       | 64             | 44.23        | stagnate        | 168      |
| 101        | 0.005       | 128            | 71.4         | stagnate        | 303      |
| 102        | 0.003       | 128            | 67.77        | stagnate        | 15       |
| 103        | 0.005       | 64             | 67.22        | stagnate        | 334      |
| 104        | 0.01        | 64             | 60.05        | stagnate        | 140      |
| 105        | 0.005       | 32             | 64.19        | stagnate        | 141      |
| 106        | 0.003       | 64             | 72.09        | stagnate        | 563      |
| 107        | 0.003       | 128            | 92.64        | stagnate        | 1198     |
| 108        | 0.2         | 128            | 35.04        | stagnate        | 3        |
| 109        | 0.005       | 128            | 77.33        | stagnate        | 180      |
| 110        | 0.0075      | 64             | 70.62        | stagnate        | 62       |
| 111        | 0.005       | 128            | 74.02        | stagnate        | 156      |
| 112        | 0.01        | 32             | 35.64        | stagnate        | 31       |
| 113        | 0.003       | 128            | 76.42        | stagnate        | 625      |
| 114        | 0.005       | 64             | 79.31        | stagnate        | 240      |
| 115        | 0.003       | 128            | 79.96        | stagnate        | 375      |
| 116        | 0.01        | 32             | 41.66        | stagnate        | 45       |
| 117        | 0.005       | 64             | 65.75        | stagnate        | 51       |
| 118        | 0.003       | 64             | 63.82        | stagnate        | 512      |
| 119        | 0.005       | 32             | 62.39        | stagnate        | 57       |
| 120        | 0.002       | 32             | 49.8         | stagnate        | 55       |
| 121        | 0.003       | 64             | 71.91        | stagnate        | 611      |
| 122        | 0.003       | 64             | 78.67        | stagnate        | 693      |
| 123        | 0.0075      | 64             | 59.49        | stagnate        | 43       |
| 124        | 0.005       | 128            | 82.16        | stagnate        | 554      |
| 125        | 0.003       | 32             | 69.01        | stagnate        | 58       |
| 126        | 0.01        | 128            | 70.3         | stagnate        | 85       |
| 127        | 0.005       | 128            | 82.25        | stagnate        | 164      |
| 128        | 0.003       | 128            | 72.6         | stagnate        | 304      |
| 129        | 0.01        | 32             | 37.56        | stagnate        | 38       |
| 130        | 0.005       | 128            | 81.42        | stagnate        | 312      |
| 131        | 0.005       | 64             | 78.94        | stagnate        | 189      |
| 132        | 0.002       | 32             | 36.69        | stagnate        | 32       |
| 133        | 0.003       | 128            | 83.86        | stagnate        | 271      |
| 134        | 0.01        | 128            | 74.52        | stagnate        | 85       |
| 135        | 0.005       | 128            | 72.23        | stagnate        | 302      |
| 136        | 0.003       | 64             | 80.28        | stagnate        | 181      |
| 137        | 0.003       | 64             | 71.35        | stagnate        | 552      |
| 138        | 0.003       | 128            | 86.94        | stagnate        | 500      |
| 139        | 0.003       | 64             | 79.49        | stagnate        | 272      |
| 140        | 0.01        | 128            | 69.88        | stagnate        | 64       |
| 141        | 0.003       | 128            | 95.17        | stagnate        | 1493     |
| 142        | 0.01        | 32             | 36.05        | stagnate        | 44       |
| 143        | 0.003       | 128            | 85.33        | stagnate        | 313      |
| 144        | 0.003       | 128            | 81.93        | stagnate        | 422      |
| 145        | 0.003       | 128            | 70.8         | stagnate        | 297      |
| 146        | 0.005       | 128            | 77.1         | stagnate        | 197      |
| 147        | 0.006       | 128            | 94.99        | stagnate        | 1902     |
| 148        | 0.003       | 64             | 84.64        | stagnate        | 168      |
| 149        | 0.005       | 128            | 67.95        | stagnate        | 270      |
| 150        | 0.003       | 128            | 83.22        | stagnate        | 267      |
| 151        | 0.002       | 64             | 62.85        | keyboard        | 63       |
| 152        | 0.003       | 256            | 77.85        | keyboard        | 486      |
| 153        | 0.01        | 32             | 36.88        | stagnate        | 62       |
| 154        | 0.01        | 128            | 72.46        | stagnate        | 146      |
| 155        | 0.003       | 128            | 62.16        | stagnate        | 380      |
| 156        | 0.0075      | 32             | 47.73        | stagnate        | 43       |
| 157        | 0.003       | 64             | 78.39        | stagnate        | 314      |
