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

Install a conda env or your choice of sandboxed environment, yml files included.
We just need python=3.x, pip, pytorch, torchvision, maybe some numpy and pandas.

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

With an initial understanding of the landscape of the learning rate and batch size selections, we move on to more specific grid searches of longer duration, extending early stopping to 100 steps. 100 steps still only represents just 0.735 or 1.47 epochs respectively for batch sizes 64 and 128 (based on the 8707 image training set). We can standardise the testing to account for this, making the early stopping a function of batch size, and early stopping if there is no learning for 3 epochs (408 and 204 steps each for batch sizes 64 and 128).
