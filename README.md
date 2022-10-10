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


