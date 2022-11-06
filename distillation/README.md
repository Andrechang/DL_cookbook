## Model distillation

Here we study model distillation with a very simple example and code implementation

### Install libs

It uses [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch), which has various implementation of ViT and train methods in simple to understand code.
And it is example based on: [cats\_and\_dogs](https://www.kaggle.com/code/reukki/pytorch-cnn-tutorial-with-cats-and-dogs/notebook)

Thus, install vit-pytorch: 

`pip install vit-pytorch`

Also install all the common ML libaries: torch, pandas, sklearn, tensorboard, etc 
You can use for dependencies libs: `pip install -r requirements.txt`

### Download data

Download train and test data:
[train](https://www.kaggle.com/code/reukki/pytorch-cnn-tutorial-with-cats-and-dogs/data?select=train.zip)
[test](https://www.kaggle.com/code/reukki/pytorch-cnn-tutorial-with-cats-and-dogs/data?select=test.zip)

Unzip the train and test files and put them into a `data/` folder

### Download teacher

The teacher is a resnet50 finetuned for cat and dog data. You can download here:
[teacher](https://www.dropbox.com/s/966lyanwi9m1xoo/teach_model.pth?dl=0)

Or you can train your own teacher model if you like.

### Run

This will train a ViT from resnet50 teacher 

`python3 cats_and_dogs_finetune_distil.py` 


