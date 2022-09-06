# Vehicle Trajectory Prediction

The project predicts the xy coordinates of the trajectory of vehicle 

## Description

The dataset is obtained from Argoverse 1 Motion Forecasting Dataset. 
The model takes the xy coordinates of the target agent for 45 seconds, and predict its future motion for 5 seconds
A linear model and a LSTM model is trained for prediction

## Getting Started

### Executing program

```
python run.py --model_type linear/LSTM
```
default model is linear model

### File Structures

- `notebook` file: 
  - `EDA_trajectory_prediction.ipynb`: Exploratory Data Analysis for training data
- `run.py` runs the entire project to train and validate the model
- `config.json` constructs the parameters for the deep learning models
- `dataloader.py` loads the data and reconstruct data into useable format
- `utils.py` construct early stopping for models to prevent overfitting
- `linear.pt` constains weights for linear model
- `lstm.pt` contains weights for LSTM model
- `models.py` constains the structure of linear and LSTM model
- `train.py` consists of model training and validation process

## Authors

Lehan Li ll3745@nyu.edu

Qiwen Zhang qz2274@nyu.edu


## Acknowledgments

* [dataset](https://www.argoverse.org/ )
