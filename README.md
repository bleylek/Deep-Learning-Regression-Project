# Deep Learning Regression Project

This repository contains a deep learning regression model that is developed to predict house prices using a dataset of housing features. The project leverages a neural network architecture to perform regression tasks, using TensorFlow and Keras as the primary libraries.

## Project Overview

This project aims to predict the price of houses based on various features such as the number of bedrooms, bathrooms, square footage, and other attributes. The data used for training the model is derived from the `kc_house_data.csv` dataset.

### Key Features of the Project
- **Data Exploration and Visualization**: The data is thoroughly analyzed and visualized using libraries such as Pandas, Seaborn, and Matplotlib to understand the relationships between features and the target variable (price).
- **Data Preprocessing**: Includes handling missing values, feature scaling, and removing irrelevant features to optimize the model's performance.
- **Model Architecture**: A sequential neural network model with multiple dense layers, using ReLU activation functions, is constructed to perform the regression task.
- **Model Evaluation**: The model's performance is evaluated using mean squared error (MSE) and visualized using loss plots.

## Data

The dataset used in this project is `kc_house_data.csv`, which contains various features of houses including:
- `id`
- `date`
- `price`
- `bedrooms`
- `bathrooms`
- `sqft_living`
- `sqft_lot`
- `floors`
- `waterfront`
- `view`
- `condition`
- `grade`
- `sqft_above`
- `sqft_basement`
- `yr_built`
- `yr_renovated`
- `zipcode`
- `lat`
- `long`
- `sqft_living15`
- `sqft_lot15`

### Prerequisites
To run this project, you need to have the following installed:
- Python 3.x
- Jupyter Notebook
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## Model Details

The model architecture consists of the following layers:
- Input Layer: Number of input features (17)
- Hidden Layers: Four dense layers with 17 neurons each and ReLU activation
- Output Layer: A single neuron output with linear activation to predict the house price

The model is compiled using the Adam optimizer and mean squared error (MSE) as the loss function.

### Model Training

The model is trained on the processed data with the following configurations:
- **Batch Size**: 128
- **Epochs**: 400
- **Validation Split**: 30% of the data used for validation

### Results

After training, the model's loss values are visualized, and the model's performance is evaluated based on the test set's MSE.

## Visualization and Analysis

Various plots, such as distribution plots, scatter plots, and box plots, are used to visualize the data and the model's performance.

## Conclusion

This deep learning regression project demonstrates the ability to predict house prices using a neural network model. It highlights the importance of data preprocessing, proper model selection, and evaluation in achieving accurate predictions.
