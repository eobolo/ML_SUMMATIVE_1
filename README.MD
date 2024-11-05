# TV Marketing Sales Prediction

This project demonstrates building a linear regression model to predict sales based on TV marketing expenses.

## Problem Statement

The goal is to predict sales based on TV marketing expenses using different machine learning techniques, including:

* Linear Regression
* Decision Trees
* Random Forests
* Gradient Descent Optimization

## Results

The project compares the performance of different scikit learn models using RMSE and ranks them accordingly.
and finally implementing gradient descent from scratch which does better than the rest of the previous
scikit-learn models

## Dataset

The project uses the `tvmarketing.csv` dataset, containing TV marketing expenses and sales data.

## Implementation

The project uses Python libraries like NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn for data analysis, visualization, and model building.
and finally a final implementation of gradient descent from scratch.

## Usage

1. Install the required packages: `pip install -r requirements.txt`
2. Run the Jupyter Notebook or Python script to train and evaluate the models in notebooks folder.

## FastAPI API

The FastAPI framework is used to create an API for making predictions.
to run it after creating your virtual environment in the previous directory
before the git repo

```cmd
pyton -m venv [virtual_envirorment_name]
[virtual_envirorment_name]\Scripts\activate
```
navigate to the github repo and run

```cmd
uvicorn api.main:app --reload
```

You can send a POST request to the `/predict` endpoint with a JSON payload.
from the browser swagger ui [API Documentation](http://127.0.0.1:8000/docs/)
and you get the response payload

```python
{
    "tv_sales": tv_sales_prediction
}
```
The API will return the predicted tv sales value or error if constraints are triggered.

## Further Development
* Explore additional features and datasets for improved prediction accuracy.
* Implement model deployment for real-time predictions.