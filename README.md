
```markdown
# Car Price Prediction

This project is designed to predict car prices using an XGBoost model. It includes data preprocessing, model training, and prediction steps. The model is trained on a dataset containing various features of cars, and it can be used to estimate the price of a car based on user-provided input data.

## Table of Contents

- [Prerequisites]
- [Getting Started]
- [Usage]
- [Model Evaluation]
- [Making Predictions]
- [Author]
## Prerequisites

Before running the car price prediction script, you need to have the following prerequisites in place:

- **Python:** Make sure you have Python installed on your system.
- **Libraries:** Install the required Python libraries mentioned in the script. You can typically use `pip` to install them.

```bash
pip install pandas scikit-learn xgboost joblib
```

## Getting Started

1. **Clone the Repository:**
   Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/car_price_prediction.git
   ```

2. **Data:**
   Ensure that you have the dataset you want to use for car price prediction. In this example, the dataset is 'turboaz_27_09_2023.csv'.

3. **Run the Script:**
   Open a terminal, navigate to the project's root directory, and run the script:

   ```bash
   python main.py
   ```

   This script will preprocess the data, train the XGBoost model, and display model evaluation metrics.

## Usage

### Model Training

The script `main.py` performs the following tasks:

- **Data preprocessing:** It cleans and encodes the dataset, preparing it for training.
- **Model training:** It uses an XGBoost model to predict car prices.

### Making Predictions

The script can also make predictions for a new data point. To do this:

1. Update the `new_data_point` dictionary in the script with the features of the car you want to predict the price for.
2. Run the script again to make predictions for the new data point.

## Model Evaluation

After training the model, the script will display the following model evaluation metrics:

- **R-squared (R²):** A measure of the model's goodness-of-fit.
- **Root Mean Squared Error (RMSE):** A measure of the prediction error.
- **Mean Absolute Error (MAE):** A measure of the absolute prediction error.

## Author

- Ismat Samadov
- Email: ismetsemedov@gmail.com

```
