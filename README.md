# 🏠 California Housing Price Prediction Project 💰

## Overview

This project predicts California housing prices using machine learning. It covers the full data science lifecycle from **data loading** to **model evaluation**, and compares several machine learning models.

- **Data Loading** 📥
- **Exploration (EDA)** 🔍
- **Model Training** 🏋️
- **Evaluation** 📊
- **Visualization** 📈
- **Model Comparison** 🏆

## Dataset

* The **California Housing dataset** from `sklearn.datasets` is used. This dataset contains information about housing in California, such as **median income, house age, average number of rooms,** and the **target variable** (median house value).
* You can load it using:

    ```python
    from sklearn.datasets import fetch_california_housing
    housing_data = fetch_california_housing()
    ```

## Key Technologies

* **Python** 🐍
* **Pandas** 🐼
* **NumPy** 🔢
* **Scikit-learn** 🤖
* **Matplotlib** 📉
* **Seaborn** 🎨
* **XGBoost** 🚀

## Setup

1. **Clone the repository** to your local machine:

    ```bash
    git clone <your-repo-link>
    cd california-housing-prediction
    ```

2. **Install the required libraries**:

    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, if you don't have `requirements.txt`, use the following:

    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn xgboost
    ```

3. **Run the notebook** `california_housing_model.ipynb` in Jupyter Notebook or any Python environment.

## Data Exploration

* Loaded data into a **Pandas DataFrame**. You can inspect the first few rows by using:

    ```python
    house_price_dataframe.head()
    ```

* Checked the **shape** of the data:

    ```python
    house_price_dataframe.shape
    ```

* **Handled missing values** (none found).
* Analyzed **data statistics**:

    ```python
    house_price_dataframe.describe()
    ```

* **Visualized feature correlations** using a heatmap:

    ```python
    sns.heatmap(correlation, annot=True)
    plt.show()
    ```

## Model Training

We trained and compared the following regression models:

* **Linear Regression** 📏 *(Baseline)*
* **Random Forest Regressor** 🌳 *(tuned with GridSearchCV)*
* **XGBoost Regressor** 🔥 *(tuned with GridSearchCV)*

Hyperparameter tuning was done using **GridSearchCV** to improve model performance.

## Model Evaluation

### Evaluation Metrics:

* **R² (R-squared)** 🎯
* **Mean Absolute Error (MAE)** 📉
* **Root Mean Squared Error (RMSE)** 📏

Residual plots were analyzed to evaluate model errors.

### Results

| Model             | R²    | MAE   | RMSE  |
| :---------------- | :---- | :---- | :---- |
| **XGBoost**       | 0.841 | 0.304 | 0.463 |
| **Random Forest** | 0.805 | 0.333 | 0.512 |
| **Linear Regression** | 0.601 | 0.536 | 0.733 |
 **XGBoost** provided the highest **R²** score (0.841) and had the lowest **MAE** and **RMSE**, outperforming both **Random Forest** and **Linear Regression** models.

* **Visualization** 📊: Scatter plots and bar plots were used to compare model performance.

## Visuals

### Actual vs Predicted Prices

Here’s a scatter plot showing the **Actual vs Predicted Prices**:

![Actual vs Predicted Prices](c:\Users\RISHABH\OneDrive\Desktop\California-House-Price-Pediction\Images\actual-predicted model comparison plot.png)  


### R² Comparison

Bar plot comparing the **R² score** of each model:

![R² Comparison](c:\Users\RISHABH\OneDrive\Desktop\California-House-Price-Pediction\Images\R² model comparison.png)  
### RMSEComparison

Bar plot comparing the **RMSE score** of each model:

![RMSE Comparison](c:\Users\RISHABH\OneDrive\Desktop\California-House-Price-Pediction\Images\RMSE model comparison.png)  
### MAE Comparison

Bar plot comparing the **MAE score** of each model:

![MAE Comparison](c:\Users\RISHABH\OneDrive\Desktop\California-House-Price-Pediction\Images\MAE model comparison.png)  

## Next Steps

* **Model Interpretability**: We plan to integrate **SHAP** to explain model predictions. 💡
* Experiment with **additional models** like **Gradient Boosting** or **Neural Networks**.
* **Hyperparameter optimization** via **RandomizedSearchCV** or **Bayesian optimization**.

## Author

This project was created by **[Rishabh Yadav]**.  


---

### 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
