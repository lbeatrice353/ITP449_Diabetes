""" Bebe Lin
    ITP-449
    HW 8 - Diabetes Regression
    Creating a line of best fit of quantitative diabetes progression
"""
# importing the packages needed for this assignment
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# importing the specific parts of these packages
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main():
    # loading the file and seeing what's in it, making sure to skip the header row
    file_path = 'diabetes.csv'
    df_diabetes = pd.read_csv(file_path, skiprows=1)
    print(df_diabetes.info())

    # dropping dupes and empty records
    df_diabetes = df_diabetes.drop_duplicates()
    df_diabetes = df_diabetes.dropna()

    # making a correlation matrix to figure out which variables to keep
    corr_matrix = df_diabetes.corr(numeric_only=True)
    print(corr_matrix)
    # selecting the variables needed for this exercise
    df_diabetes = df_diabetes[['BMI', 'Y']]
    print(df_diabetes.describe())

    # setting the data for x and y
    x = df_diabetes['BMI']
    y = df_diabetes['Y']

    # reshaping the data so the dimensions match
    X = x.values.reshape(-1, 1)

    # reconvert back into pandas object to preserve metadata
    X = pd.DataFrame(X, columns=[x.name])

    # training the model
    model_linreg = LinearRegression()
    model_linreg.fit(X, y)

    # make predictions to create a line of best fit
    X_trend = np.array([[x.min()], [x.max()]])
    y_pred = model_linreg.intercept_ + model_linreg.coef_[0]*X_trend
    y_pred = model_linreg.predict(X_trend)

    # code needed to make the overall plot
    # setting up the subplot
    fig, ax = plt.subplots(1, 1)
    # scatter plot for the data points in the file
    ax.scatter(X, y, label='Data Points')  # Added label for data points
    # making the plot for the line of best fit and defining the color
    ax.plot(X_trend, y_pred, color='red', label='Line of Best Fit')  # Added label for the line of best fit
    # labeling the axes and title, adding a legend, and saving the figure
    plt.xlabel('BMI')
    plt.ylabel('Progression')
    plt.title('Diabetes Data: Progression vs. BMI (Linear Regression)')
    plt.legend()
    plt.savefig('diabetes_linear_regression.png')

if __name__ == '__main__':
    main()



