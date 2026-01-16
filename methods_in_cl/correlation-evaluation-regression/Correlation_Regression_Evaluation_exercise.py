from typing import List, Union, Tuple
import math  # for math.sqrt to calculate square root


def calculate_pearsons_r(
    x: List[Union[float, int]], y: List[Union[float, int]]
) -> float:
    """
    Calculate Pearson's correlation coefficient between two lists of numbers.

    Parameters:
    x (List[Union[float, int]]): The first list of numerical values.
    y (List[Union[float, int]]): The second list of numerical values.

    Returns:
    float: The Pearson correlation coefficient, which measures the linear correlation
           between the two lists. The value ranges from -1 to 1.

    Raises:
    ValueError: If the input lists do not have the same length.
    """
    if len(x) != len(y):
        raise ValueError("Input lists must have the same length.")

    r = 0  # 0 is placeholder to avoid errors
    """ ADD CODE HERE """

    return r  # Return the calculated correlation coefficient


def calculate_standard_deviation(v: List[Union[float, int]]) -> float:
    """
    Calculate the standard deviation of a list of numbers.

    Parameters:
    v (List[Union[float, int]]): A list of numerical values (integers or floats).

    Returns:
    float: The standard deviation of the input list.
    """

    std = 0  #  0 is placeholder to avoid errors
    """ ADD CODE HERE """

    return std


def calculate_regression_coefficients(
    x: List[Union[float, int]], y: List[Union[float, int]]
) -> Tuple[float, float]:
    """
    Calculate the coefficients (intercept and slope) for linear regression
    using the least squares method.

    Parameters:
    x (List[Union[float, int]]): A list of independent variable values.
    y (List[Union[float, int]]): A list of dependent variable values.

    Returns:
    Tuple[float, float]: A tuple containing the intercept (a) and slope (b) of the regression line.

    Raises:
    ValueError: If the input lists do not have the same length.
    """

    if len(x) != len(y):
        raise ValueError("Input lists must have the same length.")

    b = 0  #  0 is placeholder to avoid errors
    a = 0  #  0 is placeholder to avoid errors

    """ ADD CODE HERE """

    return a, b  # Return the coefficients


def predict(x: float, a: float, b: float) -> float:
    """
    Predict the value of the dependent variable using the regression coefficients.

    Parameters:
    x (float): The independent variable value for which to predict the dependent variable.
    a (float): The intercept of the regression line.
    b (float): The slope of the regression line.

    Returns:
    float: The predicted value of the dependent variable.
    """
    y_hat = 0  #  0 is placeholder to avoid errors

    """ ADD CODE HERE """

    return y_hat  # Return the predicted value


def calculate_R2(y_pred, y_true):
    """
    Calculate the R-squared (R2) value, which indicates the proportion of the variance
    in the dependent variable that is predictable from the independent variable(s).

    Parameters:
    y_pred (list or array-like): Predicted values from the regression model.
    y_true (list or array-like): Actual observed values.

    Returns:
    float: The R-squared value, ranging from 0 to 1, where 1 indicates perfect prediction.

    Raises:
    ValueError: If the input lists do not have the same length.
    """

    if len(y_pred) != len(y_true):
        raise ValueError("Input lists must have the same length.")

    r2 = 0  #  0 is placeholder to avoid errors
    """ ADD CODE HERE """

    return r2  # Return the R-squared value


def classify(y_pred: float) -> int:
    """
    Classifies the predicted value into binary classes.

    Args:
        y_pred (float): The predicted value.

    Returns:
        int: Returns 1 if the predicted value is greater than or equal to 0.5, otherwise returns 0.
    """
    y_class = 0  # 0 is placeholder to avoid errors
    """ ADD CODE HERE """

    return y_class


def calculate_accuracy(y_pred: List[int], y_true: List[int]) -> float:
    """
    Calculates the accuracy of predictions compared to true labels.

    Args:
        y_pred (List[int]): A list of predicted class labels (0 or 1).
        y_true (List[int]): A list of true class labels (0 or 1).

    Returns:
        float: The accuracy of the predictions as a float between 0 and 1.

    Raises:
    ValueError: If the input lists do not have the same length.
    """

    if len(y_pred) != len(y_true):
        raise ValueError("Input lists must have the same length.")

    accuracy = 0  # 0 is placeholder to avoid errors

    """ ADD CODE HERE """

    return accuracy


if __name__ == "__main__":

    x_train = [7, 3, 3, 0, 0, 1, 4, 2, 0, 1]
    y_train = [1.00, 0.39, 0.00, 0.00, 0.25, 0.00, 0.27, 0.17, 0.38, 0.12]

    x_test = [1, 5, 0, 0, 6, 2, 0, 1, 2, 4]
    y_test = [0.62, 0.45, 0.08, 0.55, 0.88, 0.30, 0.12, 0.20, 0.58, 0.43]

    # calculate the Pearson correlation coefficient on the training data
    r = 0  # 0 is placeholder to avoid errors, add code here
    print(f"Pearson's r: {round(r, 3)}")

    # calculate regression coefficients
    a, b = (0, 0)  # placeholders to avoid errors, add code here
    print(
        f"Intercept (a): {round(a, 3)}, Slope (b): {round(b, 3)} -> y = {round(a, 3)} + {round(b, 3)}x"
    )

    # get the predicted values for the training set
    y_pred = []
    """ ADD CODE HERE """

    # calculate and print R squared
    r2 = 0  # 0 is placeholder to avoid errors, add code here
    print(f"R squared: {round(r2, 3)}")

    # convert predictions and true labels to binary classes for accuracy calculation
    y_pred_classes = []
    y_true_classes = []
    """ ADD CODE HERE """

    # Calculate and return the accuracy of the predictions
    acc = 0  # 0 is placeholder to avoid errors, add code here
    print(f"Accuracy: {round(acc, 3)}")
