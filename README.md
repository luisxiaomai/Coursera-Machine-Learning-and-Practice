# Coursera-Machine-Learning-and-Practice

*A study recording of [Coursera's Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning), but added some python practice for 
reinforceing learning.*

## Table of Contents

  1. [Week1](#week1)
  1. [Week2](#week2)
  1. [Week3](#week3)
  1. [Week4](#week4)
  1. [Week5](#week5)
  1. [Week6](#week6)
  1. [Week7](#week7)
  1. [Week8](#week8)
  1. [Week9](#week9)
  1. [Week10](#week10)
  1. [Week11](#week11)
## Week1

  - **Introduction** 
    - `Machine Learning definition`:A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E
    - `Supervised learning`:"Right answer given" e.g. Regression, Classification...
    - `Unsupervised learning`:"No right answer given" e.g. Clustering, Gradient descent...
  - **Linear Regression with One Variable**
    - `Model representation`
    - `Cost function`
    - `Gradient Descent`
  - **Linear Algebra Review**
  - **Python Practice**
    > 1.PREDICTING HOUSE PRICES
    
    We have the following dataset:
    
    | Entry No.  | Square_Feet | Price  |
    | -----------|:-----------:| -----: |
    | 1          | 150         | 6450   |
    | 2          | 200         | 7450   |
    | 3          | 250         | 8450   |
    | 4          | 300         | 9450   |
    | 5          | 350         | 11450  |
    | 6          | 400         | 15450  |
    | 7          | 600         | 18450  |

    With linear regression, we know that we have to find a linearity within the data so we can get θ0 and θ1
    Our hypothesis equation looks like this:
    
    ![alt text]( https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week1/Hypothesis.png)
    
    **Where**
    - hθ(x) is the value price (which we are going to predicate) for particular square_feet  (means price is a linear function of square_feet)
    - θ0 is a constant
    - θ1 is the regression coefficient

    **Coding**
    - See week1 python codes which used Python Packages for Data Mining like NumPy, SciPy, Pandas, Matplotlib, Scikit-Learn to implement it.
    - Script Output:
    
    ![alt text]( https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week1/linear_line.png)
    
    > 2.PREDICTING WHICH TV SHOW WILL HAVE MORE VIEWERS NEXT WEEK
    
    The **Flash** and **Arrow** are American television series, each one is popular with people. It's interesting that which one will ultimately win the ratings war, so lets write a program which predicts which TV Show will have more viewers.
    
    We have the following dataset:
    
    | FLASH_EPISODE  | FLASH_US_VIEWERS | ARROW_EPISODE  | ARROW_US_VIEWERS |
    | -----------    |:-----------:     | -----:         | -----: |
    | 1              | 4.83             | 1              | 2.84 |
    | 2              | 4.27             | 2              | 2.32 |
    | 3              | 3.59             | 3              | 2.55 |
    | 4              | 3.53             | 4              | 2.49 |
    | 5              | 3.46             | 5              | 2.73 |
    | 6              | 3.73             | 6              | 2.6  |
    | 7              | 3.47             | 7              | 2.64 |
    | 8              | 4.34             | 8              | 3.92 |
    | 9              | 4.66             | 9              | 3.06 |
    **Steps to solving this problem:**
    - First we have to convert our data to X_parameters and Y_parameters, but here we have two X_parameters and Y_parameters. So, lets’s
    - name them as flash_x_parameter, flash_y_parameter, arrow_x_parameter , arrow_y_parameter.
    - Then we have to fit our data to two different  linear regression models- first for Flash, and the other for Arrow.
    - Then we have to predict the number of viewers for next episode for both of the TV shows.
    - Then we can compare the results and we can guess which show will have more viewers.
   
    **Coding**
    - See week1 python codes which used Python Packages for Data Mining like NumPy, SciPy, Pandas, Matplotlib, Scikit-Learn to implement it.
    - Script Output:
    
    
  **[⬆ back to top](#table-of-contents)**

## Week2

## Week3

## Week4

## Week5

## Week6

## Week7
  
## Week8

## Week9

## Week10

## Week11
