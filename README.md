# Coursera-Machine-Learning-and-Practice

*A study recording of [Coursera's Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning), but added some python practice for 
reinforceing learning.*

## Table of Contents

  1. [Week1](#Week1)
  1. [Week2](#Week2)
  1. [Week3](#Week3)
  1. [Week4](#Week4)
  1. [Week5](#Week5)

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
    > PREDICTING HOUSE PRICES
    
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

  
  
