# Coursera-Machine-Learning-and-Practice

*A study recording of [Coursera's Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning), but added some practices for reinforceing learning.*

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
    - `Machine Learning definition`: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E
    - `Supervised learning`: "Right answer given" e.g. Regression, Classification...
    - `Unsupervised learning`: "No right answer given" e.g. Clustering, Gradient descent...
    
  - **Linear Regression with One Variable**
    - `Model representation`
    - `Cost function`
    - `Gradient Descent`
    
  - **Linear Algebra Review**
  
  - **Python Practice for Simple Linear Regression**
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
    
    **Where:**
    - hθ(x) is the value price (which we are going to predicate) for particular square_feet  (means price is a linear function of square_feet)
    - θ0 is a constant
    - θ1 is the regression coefficient

    **Coding:**
    - See [week1 python codes](https://github.com/luisxiaomai/Coursera-Machine-Learning-and-Practice/tree/master/Week1/scipts/house_price) which used Python Packages for Data Mining like NumPy, SciPy, Pandas, Matplotlib, Scikit-Learn to implement it.
    - Script Output:
    
    ![alt text]( https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week1/linear_line.png)
    
    
    > PREDICTING WHICH TV SHOW WILL HAVE MORE VIEWERS NEXT WEEK
    
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
    - First we have to convert our data to X_parameters and Y_parameters, but here we have two X_parameters and Y_parameters. So, lets’s name them as flash_x_parameter, flash_y_parameter, arrow_x_parameter , arrow_y_parameter.
    - Then we have to fit our data to two different  linear regression models- first for Flash, and the other for Arrow.
    - Then we have to predict the number of viewers for next episode for both of the TV shows.
    - Then we can compare the results and we can guess which show will have more viewers.
   
    **Coding:**
    - See [week1 python codes](https://github.com/luisxiaomai/Coursera-Machine-Learning-and-Practice/tree/master/Week1/scipts/TV_viewers) which used Python Packages for Data Mining like Pandas, Scikit-Learn to implement it. See 
    - Script Output:
    The flash TV show will have more viewer for next week9.
    
  **[⬆ back to top](#table-of-contents)**

## Week2

  - **Linear Regression with Multiple Variables** 
    - `Multiple features`
    - `Gradient Descent for multiple variables`
    - `Feature scaling`: Make sure features are on a similar scale.
    - `Learning rate`: Making sure gradient descent is working correctly.
      * If α is too small: slow convergence.
      * If α is too large: J(θ) may not decrease on every every iteration; may not converge
    - `Features and Polynomial Regression`    
   
  - **Computing Parameters Analytically**
    - `Normal equation`: Method to solve θ for analytically.
    - `Normal equation Noninvertibility`
    
  - **Octave Tutorial**
    - `Basic operation`
    - `Moving Data Around`
    - `Computing on Data`
    - `Plotting Data`
    - `Control statement: for, while, if statement`   
    - `Vectorization`   
    
  - **Octave Practice for Linear Regression**
   
     In this practice, we will implement linear regression and get to see it work
on data. See related [exercises and scripts](https://github.com/luisxiaomai/Coursera-Machine-Learning-and-Practice/tree/master/Week2/ex1) which used officeal coursera's exercise. All what we did in the exercises are summarised as below sections:

      > Linear regression with one variable
      
      In this part of this exercise, you will implement linear regression with onevariable to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities.

      **1. Plotting the Data**
      
      Before starting on any task, it is often useful to understand the data by visualizing it. For this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population).
      
      ![alt text](https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week2/plot1.png)

      **2. Gradient Descent**
      
      In this part, you will fit the linear regression parameters θ to our dataset using gradient descent.
      
      The objective of linear regression is to minimize the cost function
      
      ![alt text](https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week2/cost_function.png)
      
      where the hypothesis hθ(x) is given by the linear model 
      
      ![alt text](https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week2/hypothesis.png)
      
      Recall that the parameters of your model are the θj values. These arethe values you will adjust to minimize cost J(θ). One way to do this is to use the batch gradient descent algorithm. In batch gradient descent, each iteration performs the update
      
      ![alt text](https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week2/theta.png)
      
      With each step of gradient descent, your parameters θj come closer to the optimal values that will achieve the lowest cost J(θ).
      
      So after computing the cost function J(θ) and θ, you can plot the linear fit like below picture :
      
      ![alt text](https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week2/plot2.png)
      
      To understand the cost function J(θ) better, you will now plot the cost over a 2-dimensional grid of θ0 and θ1 values.

      Surface:
      
      ![alt text](https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week2/plot3.png)

      Contour:
      
      ![alt text](https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week2/plot4.png)

      > Linear regression with multiple variables
      
      In this part, you will implement linear regression with multiple variables to predict the prices of houses. Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to first collect information on recent houses sold and make a model of housing prices.

      **1. Feature Normalization**
      By looking at the data set avlue, note that house sizes are about 1000 times the number of bedrooms. When features differ by orders of mag-nitude, first performing feature scaling can make gradient descent converge much more quickly.
      
      * Subtract the mean value of each feature from the dataset.
      * After subtracting the mean, additionally scale (divide) the feature values by their respective \standard deviations."
      
      **2. Gradient Descent**
      
      Previously, you implemented gradient descent on a univariate regression problem. The only difference now is that there is one more feature in the matrix X. The hypothesis function and the batch gradient descent update rule remain unchanged.
      Also we can try out different learning rates for the dataset and find a learning rate that converges quickly.
      If you picked a learning rate within a good range, your plot look similar figure like below: 
      
      ![alt text](https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week2/learning_rate.png)

      **3. Normal Equationst**
      In the lecture videos, you learned that the closed-form solution to linear regression is
      
      ![alt text](https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week2/equation.png)
      
      Using this formula does not require any feature scaling, and you will get an exact solution in one calculation: there is no \loop until convergence" like in gradient descent.
      https://github.com/luisxiaomai/Images/blob/master/Machine-Learning/Week2/equation.png
     
   **[⬆ back to top](#table-of-contents)**
   
## Week3

## Week4

## Week5

## Week6

## Week7
  
## Week8

## Week9

## Week10

## Week11
