# Machine Learning Regression
I have implemented a general-purpose regression model fitting pipeline.  I have trained my model with 2 variants namely Multi-Linear Regression and Binary Logistic Regression.

## Project structure
* **MLW_Regression/main.c**:  contains the solution of the project.
* **CSV**: directory to store CSV files.
* **Input/CSVInput.txt**: loading given CSV file into the program.
* **OutputLog**: : directory of output logs of the program.

## Project Flow
*  The user is asked to either apply Multi-Linear regression or Binary Logistic Regression.
*  After selecting the type of regression, it loads data from the given CSV file mentioned in CSVInput.txt.
*  Then the user is asked to enter the proportion to split the given dataset into train and test.  For example, if the user enters 0.8 then 80% of the
   dataset is used for training and the remaining 20% for testing. 
*  The user is also asked to enter the learning rate and the number of iterations.
*  The given regression function is executed and logs entire data to the OutputLog directory and prints some relevant data on the console.

## Deployment
* In Multi-Linear Regression, There are 2 macros namely **MULTI_LINEAR_ROWS** and **MULTI_LINEAR_NUMBER_OF_XS** at the top of the solution. Here, the user has to input the number of rows in the CSV file to process and the number of columns dedicated for X. Usually it is: (Total number of columns - 1).
*  The same as above for Binary Logistic Regression.
*  Before the CSV file is read successfully, if the user wants to use another CSV file, he has to replace the sscanf s function with the given number     of X arguments and a single Y argument.


## Authors
* **Samuel Menezes**
