# import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# make x,y data for linear regression with some noise

x=np.array([1,1.5,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,4,7,8,8.8,10])+np.random.randint(10)


# mean

def get_mean(arr):
    return np.sum(arr)/len(arr)

# variance

def get_variance(arr,mean):
    return np.sum((arr-mean)**2)

# covariance

def get_covariance(arr_x,x_mean,arr_y,y_mean):
    final_arr=(arr_x-x_mean)*(arr_y-y_mean)
    return np.sum(final_arr)


# coefficients

def get_coefficients(x,y):
    x_mean=get_mean(x)
    y_mean=get_mean(y)
    m=get_covariance(x,x_mean,y,y_mean)/get_variance(x,x_mean)
    b=y_mean-x_mean*m
    return m,b


# regression

def linear_regression(x_train,y_train,x_test,y_test):
    prediction=[]
    m,b=get_coefficients(x_train,y_train)
    for x in x_test:
        y=m*x+b
        prediction.append(y)
    print(prediction)
    return prediction

# linear regression for 90% train data and 10% test data

def main():
    x_train=x[:9]
    y_train=y[:9]
    x_test=x[1:]
    y_test=y[1:]
    prediction=linear_regression(x_train,y_train,x_test,y_test)
    plt.scatter(x_train,y_train)
    plt.scatter(x_test,y_test)
    plt.plot(x_test,prediction)
    plt.show()
 
if __name__=="__main__":
    main()
