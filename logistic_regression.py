# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:45:13 2020

@author: harshitm
"""


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import math

def normalize(x):
    meanX1 = np.mean(x[0])
    varX1 = np.std(x[0])
    meanX2 = np.mean(x[1])
    varX2 = np.std(x[1])
    x[0] = (x[0] - meanX1) / varX1
    x[1] = (x[1] - meanX2) / varX2
    return x

"""
Sigmoid function used in logistic regression.
"""
def get_sigmoid_val(x, theta):
    h_theta_x = 1/(1 + np.exp( -(np.dot(x, theta))))
    return h_theta_x

"""
Using newton method to maximize the log likelyhood function here.
"""

def logistic_regression(x, y):
  DELTA_THETA = 1e-6 
  converged = False
  count = 0    
  x_comb =np.c_[np.ones([x.shape[0],1]), x] #adding extra intercept term
  thetas = np.zeros([x_comb.shape[1], 1])
  thetas_list = [np.zeros((thetas.shape))] #to record all the theta values.
  while converged == False:
      count = count + 1
      h_theta = get_sigmoid_val(x_comb, thetas)
      del_theta = np.dot(x_comb.T, (h_theta - y))
      h_theta.resize(h_theta.shape[0])
      hessian = x_comb.T.dot(np.diag(((1 - h_theta)*h_theta))).dot(x_comb)
      temp = np.dot(np.linalg.inv(hessian), del_theta)
      
      thetas = thetas - temp
      thetas_list.append(thetas)
      if((count > 1) and (np.abs(thetas_list[count][0] - thetas_list[count - 1][0]) < DELTA_THETA) and (np.abs(thetas_list[count][1] - thetas_list[count - 1][1]) < DELTA_THETA) and (np.abs(thetas_list[count][2] - thetas_list[count - 1][2]) < DELTA_THETA)):
          converged = True
          
  print("Thetas are : ", thetas[:,0])
  print("Number of iterations: ", count)
  return thetas

def plot_graph(x, y, thetas):
    label0x1, label0x2 = x[np.where(y==0)[0]].T
    label1x1, label1x2 = x[np.where(y==1)[0]].T
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(label0x1, label0x2, s=10, label="0", color="green")
    plt.scatter(label1x1, label1x2, s=10, label="1", color="red")
    
    xtest = np.linspace(-3, 3)
    plt.plot(x, eval(str(thetas[0]/(-thetas[2])) +"+"+ str(thetas[1]/(-thetas[2]))+"*x"), color="blue")
    plt.legend()
    plt.title("LogisticRegression")
    plt.savefig("LogisticRegression.jpg")
    plt.show()

def main():
    #firstly read the data from the  file
    X_in = pd.read_csv("./logisticX.csv").values
    Y_in = pd.read_csv("./logisticY.csv").values
    #norlaize the data 
    X = normalize(X_in)
    #Evaluate the thetas and plot the decision boundary
    thetas = logistic_regression(X, Y_in)
    plot_graph(X, Y_in, thetas)
if __name__ == "__main__":
    main()
    
