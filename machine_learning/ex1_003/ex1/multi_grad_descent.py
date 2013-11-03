import sys
import numpy as np
import matplotlib.pyplot as plt

def feature_normalize(data):
    '''
    Parameters
    ----------
    data:     list of list(rows)
    Returns
    -------
    normal_data:    list of list(rows) normalized
    mu
    sigma
    '''
    normal_data = data
    num_ex = len(data)
    num_feat = len(data[0])
    mu = np.zeros(num_feat)
    sigma = np.zeros(num_feat)
    
    # replace with np.mean, np.std
    for row in normal_data:
        for i in range(num_feat):
            mu[i] += row[i]
            sigma[i] += row[i]**2
    for i in range(num_feat):
        sigma[i] /= num_ex
        mu[i] /= num_ex
    for i in range(num_feat):
        sigma[i] -= mu[i]**2
        sigma[i] = np.sqrt(sigma[i])
        
    # normalize the data set so mean = 0, std = 1 for all features
    for row in range(num_ex):
        for i in range(num_feat):
            normal_data[row][i] -= mu[i]
            normal_data[row][i] /= sigma[i]
        
    return normal_data, mu, sigma

def h_theta(x,theta):
    '''
    Parameters
    ----------
    x:        feature vector for a training example
    theta:    parameter with dimension matching x
    '''
    #return theta[0]*1.0 + theta[1]*x[1]
    return np.dot(theta,x)

def J(X,y,theta):
    '''
    same as before
    '''
    m = len(y)
    summ = 0
    for i in range(m):
        summ += (h_theta(X[i],theta) - y[i])**2
    return summ / (2.0*m)

def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = np.zeros(num_iters)
    theta_history = np.zeros((num_iters,3))
    for i in range(num_iters):
        temp0 = gradj_J(X,y,theta,0)
        temp1 = gradj_J(X,y,theta,1)
        temp2 = gradj_J(X,y,theta,2)
        theta[0] = theta[0] - alpha*temp0
        theta[1] = theta[1] - alpha*temp1
        theta[2] = theta[2] - alpha*temp2
        J_history[i] = J(X,y,theta)
        theta_history[i] = theta
        print J_history[i], theta
    return J_history, theta_history

def gradj_J(X,y,theta,j):
    m = len(y)
    summ = 0
    for i in range(m):
        summ += (h_theta(X[i],theta) - y[i])*X[i][j]
    return summ / m
'''
def plotting(theta, data):
    t = np.arange(0,5,.1)

    m = len(data[0])

    ax1 = plt.subplot(121)
    ax1.plot(t,
'''
if __name__ == "__main__":
    fname = sys.argv[1]

    data = np.loadtxt(fname, delimiter=',')
    X_norm, mu, sigma = feature_normalize(data)
    
    theta = np.zeros(3)
    y = X_norm[:,2]
    
    alpha = .01
    iterations = 5000

    J_val, th_val = gradient_descent(X_norm, y, theta, alpha, iterations)

    print theta
