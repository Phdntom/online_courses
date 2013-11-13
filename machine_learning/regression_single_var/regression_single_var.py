import numpy as np
import matplotlib.pyplot as plt
import sys

def add_column(col, array, loc=0):
    '''
    Parameters
    ----------
    col:         col to be inserted into array
    array:       np array
    loc:         integer specify the position to insert array

    Returns
    -------
    array_new:   a new table with column added at position loc
    '''
    as_rows = array.transpose().tolist()
    as_rows.insert(loc, col)
    array_new = np.array(as_rows).transpose()
        
    return array_new

def h_theta(x, th):
    '''
    Parameters
    ----------
    x:          feature vector for a training example
    th:         current regression parameters

    Returns
    -------
    h(x,th), the hypothesis cost for example x with fit parameters th
    '''
    assert len(x) == len(th)
    return th[0] * 1.0 + th[1] * x[1]

def J(X, Y, th):
    '''
    Parameters
    ----------
    X:          ndarray of training examples
    Y:          ndarray of training values
    th:         current regression parameters

    Returns
    -------
    the cost J(X,Y,th) for the current parameters on training data X,Y
    '''
    summ = 0
    for x,y in zip(X,Y):
        summ += (h_theta(x, th) - y) ** 2
    return summ / (2.0 * len(X) )

def gradJ(X, Y, th, j):
    '''
    Parameters
    ----------
    X:          ndarray of training examples
    Y:          ndarray of training values
    th:         current value of parameters
    j:          the gradient is taken with respect to the jth data column

    Returns
    -------
    the gradient of J(X,Y,th) for the current parameters on training data X,Y
    
    '''
    summ = 0
    for x,y in zip(X,Y):
        summ += (h_theta(x,th) - y) * x[j]
    return summ / len(X)

def gradient_descent(X, Y, th, alpha, num_iters):
    '''
    '''
    J_history = np.zeros(num_iters)
    th_history = np.zeros( shape=(num_iters,2) )
    for i in range(num_iters):
        grad0 = gradJ(X,Y,th,0)
        grad1 = gradJ(X,Y,th,1)
        th[0] -= alpha * grad0
        th[1] -= alpha * grad1
        J_history[i] = J(X,Y,th)
        th_history[i] = th
    return J_history, th_history

if __name__ == "__main__":
    #        fname = 'ex1data1.txt'
    with open(sys.argv[1]) as fname:

        data = np.loadtxt(fname, delimiter=',', skiprows=1)
        x, y = data[:,0], data[:,1]

        plt.plot(x, y, 'rx')
        plt.ylabel('Profit in $10,000s')
        plt.xlabel('Population of City in 10,000s')
        plt.axis([4, 24, -5, 25])


        ones = np.ones(len(data))

        X = add_column(ones, data[:,0:-1])
        Y = data[:,-1]

        theta = np.zeros(2)
        # theta = np.zeros( len(X[0]) )
        iterations = 1500
        alpha = 0.01
       
        J_val, th_val = gradient_descent(X,Y,theta,alpha,iterations)



        t = np.arange(0, 26, .2)
        plt.plot( t, theta[0] + theta[1]*t, 'k--', linewidth = 3.0)
        plt.plot(x,y,'rx')
        plt.ylabel('Profit in $10,000s')
        plt.xlabel('Population of City in 10,000s')
        plt.axis([4,24,-5,25])

        plt.show()









