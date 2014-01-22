import numpy as np
import matplotlib.pyplot as plt
import sys

def feature_normalize(train_data):
    '''
    Parameters
    ----------
    train_data:        list of training examples; feature/target lists

    Returns
    -------
    norm_feat_data:    list of normalized example features lists
    mu                 the mean for each feature in train_data
    sigma              the stan dev for each feature in train_data
    '''
    norm_feat_data= train_data[:,:-1]
    num_ex = len(norm_feat_data)
    num_feat = len(norm_feat_data[0])
    #print num_feat
    mu = np.zeros(num_feat)
    sigma = np.zeros(num_feat)
    
    # replace with np.mean, np.std
    for row in norm_feat_data:
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
            norm_feat_data[row][i] -= mu[i]
            norm_feat_data[row][i] /= sigma[i]

    return norm_feat_data, mu, sigma

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
    h(x,th),the hypothesis cost for example x with fit parameters th
    '''
    return np.dot(th,x)

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
        summ += (h_theta(x,th) - y) ** 2
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

def gradient_descent_multi(X, Y, th, alpha, num_iters):
    '''
    '''
    features = len(th)

    J_history = np.zeros(num_iters)
    th_history = np.zeros( shape=(num_iters,features) )

    dJd = np.zeros(features)

    for i in range(num_iters):
        for j in range(features):
            dJd[j] = gradJ(X,Y,th,j)
        for j in range(features):
            th[j] -= alpha * dJd[j]
        J_history[i] = J(X,Y,th)
        th_history[i] = th
    return J_history, th_history

def normal_eq(X,Y):
    '''
    Parameters
    ----------
    X:          m x n matrix
    Y:          m x 1 matrix

    Returns
    -------
    th:         n x 1 matrix, solution to  (X.T X) th =  X.T Y
    .T is the transpose
    .inv is the inverse
    '''

    assert len(X) == len(Y)
    dot = np.dot
    solve = np.linalg.solve

    #X[0] = X[-1]
    #test = dot(X.T, X)
    #print test

    #print np.linalg.matrix_rank(test)

    #solve( test, dot(X.T,Y) )    

    return solve( dot(X.transpose(),X), dot(X.transpose(),Y) )

def predict(x,th):
    pass

if __name__ == "__main__":
    if len(sys.argv) > 2:
        alpha = float(sys.argv[2])
    else:
        alpha = 0.33
    #    fname = 'ex1data2.txt'
    print alpha
    with open(sys.argv[1]) as fname:
        data = np.loadtxt(fname, delimiter=',')
        norm_train, mu, sigma = feature_normalize(data)
        ones = np.ones( len(norm_train) )
        X = add_column(ones, norm_train)
        Y = data[:, -1]
        theta = np.zeros( len(X[0]) )

        iterations = 200
        J_hist, th_hist = gradient_descent_multi(X,Y,theta,alpha,iterations)
        #print J_hist
        plt.plot(J_hist)
        #plt.show()
        print theta
        print normal_eq(X,Y)


