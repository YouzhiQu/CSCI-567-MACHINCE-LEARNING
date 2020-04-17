import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        y = np.where(y == 0, -1, 1)
        transpose_x = np.transpose(X)
        for time in range(max_iterations):
            condition = (np.dot(w, transpose_x) + b) * y
            calculate = np.where(condition <= 0, 1, 0)
            w += step_size * (np.dot((calculate * y), X))/N
            b += step_size * (np.sum(calculate * y))/N
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        y = np.where(y == 0, -1, 1)
        transpose_x = np.transpose(X)
        for time in range(max_iterations):
            condition = -1 * (np.dot(w, transpose_x) + b) * y
            calculate = sigmoid(condition)
            w += step_size * (np.dot((calculate * y), X)) / N
            b += step_size * (np.sum(calculate * y)) / N
        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(-z))
    ############################################
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        # preds = np.zeros(N)
        transpose_x = np.transpose(X)
        preds = (np.dot(w, transpose_x) + b)
        preds = np.where(preds > 0, 1, 0)
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        #preds = np.zeros(N)
        transpose_x = np.transpose(X)
        preds = (np.dot(w, transpose_x) + b)
        preds = np.where(preds > 0, 1, 0)
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        # np.random.seed(1)
        b_transpose = np.transpose(b)
        for time in range(max_iterations):
            index = np.random.choice(N)
            new_x = X[index]
            # new_x_transpose -> D*1
            new_x_transpose = np.transpose(new_x)
            # w_x -> C*D dot D*1 -> C*1
            w_x = np.dot(w, new_x_transpose) + b_transpose
            w_x -= w_x.max()
            w_x = np.exp(w_x)
            p_x = w_x / np.sum(w_x)
            p_x[y[index]] -= 1
            w -= step_size * np.dot(p_x.reshape(C, 1), new_x.reshape(1, D))
            b -= step_size * p_x

        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        w = np.zeros((C, D))
        b = np.zeros(C)
        step_size /= N

        onehot_y = np.zeros((N, C))
        onehot_y[np.arange(N), y] = 1
        onehot_y = np.transpose(onehot_y)
        for time in range(max_iterations):
            x_transpose = np.transpose(X)
            w_x = np.dot(w, x_transpose) + b.reshape(C, 1)
            w_x = np.exp(w_x)
            p_x = w_x / np.sum(w_x, axis=0)
            new_px = p_x - onehot_y
            # w -> C*N N*D
            w -= step_size * np.dot(new_px, X)
            b -= step_size * np.sum(new_px, axis=1)
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    #preds = np.zeros(N)
    result = np.dot(X, w.T) + b
    preds = np.argmax(result, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds




        