import numpy as np
from codex.loglinear import softmax
from codex.utils import one_hot_vector
from collections import namedtuple

STUDENT={'name': 'Vladimir Balagula',
         'ID': '323792770'}



def classifier_output(x, params):
    # YOUR CODE HERE.
    out = x.copy()
    layers_num = int(len(params)/2)
    for i in range(0, layers_num-1, 2):
        W, b = params[i], params[i+1]
        out = np.tanh(np.dot(x, W) + b)
    U, b_tag = params[-2], params[-1]
    return softmax(np.dot(out, U)+b_tag)


def predict(x, params):
    return np.argmax(classifier_output(x, params))

def classifier_with_cache(x, params):
    # YOUR CODE HERE.
    out = x.copy()
    temp_storage = []
    layers_num = int(len(params) / 2)
    for i in range(0, layers_num - 1, 2):
        W, b = params[i], params[i + 1]
        temp_storage.append(single_feed_forward(out, [W, b]))
        out = temp_storage[-1][1]
    U, b_tag = params[-2], params[-1]
    return softmax(np.dot(out, U) + b_tag), temp_storage

def single_feed_forward(x, params):
    W, b = params
    z = np.dot(x,W)+b
    a = np.tanh(z)
    return z, a


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    y_tag, storage = classifier_with_cache(x, params)
    loss = -np.log(y_tag[y])

    y = one_hot_vector(len(y_tag), y)
    storage.reverse()
    params.reverse()

    prev_derive = y_tag - y
    gradients = []
    derive = 0
    for i in range(len(storage)+1):
        W = params[i*2+1]
        if i != len(storage):
            prev_layer = storage[i][1] # z(x)
            derive = 1 - np.tanh(storage[i][0])**2 # dz
        else:
            prev_layer = x
        gW = np.outer(prev_layer, prev_derive)
        gb = prev_derive
        gradients += [gb, gW]
        if i != len(storage):
            prev_derive = np.dot(W, prev_derive)*derive
    params.reverse()
    return loss, gradients[::-1]

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []

    for i in range(len(dims)-1):
        W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2 / (dims[i] + dims[i+1]))
        b = np.random.randn(dims[i+1]) * np.sqrt(2 / (dims[i] + dims[i+1]))
        params += [W, b]

    return params

if __name__ == '__main__':
    from codex.grad_check import gradient_check
    params = create_classifier([3, 2, 4])
    [W, b] = params[0]
    [U, b_tag] = params[1]

    def _loss_and_W_grad(W):
        global b, U, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W, U, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[1]


    def _loss_and_U_grad(U):
        global W, b, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[2]


    def _loss_and_b_tag_grad(b_tag):
        global W, U, b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[3]


    for _ in range(10):
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)
