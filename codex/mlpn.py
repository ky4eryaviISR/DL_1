import numpy as np
from codex.loglinear import softmax
from codex.utils import one_hot_vector
from collections import namedtuple

STUDENT={'name': 'Vladimir Balagula',
         'ID': ''}



def classifier_output(x, params):
    # YOUR CODE HERE.
    out = x.copy()
    temp_storage = []
    for W, b in (zip(params[::2], params[1::2])):
        temp_storage.append(single_feed_forward(out, [W, b]))
        out = temp_storage[-1][1]

    return softmax(out), temp_storage


def predict(x, params):
    return np.argmax(classifier_output(x, params))


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
    y_tag, storage = classifier_output(x, params)
    loss = -np.log(y_tag[y])

    y = one_hot_vector(len(y_tag), y)
    storage.reverse()
    prev_layer = storage[0][1]

    gU = np.outer(prev_layer, y_tag-y)
    gb_tag = y_tag - y
    it = iter(storage)
    prev_dA = gU
    gradients = []
    gradients.append([gU, gb_tag])
    for i in range(len(params)-1, -1, -2):
        W = params[i-1]
        b = params[i]
        prev_layer = storage[-2][1]
        prev_dA = prev_dA*(np.tanh(np.dot(x, W) + b)**2)
        gW = np.outer(storage[-i][1], prev_dA)
        gb = prev_dA
        gradients.append([gW, gb])
    gradients.reverse()
    return loss, [x ,y]

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
        params.append((W, b))

    return params

if __name__ == '__main__':
    from codex.grad_check import gradient_check
    params = create_classifier([3, 8, 4])
    [W, b] = params[0]
    [U, b_tag] = params[1]
    # [[W2, b2], [W, b], [U, b_tag]] = params
    #
    # def _loss_and_W_grad(W):
    #     global W2, b2, b, U, b_tag
    #     loss, grads = loss_and_gradients([1, 2, 3], 0, params=[W2, b2, W, b, U, b_tag])
    #     return loss, grads[2]
    #
    # def _loss_and_b_grad(b):
    #     global W2, b2, W, U, b_tag
    #     loss, grads = loss_and_gradients([1, 2, 3], 0, [W2, b2, W, b, U, b_tag])
    #     return loss, grads[3]
    def _loss_and_W_grad(W):
        global b, U, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W, U, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[1]

    for _ in range(10):
        gradient_check(_loss_and_W_grad, W)
    #     gradient_check(_loss_and_U_grad, U)
    #     gradient_check(_loss_and_W_grad, W)
    #     gradient_check(_loss_and_b_grad, b)