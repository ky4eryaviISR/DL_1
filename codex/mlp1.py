from loglinear import softmax
from utils import *
import numpy as np

STUDENT={'name': 'Vladimir Balagula',
         'ID': '323792770'}


def classifier_output(x, params):
    [W, b, U, b_tag] = params
    probs = softmax(np.tanh(np.dot(x, W)+b).dot(U)+b_tag)
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    y_tag = classifier_output(x, params)
    loss = -np.log(y_tag[y])
    W, b, U, b_tag = params
    y = one_hot_vector(len(y_tag), y)

    hid_out = np.tanh(np.dot(x, W) + b)

    gU = np.outer(hid_out, y_tag - y)
    gb_tag = y_tag - y

    dl_dz = np.dot(U, y_tag-y)
    dz_dh = 1.0 - np.tanh(np.dot(x, W) + b)**2
    dl_dh = dl_dz*dz_dh
    gW = np.outer(x, dl_dh)
    gb = dl_dh
    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    eps = np.sqrt(6.0 / (in_dim + hid_dim))
    W = np.random.uniform(-eps, eps, (in_dim, hid_dim))
    b = np.random.uniform(-eps, eps, hid_dim)

    eps = np.sqrt(6.0 / (out_dim + hid_dim))
    U = np.random.uniform(-eps, eps, (hid_dim, out_dim))
    b_tag = np.random.uniform(-eps, eps, out_dim)

    params = [W, b, U, b_tag]
    return params


if __name__ == '__main__':
    from codex.grad_check import gradient_check

    W, b, U, b_tag = create_classifier(3, 8, 4)

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
        global b, U, W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[3]

    in_dim = 3
    hid_dim = 8
    out_dim = 4
    for _ in range(10):
        W = np.random.randn(in_dim,hid_dim)
        b = np.random.randn(hid_dim)
        U = np.random.randn(hid_dim,out_dim)
        b_tag = np.random.randn(out_dim)
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)
