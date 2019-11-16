import numpy as np
from codex.loglinear import softmax
from codex.utils import *

STUDENT={'name': 'Vladimir Balagula',
         'ID': '323792770'}

def classifier_output(x, params):
    # YOUR CODE HERE.
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
    # YOU CODE HERE
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

    eps = np.sqrt(6.0 / (hid_dim + out_dim))
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

    for _ in range(1):
        W = np.array([[-0.36057419,  0.76479367, -0.1875277 , -0.85998321, -0.98060785,
        -2.33460946,  0.15271204,  0.09238832],
       [ 0.19587871, -1.56650109,  1.53510745,  1.16082041,  1.04424334,
         1.57928536,  0.4175985 ,  1.0578678 ],
       [ 0.40758239, -1.04897485,  0.33338873,  0.81683527, -2.19201782,
        -1.38026268,  2.02652151, -1.38074707]])
        b = np.array([-0.33287235, -1.30054838, -0.5003385 ,  0.26568163, -0.27012993,-0.1064041 ,  0.84897605,  0.07916043])
        U = np.array([[-2.23955236,  1.45416519, -1.27313939,  1.88343393],
       [-0.67034757, -0.45168266, -0.15122679,  1.24261772],
       [-0.46371153, -2.36711967, -0.15273517, -0.65244235],
       [ 0.65152678, -0.70806281, -1.61905413, -0.95501001],
       [ 0.40302678, -0.35741077,  0.36389198, -1.43740199],
       [ 0.10896094,  1.00726652,  0.24565005,  2.50905021],
       [-0.0491517 , -1.41004767, -0.72373113,  1.02533632],
       [-1.2085249 ,  1.82082444,  0.20837019,  0.69402264]])
        b_tag = np.array([ 0.21836401,  2.04694079,  1.23373652, -0.46275933])
        #gradient_check(_loss_and_b_tag_grad, b_tag)
        #gradient_check(_loss_and_U_grad, U)
        #gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_b_grad, b)
