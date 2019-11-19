import time

from codex import mlp1
from codex.utils import *
from codex.train_loglin import feats_to_vec
from codex.xor_data import data

STUDENT={'name': 'Vladimir Balagula',
         'ID': '323792770'}

def accuracy_on_dataset(dataset, params):
    """
    Compute the accuracy (a scalar) of the current parameters
    on the dataset.
    accuracy is (correct_predictions / all_predictions)
    """
    good = total = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        good += 1 if mlp1.predict(x, params) == L2I[label] else 0
        total += 1
    return good / total


def accuracy_on_dataset_XOR(dataset, params):
    """
    Compute the accuracy (a scalar) of the current parameters
    on the dataset.
    accuracy is (correct_predictions / all_predictions)
    """
    good = total = 0.0
    for label, features in dataset:
        good += 1 if mlp1.predict(features, params) == label else 0
        total += 1
    return good / total


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = L2I[label]                  # convert the label to number if needed.
            loss, grads = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss
            params = [params[i] - learning_rate*grads[i] for i in range(len(grads))]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

def train_classifier_XOR(train_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier for XOR
    Because skeleton was build for language and we don't want to change it

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = features
            y = label
            loss, grads = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss
            params = [params[i] - learning_rate*grads[i] for i in range(len(grads))]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset_XOR(train_data, params)
        print(I, train_loss, train_accuracy)
        if train_accuracy == 1.0:
            print("Early stop because the model got maximum accuracy")
            break
    return params

def create_model_BI_grams():
    print("------------------------- Start to train with BI grams -------------------------")
    num_iterations = 70
    learning_rate = 0.001

    in_dim = len(F2I)
    hid_dim = 200
    out_dim = len(L2I)
    params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(TRAIN, DEV, num_iterations, learning_rate, params)

def create_model_UNI_grams():
    print("------------------------- Start to train with UNI grams -------------------------")
    num_iterations = 70
    learning_rate = 0.001

    in_dim = len(F2I_UNI)
    hid_dim = 200
    out_dim = len(L2I)
    params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(TRAIN_UNI, DEV_UNI, num_iterations, learning_rate, params)


def create_model_4Xor():
    print("------------------------- Start to train XOR m0del -------------------------")
    in_dim = 2
    hid_dim = 3
    out_dim = 2
    params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier_XOR(data,
                                          num_iterations=5000,
                                          learning_rate=0.05,
                                          params=params)



if __name__ == '__main__':
    create_model_BI_grams()
    create_model_UNI_grams()
    create_model_4Xor()



