import mlp1
from utils import *
from train_loglin import feats_to_vec,feats_to_vec_uni,ParseOption
from xor_data import data
import enum

STUDENT={'name': 'Vladimir Balagula',
         'ID': '323792770'}


def accuracy_on_dataset_uni(dataset, params):
    """
    Compute the accuracy (a scalar) of the current parameters
    on the dataset.
    accuracy is (correct_predictions / all_predictions)
    """
    good = 0
    for label, features in dataset:
        x = feats_to_vec_uni(features)
        good += 1 if mlp1.predict(x, params) == L2I[label] else 0
    return good / len(dataset)



def accuracy_on_dataset(dataset, params):
    """
    Compute the accuracy (a scalar) of the current parameters
    on the dataset.
    accuracy is (correct_predictions / all_predictions)
    """
    good = 0
    for label, features in dataset:
        x = feats_to_vec(features)
        good += 1 if mlp1.predict(x, params) == L2I[label] else 0
    return good / len(dataset)


def accuracy_on_dataset_XOR(dataset, params):
    """
    Compute the accuracy (a scalar) of the current parameters
    on the dataset.
    accuracy is (correct_predictions / all_predictions)
    """
    good = 0
    for label, features in dataset:
        good += 1 if mlp1.predict(features, params) == label else 0
    return good / len(dataset)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params, parse=ParseOption.BI):
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
            if parse == ParseOption.BI:
                x = feats_to_vec(features) # convert features to a vector.
                y = L2I[label]                  # convert the label to number if needed.
            elif parse == ParseOption.XOR:
                x = features
                y = label
            else:
                x = feats_to_vec_uni(features)
                y = L2I[label]
            loss, grads = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss
            params = [params[i] - learning_rate*grads[i] for i in range(len(grads))]
        train_loss = cum_loss / len(train_data)
        if parse in [ParseOption.BI, ParseOption.UNI]:
            accuracy_func = accuracy_on_dataset if parse == ParseOption.BI else accuracy_on_dataset_uni
            train_accuracy = accuracy_func(train_data, params)
            dev_accuracy = accuracy_func(dev_data, params)
            print(I, train_loss, train_accuracy, dev_accuracy)
        else:
            train_accuracy = accuracy_on_dataset_XOR(train_data, params)
            print(I, train_loss, train_accuracy)
            if train_accuracy == 1.0:
                print("Early stop the model has learn the XOR function")
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
    learning_rate = 0.005

    in_dim = len(F2I_UNI)
    hid_dim = 300
    out_dim = len(L2I)
    params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(TRAIN_UNI, DEV_UNI, num_iterations, learning_rate, params,ParseOption.UNI)


def create_model_4Xor():
    print("------------------------- Start to train XOR m0del -------------------------")
    in_dim = 2
    hid_dim = 5
    out_dim = 2
    params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(data,data,
                                      num_iterations=500,
                                      learning_rate=0.05,
                                      params=params,
                                      parse=ParseOption.XOR)


if __name__ == '__main__':
    create_model_BI_grams()
    create_model_UNI_grams()
    create_model_4Xor()



