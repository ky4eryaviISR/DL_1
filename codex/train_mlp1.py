import mlp1
from utils import *
from train_loglin import feats_to_vec

STUDENT={'name': 'Vladimir Balagula',
         'ID': '323792770'}


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


def create_model_BI_grams():
    print("------------------------- Start to train with BI grams -------------------------")
    num_iterations = 70
    learning_rate = 0.001

    in_dim = len(F2I)
    hid_dim = 200
    out_dim = len(L2I)
    params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(TRAIN, DEV, num_iterations, learning_rate, params)


if __name__ == '__main__':
    create_model_BI_grams()



