import loglinear as ll
from utils import *

STUDENT={'name': 'Vladimir Balagula',
         'ID': '323792770'}


def feats_to_vec(features):
    """
    converting features to vectors
    """
    vec = [0] * len(F2I)
    for feature in features:
        if feature in F2I:
            vec[F2I[feature]] += 1
    return vec


def accuracy_on_dataset(dataset, params):
    """
    Compute the accuracy (a scalar) of the current parameters
    on the dataset.
    accuracy is (correct_predictions / all_predictions)
    """
    good = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        good += 1 if ll.predict(x, params) == L2I[label] else 0
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
            x = feats_to_vec(features)  # convert features to a vector.
            y = L2I[label]  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            # update weights and bias
            params = [params[i] - learning_rate * grads[i] for i in range(len(grads))]
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def create_model_BI_grams():
    # set parameters for model
    num_iterations = 75
    learning_rate = 0.0007
    in_dim = len(F2I)
    out_dim = len(L2I)
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(TRAIN, DEV, num_iterations, learning_rate, params)
    with open("test.pred", "w") as file:
        for _, feature in TEST:
            # convert feature to vectors
            x = feats_to_vec(feature)
            # getting predicted value index and find it value
            y_tag = ll.predict(x, trained_params)
            value = list(L2I.keys())[list(L2I.values()).index(y_tag)]
            file.write(value + "\n")


if __name__ == '__main__':
    create_model_BI_grams()





