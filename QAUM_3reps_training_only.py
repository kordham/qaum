
#Importing Libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing

#PennyLane for QNN
import pennylane as qml
from pennylane.optimize import AdamOptimizer

import time
num_epochs = 150
loss_curve_x = np.array(range(num_epochs))


def quantum_model_train(train_X, train_Y, depth=0):
    from pennylane import numpy as np

    train_X = np.array(train_X, requires_grad=False)
    train_Y = np.array(train_Y, requires_grad=False)

    train_data = list(zip(train_X, train_Y))

    dev = qml.device("default.qubit.autograd", wires=1)

    def variational_circ(i, w):
        qml.RZ(w[i][0], wires=0)
        qml.RX(w[i][1], wires=0)
        qml.RY(w[i][2], wires=0)

    def quantum_neural_network(x, w, depth=depth):
        qml.Hadamard(wires=0)

        variational_circ(0, w)
        for i in range(0, depth):
            for j in range(8):
                qml.RZ(x[j], wires=0)
                variational_circ(j + 8 * i, w)

    @qml.qnode(dev, diff_method='backprop')
    def get_output(x, w):
        quantum_neural_network(x, w)
        return qml.expval(qml.PauliZ(wires=0))

    @qml.qnode(dev)
    def get_state(x, w):
        quantum_neural_network(x, w)
        return qml.state()

    def get_parity_prediction(x, w):

        np_measurements = (get_output(x, w) + 1.) / 2.

        return np.array([1. - np_measurements, np_measurements])

    def average_loss(w, data):
        cost_value = 0
        for i, (x, y) in enumerate(data):
            cost_value += single_loss(w, x, y)

        return cost_value / len(data)

    def single_loss(w, x, y):
        prediction = get_parity_prediction(x, w)
        return rel_ent(prediction, y)

    def rel_ent(pred, y):
        return -1. * np.log(pred)[int(y)]

    def categorise(x, w):
        out = get_parity_prediction(x, w)
        return np.argmax(out)

    def accuracy(data, w):
        correct = 0
        for ii, (x, y) in enumerate(data):
            cat = categorise(x, w)
            if (int(cat) == int(y)):   correct += 1
        return correct / len(data) * 100

    # initialise weights
    w = np.array(np.split(np.random.uniform(size=(3(8*depth+1),), low=-1, high=1), 8*depth+1),
                 requires_grad=True) * 2 * np.pi
    learning_rate = 0.1
    train_losses = np.zeros((num_epochs))
    # Optimiser
    optimiser = AdamOptimizer(learning_rate)
    for i in range(num_epochs):
        start = time.time()
        w, train_loss_value = optimiser.step_and_cost(lambda v: average_loss(v, train_data), w)
        end = time.time()
        train_losses[i] = train_loss_value
        print("Epoch = ", i, " Training Loss = ", train_loss_value, "  Time taken = ", end - start)

    return train_losses


def fetch_data_random_seed(n_samples, seed):
    dataset = pd.read_csv('pulsar.csv')

    data0 = dataset[dataset[dataset.columns[8]] == 0]
    data0 = data0.sample(n=int(n_samples / 2), random_state=seed)
    X0 = data0[data0.columns[0:8]].values
    Y0 = data0[data0.columns[8]].values

    data1 = dataset[dataset[dataset.columns[8]] == 1]
    data1 = data1.sample(n=int(n_samples / 2), random_state=seed)
    X1 = data1[data1.columns[0:8]].values
    Y1 = data1[data1.columns[8]].values
    X = np.append(X0, X1, axis=0)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi))
    X = min_max_scaler.fit_transform(X)

    return X, np.append(Y0, Y1, axis=0)


num_epochs = 150
n_iteration = 5
depth = 5

losses_qaoa = np.zeros((depth,n_iteration,num_epochs))
for j in range(depth):
    for i in range(n_iteration):
        X,Y = fetch_data_random_seed(n_samples=100,seed=i)
        print("BORN MACHINE")
        losses_qaoa[j][i] = quantum_model_train(X,Y,depth=j+1)

np.save("losses_qaoa.npy",losses_qaoa)
