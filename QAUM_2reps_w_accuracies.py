#Importing Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import preprocessing


from sklearn.model_selection import train_test_split


#PennyLane for QNN
import pennylane as qml
from pennylane.optimize import AdamOptimizer

import time


def fetch_data_random_seed_val(n_samples, seed):
    dataset = pd.read_csv('pulsar.csv')

    data0 = dataset[dataset[dataset.columns[8]] == 0]
    data0 = data0.sample(n=n_samples, random_state=seed)
    X0 = data0[data0.columns[0:8]].values
    Y0 = data0[data0.columns[8]].values

    data1 = dataset[dataset[dataset.columns[8]] == 1]
    data1 = data1.sample(n=n_samples, random_state=seed)
    X1 = data1[data1.columns[0:8]].values
    Y1 = data1[data1.columns[8]].values

    X = np.append(X0, X1, axis=0)
    Y = np.append(Y0, Y1, axis=0)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi))
    X = min_max_scaler.fit_transform(X)

    # Separate the test and training datasets
    train_X, validation_X, train_Y, validation_Y = train_test_split(X, Y, test_size=0.5, random_state=seed)

    return train_X, validation_X, train_Y, validation_Y


def quantum_model_train(train_X, train_Y, validation_X=None, validation_Y=None, depth=1):
    from pennylane import numpy as np

    train_X = np.array(train_X, requires_grad=False)
    train_Y = np.array(train_Y, requires_grad=False)

    validation_X = np.array(validation_X, requires_grad=False)
    validation_Y = np.array(validation_Y, requires_grad=False)
    validation_data = list(zip(validation_X, validation_Y))

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

    def R2P(x):
        return np.abs(x), np.angle(x)

    def plot_data(w, data):
        import numpy as np
        Xs = np.array([])
        Ys =np.array([])
        Zs = np.array([])
        labels = np.array([])
        for i, (x, y) in enumerate(data):
            state = get_state(x, w)
            polar_state = R2P(state)
            theta = 2 * np.arctan2(polar_state[1][0], polar_state[0][0])
            phase = polar_state[1][1] - polar_state[0][1]
            Xs = np.append(np.sin(theta) * np.cos(phase), Xs)
            Ys = np.append(np.sin(theta) * np.sin(phase), Ys)
            Zs = np.append(np.cos(theta), Zs)
            labels = np.append(y, labels)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # draw sphere
        u, v = np.mgrid[0:2 * np.pi:40j, 0:2 * np.pi:40j]
        x1 = np.cos(u) * np.sin(v)
        y1 = np.sin(u) * np.sin(v)
        z1 = np.cos(v)
        ax.plot_wireframe(x1, y1, z1, color="0.5", linewidth=0.1)
        color = ['red', 'navy']
        for i in range(2):
            test = np.where(labels == i)
            ax.scatter(Xs[test], Ys[test], Zs[test], marker='o', s=3, color=color[i], alpha=0.9)
        plt.show()
        from pennylane import numpy as np

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
    w = np.array(np.split(np.random.uniform(size=(3 * (8 * depth + 1),), low=-1, high=1), 8 * depth + 1),
                 requires_grad=True) * 2 * np.pi
    learning_rate = 0.1

    # Optimiser
    optimiser = AdamOptimizer(learning_rate)
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        acc = np.array([])
        start = time.time()
        w, train_loss_value = optimiser.step_and_cost(lambda v: average_loss(v, train_data), w)
        end = time.time()
        w.requires_grad = False
        train_acc = accuracy(train_data, w)
        validation_loss_value = average_loss(w, validation_data)
        validation_acc = accuracy(validation_data, w)
        w.requires_grad = True

        train_accs.append(train_acc)
        train_losses.append(train_loss_value)
        val_accs.append(validation_acc)
        val_losses.append(validation_loss_value)

        print("Epoch = ", i, " Training Loss = ", train_loss_value, " Validation Loss = ", validation_loss_value,
              " Train Acc = ", train_acc, "% Val Acc = ",
              validation_acc, "%", "  Time taken = ", end - start)

    return train_accs,val_accs,train_losses,val_losses



num_epochs = 150
n_iteration = 5

losses = []
for i in range(n_iteration):
        #print("CLASSICAL NN")
        #classical_train(train_X,train_Y,all_validation_X,all_validation_Y)
        train_X, validation_X, train_Y, validation_Y = fetch_data_random_seed_val(n_samples=100, seed=i)
        print("BORN MACHINE")
        loss = quantum_model_train(train_X, train_Y, validation_X, validation_Y,depth=2)
        losses.append(loss)
print(np.array(losses,dtype=float).shape)