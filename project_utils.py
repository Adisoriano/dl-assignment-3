"""
project_utils.py

Utilities for Assignment 3 (MNIST, ch11 MLP, 2-hidden-layer adjusment + PyTorch implementation)

"""

# ==========================
# Imports
# ==========================
import numpy as np
from sklearn.metrics import roc_auc_score

# PyTorch imports for the comparison model
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==========================
# Helper functions
# ==========================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def int_to_onehot(y, num_labels):
    """Convert integer labels (N,) to one-hot matrix (N, num_labels)."""
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary


def minibatch_generator(X, y, minibatch_size, shuffle=True, drop_last=False):
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0], minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        if drop_last and len(batch_idx) < minibatch_size:
            continue
        yield X[batch_idx], y[batch_idx]

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)   # stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ==========================
# model: 1 hidden layer (original  - ch11)
# ==========================
class NeuralNetMLP:
    """
    1-hidden-layer MLP (ch11 as in instructions)
    Uses sigmoid activations and MSE loss (as in ch11).
    """

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        self.num_classes = num_classes

        rng = np.random.RandomState(random_seed)

        # hidden
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # output
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        # Hidden layer
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output layer
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        # onehot encoding
        y_onehot = int_to_onehot(y, self.num_classes)

        # dLoss/dOutAct
        # delta_out = softmax_mse_delta(a_out, y_onehot)

        # sigmoid like in chp11
        d_loss__d_a_out = 2.0 * (a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1.0 - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        # output gradients
        d_loss__dw_out = np.dot(delta_out.T, a_h)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # backprop into hidden
        d_loss__a_h = np.dot(delta_out, self.weight_out)
        d_a_h__d_z_h = a_h * (1.0 - a_h)
        delta_h = d_loss__a_h * d_a_h__d_z_h

        d_loss__d_w_h = np.dot(delta_h.T, x)
        d_loss__d_b_h = np.sum(delta_h, axis=0)

        return (d_loss__dw_out, d_loss__db_out,
                d_loss__d_w_h, d_loss__d_b_h)


# ==========================
# model: 2 hidden layers extention
# ==========================

class NeuralNetMLP2Hidden:
    """
    2-hidden-layer MLP ( extension of ch11)
    Uses sigmoid activations and MSE loss (as in ch11).

    - Adds weight_h2/bias_h2
    - Output layer connects from hidden2
    - forward returns (a_h1, a_h2, a_out)
    """

    def __init__(self, num_features, num_hidden1, num_hidden2, num_classes, random_seed=123):
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)

        # hidden 1
        self.weight_h1 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden1, num_features))
        self.bias_h1 = np.zeros(num_hidden1)

        # hidden 2 (NEW)
        self.weight_h2 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden2, num_hidden1))
        self.bias_h2 = np.zeros(num_hidden2)

        # output (connects from hidden2)
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden2))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        # Hidden layer 1
        z_h1 = np.dot(x, self.weight_h1.T) + self.bias_h1
        a_h1 = sigmoid(z_h1)

        # Hidden layer 2
        z_h2 = np.dot(a_h1, self.weight_h2.T) + self.bias_h2
        a_h2 = sigmoid(z_h2)

        # Output layer
        z_out = np.dot(a_h2, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)

        return a_h1, a_h2, a_out


# ==========================
# Metrics / evaluation helpers
# ==========================
def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas) ** 2)


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)


def macro_auc_ovr(y_true, probas, num_labels=10):
    """
    Macro AUC for multiclass via One-vs-Rest.
    """
    y_onehot = int_to_onehot(y_true, num_labels)
    return roc_auc_score(y_onehot, probas, average="macro", multi_class="ovr")


def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    #computes MSE and Accuracy
    mse, correct_pred, num_examples = 0., 0, 0

    minibatch_gen = minibatch_generator(X, y, minibatch_size, shuffle=False, drop_last=False)

    for i, (features, targets) in enumerate(minibatch_gen):
        out = nnet.forward(features)
        probas = out[-1]

        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas) ** 2)

        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss

    mse = mse / (i + 1)
    acc = correct_pred / num_examples
    return mse, acc

def softmax_mse_delta(a_out, y_onehot):
    """
    For softmax output
    """
    N = y_onehot.shape[0]
    dL_da = 2.0 * (a_out - y_onehot) / N
    # Jacobian-vector product for softmax:
    return a_out * (dL_da - np.sum(dL_da * a_out, axis=1, keepdims=True))


def predict_probas(model, X, minibatch_size=100):
    """
    Returns probas for all X using minibatches.
    Compatible with both forward() signatures by using out[-1].
    """
    probas_list = []
    dummy_y = np.zeros(X.shape[0], dtype=int)
    mb_gen = minibatch_generator(X, dummy_y, minibatch_size)

    for features, _ in mb_gen:
        out = model.forward(features)
        probas = out[-1]
        probas_list.append(probas)

    return np.vstack(probas_list)


def evaluate_model(model, X, y, minibatch_size=100, num_labels=10):
    """
    Returns (mse, acc, macro_auc) on the given set.
    """
    probas = predict_probas(model, X, minibatch_size=minibatch_size)
    y_pred = np.argmax(probas, axis=1)

    acc = accuracy(y, y_pred)
    auc = macro_auc_ovr(y, probas, num_labels=num_labels)
    mse = mse_loss(y, probas, num_labels=num_labels)
    return mse, acc, auc


# ================
# Training loops
# ================
def train(model, X_train, y_train, X_valid, y_valid, num_epochs,
          learning_rate=0.1, minibatch_size=100):
    """
    similar to original ch11 training loop for 1-hidden-layer model.
    Returns: epoch_loss, epoch_train_acc, epoch_valid_acc
    """
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:
            a_h, a_out = model.forward(X_train_mini)

            dW_out, db_out, dW_h, db_h = model.backward(X_train_mini, a_h, a_out, y_train_mini)

            model.weight_h   -= learning_rate * dW_h
            model.bias_h     -= learning_rate * db_h
            model.weight_out -= learning_rate * dW_out
            model.bias_out   -= learning_rate * db_out

        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train, minibatch_size=minibatch_size)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid, minibatch_size=minibatch_size)

        epoch_loss.append(train_mse)
        epoch_train_acc.append(train_acc * 100.0)
        epoch_valid_acc.append(valid_acc * 100.0)

        if (e + 1) % 10 == 0 or e == 0:
            print(f'[1H] Epoch: {e + 1:03d}/{num_epochs:03d} '
                  f'| Train MSE: {train_mse:.2f} '
                  f'| Train Acc: {train_acc:.2f}% '
                  f'| Valid Acc: {valid_acc:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc


def train_2hidden(model, X_train, y_train, X_valid, y_valid,
                  num_epochs, learning_rate=0.1, minibatch_size=100):
    """
    Training loop for NeuralNetMLP2Hidden (2 hidden layers).
    Returns: epoch_loss, epoch_train_acc, epoch_valid_acc
    """
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

        for X_mb, y_mb in minibatch_gen:
            a_h1, a_h2, a_out = model.forward(X_mb)
            y_onehot = int_to_onehot(y_mb, model.num_classes)

            # Output layer
            # delta_out = softmax_mse_delta(a_out, y_onehot)

            d_loss__d_a_out = 2.0 * (a_out - y_onehot) / y_mb.shape[0]
            d_a_out__d_z_out = a_out * (1.0 - a_out)
            delta_out = d_loss__d_a_out * d_a_out__d_z_out

            dW_out = np.dot(delta_out.T, a_h2)
            db_out = np.sum(delta_out, axis=0)

            # Hidden layer 2
            d_loss__a_h2 = np.dot(delta_out, model.weight_out)
            d_a_h2__d_z_h2 = a_h2 * (1.0 - a_h2)
            delta_h2 = d_loss__a_h2 * d_a_h2__d_z_h2

            dW_h2 = np.dot(delta_h2.T, a_h1)
            db_h2 = np.sum(delta_h2, axis=0)

            # Hidden layer 1
            d_loss__a_h1 = np.dot(delta_h2, model.weight_h2)
            d_a_h1__d_z_h1 = a_h1 * (1.0 - a_h1)
            delta_h1 = d_loss__a_h1 * d_a_h1__d_z_h1

            dW_h1 = np.dot(delta_h1.T, X_mb)
            db_h1 = np.sum(delta_h1, axis=0)

            # Parameter updates
            model.weight_h1  -= learning_rate * dW_h1
            model.bias_h1    -= learning_rate * db_h1
            model.weight_h2  -= learning_rate * dW_h2
            model.bias_h2    -= learning_rate * db_h2
            model.weight_out -= learning_rate * dW_out
            model.bias_out   -= learning_rate * db_out

        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train, minibatch_size=minibatch_size)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid, minibatch_size=minibatch_size)

        epoch_loss.append(train_mse)
        epoch_train_acc.append(train_acc * 100.0)
        epoch_valid_acc.append(valid_acc * 100.0)

        if (e + 1) % 10 == 0 or e == 0:
            print(f"[2H] Epoch: {e + 1:03d}/{num_epochs:03d} "
                  f"| Train MSE: {train_mse:.2f} "
                  f"| Train Acc: {train_acc * 100:.2f}% "
                  f"| Valid Acc: {valid_acc * 100:.2f}%")

    return epoch_loss, epoch_train_acc, epoch_valid_acc


# ==========================
# PyTorch framework model (for comparison)
# ==========================
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class TorchMLP(nn.Module):
    """
    PyTorch fully-connected ANN:
    784 -> 500 -> 500 -> 10
    Hidden activations: sigmoid
    Output activation: softmax
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        logits = self.fc3(x)
        probas = torch.softmax(logits, dim=1)  # softmax on output
        return probas


def train_torch_mse_softmax(
    X_train, y_train, X_test, y_test,
    lr=0.1, epochs=50, batch_size=100
):
    """
    Trains TorchMLP with:
      - sigmoid hidden layers + softmax output
      - MSE loss on one-hot labels
      - SGD lr=0.1, batch=100, epochs=50

    Returns:
        model,
        final_test_acc (float in 0..1),
        final_test_auc (float),
        train_acc_hist (list of floats),
        test_acc_hist (list of floats),
        prob_test_np (np.ndarray, shape [N,10])
    """
    device = "cpu"
    model = TorchMLP().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # tensors
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    yte = torch.tensor(y_test, dtype=torch.long, device=device)

    # one-hot labels for MSE
    ytr_oh = torch.zeros((ytr.shape[0], 10), dtype=torch.float32, device=device)
    ytr_oh[torch.arange(ytr.shape[0], device=device), ytr] = 1.0

    loader = DataLoader(TensorDataset(Xtr, ytr_oh), batch_size=batch_size, shuffle=True)

    train_acc_hist, test_acc_hist = [], []

    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)           # softmax probabilities
            loss = loss_fn(out, yb)   # MSE vs one-hot
            loss.backward()
            opt.step()

        # accuracy tracking per epoch
        model.eval()
        with torch.no_grad():
            train_out = model(Xtr)
            test_out = model(Xte)

            train_pred = torch.argmax(train_out, dim=1)
            test_pred = torch.argmax(test_out, dim=1)

            train_acc = (train_pred == ytr).float().mean().item() * 100.0
            test_acc = (test_pred == yte).float().mean().item() * 100.0

        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[PyTorch-MSE] Epoch {ep+1:03d}/{epochs:03d} | "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    prob_test_np = np.array(test_out.detach().cpu().tolist())

    final_test_acc = test_acc_hist[-1] / 100.0
    final_test_auc = macro_auc_ovr(y_test, prob_test_np)

    return model, final_test_acc, final_test_auc, train_acc_hist, test_acc_hist, prob_test_np
