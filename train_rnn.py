import datetime
import logging
import os
import pathlib
import torch
from torch import nn
from logging.handlers import QueueHandler

SEQ_LEN = 10
NUM_INPUTS = 2
NUM_HIDDEN = 50
NUM_OUTPUTS = 1
BATCH_SIZE = 20
EPOCHS = 50

MODEL_WEIGHTS_DIR = 'weights'

torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def init_logger():
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    relative_path = f"logs/training_{now_time}.log"
    log_full_path = os.path.join(os.getcwd(), relative_path)

    logger.info(f'Creating log in {log_full_path}')

    console_handler = logging.StreamHandler()
    file_handler = logging.handlers.RotatingFileHandler(log_full_path)

    console_handler.setLevel(logging.INFO)
    logging.root.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    default_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(default_formatter)
    file_handler.setFormatter(default_formatter)

    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)


def read_data_adding_problem(csv_filename):
    lines = pathlib.Path(csv_filename).read_text().splitlines()
    values, markers, adding_results = [], [], []
    cnt = 0
    for line in lines:
        cnt += 1
        if cnt % 3 == 1:
            curr_values = [float(s) for s in line.split(',')]
            values.append(curr_values)
        elif cnt % 3 == 2:
            curr_markers = [float(s) for s in line.split(',')]
            markers.append(curr_markers)
        else:
            curr_adding_result = float(line.split(',')[0])
            adding_results.append(curr_adding_result)
    return values, markers, adding_results


def read_data_adding_problem_torch(csv_filename):
    values, markers, adding_results = read_data_adding_problem(csv_filename)
    assert len(values) == len(markers) == len(adding_results)
    num_data = len(values)
    seq_len = len(values[0])
    X = torch.Tensor(num_data, seq_len, 2)
    T = torch.Tensor(num_data, 1)
    for k, (curr_values, curr_markers, curr_adding_result) in \
            enumerate(zip(values, markers, adding_results)):
        T[k] = curr_adding_result
        for n, (v, m) in enumerate(zip(curr_values, curr_markers)):
            X[k, n, 0] = v
            X[k, n, 1] = m
    return X, T


def get_batches(X, T, batch_size):
    num_data, max_seq_len, _ = X.shape
    for idx1 in range(0, num_data, batch_size):
        idx2 = min(idx1 + batch_size, num_data)
        yield X[idx1:idx2, :, :], T[idx1:idx2, :]


class SimpleRNNFromBox(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(SimpleRNNFromBox, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.rnn = nn.RNN(input_size=num_inputs, hidden_size=num_hidden,
                          num_layers=1, batch_first=True, bidirectional=False)
        self.out_layer = nn.Linear(in_features=num_hidden,
                                   out_features=num_outputs)

    def forward(self, X):
        num_data, max_seq_len, _ = X.shape
        h0 = torch.zeros(1, num_data, self.num_hidden)
        output, hn = self.rnn(X, h0) # output.shape: num_data x seq_len x num_hidden
        last_output = output[:, -1, :] # num_data x num_hidden
        Y = self.out_layer(last_output) # num_data x num_outputs
        return Y


class SimpleLSTMFromBox(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(SimpleLSTMFromBox, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=num_hidden,
                          num_layers=1, batch_first=True, bidirectional=False)
        self.out_layer = nn.Linear(in_features=num_hidden,
                                   out_features=num_outputs)

    def forward(self, X):
        num_data, max_seq_len, _ = X.shape
        h0 = torch.zeros(1, num_data, self.num_hidden)
        c0 = torch.zeros(1, num_data, self.num_hidden)
        output, (hn, cn) = self.lstm(X, (h0, c0))  # output.shape: num_data x seq_len x num_hidden
        last_output = output[:, -1, :]  # num_data x num_hidden
        Y = self.out_layer(last_output)  # num_data x num_outputs
        return Y


class VectorAlarmworkRNN(nn.Module):
    # vector form

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(VectorAlarmworkRNN, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.W_in1, self.b_in1, self.W_rec1, self.W_in2, self.b_in2, self.W_rec2, self.W_out, self.b_out = \
            self._init_params()

    def forward(self, X):
        num_data, max_seq_len, _ = X.shape
        z_0 = torch.zeros(self.num_hidden)
        z_in1 = torch.zeros((num_data, max_seq_len, self.num_hidden))
        a_in1 = torch.zeros((num_data, max_seq_len, self.num_hidden))
        z_in2 = torch.zeros((num_data, max_seq_len, self.num_hidden))
        a_in2 = torch.zeros((num_data, max_seq_len, self.num_hidden))
        z_in12 = torch.zeros((num_data, max_seq_len, self.num_hidden))
        a_out = torch.zeros((num_data, self.num_outputs))
        Y = torch.zeros((num_data, self.num_outputs))
        for n in range(num_data):
            for l in range(max_seq_len):
                z_in12[n, l] = z_in1[n, l - 1].clone() if l > 1 else z_0 + z_in2[n, l - 1].clone() if l > 1 else z_0
                a_in1[n, l] = torch.matmul(self.W_in1, X[n, l, :]) + torch.matmul(
                    self.W_rec1, z_in12[n, l].clone()) + self.b_in1
                z_in1[n, l] = torch.tanh(a_in1[n, l].clone())
                a_in2[n, l] = torch.matmul(self.W_in2, X[n, l, :]) + torch.matmul(
                    self.W_rec2, z_in2[n, l - 1 if l % 2 == 0 else 2].clone() if l > 1 else z_0) + self.b_in2
                z_in2[n, l] = torch.tanh(a_in2[n, l].clone())
            a_out[n, :] = torch.matmul(self.W_out, z_in1[n, -1, :].clone()) + self.b_out
            Y[n, :] = torch.tanh(a_out[n, :].clone())
        return Y

    def _init_params(self):
        W_in1 = nn.Parameter(torch.zeros((self.num_hidden, self.num_inputs)))
        b_in1 = nn.Parameter(torch.zeros(self.num_hidden))
        W_rec1 = nn.Parameter(torch.zeros((self.num_hidden, self.num_hidden)))
        W_in2 = nn.Parameter(torch.zeros((self.num_hidden, self.num_inputs)))
        b_in2 = nn.Parameter(torch.zeros(self.num_hidden))
        W_rec2 = nn.Parameter(torch.zeros((self.num_hidden, self.num_hidden)))
        W_out = nn.Parameter(torch.zeros((self.num_outputs, self.num_hidden)))
        b_out = nn.Parameter(torch.zeros(self.num_outputs))
        return W_in1, b_in1, W_rec1, W_in2, b_in2, W_rec2, W_out, b_out


class ScalarAlarmworkRNN(VectorAlarmworkRNN):
    # scalar form

    def forward(self, X):
        pass


def adding_problem_evaluate(outputs, gt_outputs):
    assert outputs.shape == gt_outputs.shape
    num_data = outputs.shape[0]
    num_correct = 0
    for i in range(num_data):
        y = outputs[i].item()
        t = gt_outputs[i].item()
        if abs(y - t) < 0.1:
            num_correct += 1
    acc = num_correct*100 / len(outputs)
    return acc


def evaluate_model(model, X_test, T_test):
    test_acc = adding_problem_evaluate(model(X_test), T_test)
    logger.info(f'TEST accuracy for model {type(model).__name__} is {test_acc}')
    return test_acc


def train_model(model, X_train, X_dev, T_train, T_dev, T):
    model_name = _get_model_name(model)

    loaded_model = load_model_weights(model, T, MODEL_WEIGHTS_DIR, EPOCHS)

    if loaded_model:
        logger.info(f'Model exists, loaded weight for {model_name}')
        return loaded_model

    logger.info(f'Training model: {model_name}')
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for e in range(EPOCHS):
        model.eval()
        dev_acc = adding_problem_evaluate(model(X_dev), T_dev)
        logger.info(f'T = {T}, epoch = {e}, DEV accuracy = {dev_acc}%%')
        if dev_acc > 99.5:
            break
        model.train()
        for X_batch, T_batch in get_batches(X_train, T_train, batch_size=BATCH_SIZE):
            Y_batch = model(X_batch)
            loss = loss_fn(Y_batch, T_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    save_model_weights(model, T, MODEL_WEIGHTS_DIR, EPOCHS)

    return model


def _get_model_name(model):
    return type(model).__name__


def _get_model_path(output_dir, model_name, T, epochs):
    return f'{output_dir}/{model_name}_E={epochs}_T={T}.pt'


def save_model_weights(model, T, weights_dir, epochs):
    os.makedirs(weights_dir, exist_ok=True)
    model_path = _get_model_path(weights_dir, type(model).__name__, T, epochs)
    torch.save(model.state_dict(), model_path)
    logger.info(f'Saved model weights to {model_path}')


def load_model_weights(model_class, T, weights_dir, epochs):
    model_name = _get_model_name(model_class)

    model_path = _get_model_path(weights_dir, model_name, T, epochs)

    if not os.path.exists(model_path):
        return

    model_class.load_state_dict(torch.load(model_path))
    model_class.eval()

    logger.info(f'Loading model weights for {model_name} from {model_path}')
    return model_class


def main():
    print('Welcome to the Matrix!')
    init_logger()

    results = dict()
    for T in [10, 50, 70, 100]:

        logger.info(f'Staring run for sequence length T = {T}...')

        X_train, T_train = read_data_adding_problem_torch('adding_problem_data/adding_problem_T=%03d_train.csv' % T)
        X_dev, T_dev = read_data_adding_problem_torch('adding_problem_data/adding_problem_T=%03d_dev.csv' % T)
        X_test, T_test = read_data_adding_problem_torch('adding_problem_data/adding_problem_T=%03d_test.csv' % T)

        rnn_model = SimpleRNNFromBox(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
        rnn_model = train_model(rnn_model, X_train, X_dev, T_train, T_dev, T)
        rnn_test_acc = evaluate_model(rnn_model, X_test, T_test)

        lstm_model = SimpleLSTMFromBox(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
        lstm_model = train_model(lstm_model, X_train, X_dev, T_train, T_dev, T)
        lstm_model_acc = evaluate_model(lstm_model, X_test, T_test)

        alarmwork_model = VectorAlarmworkRNN(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
        alarmwork_model = train_model(alarmwork_model, X_train, X_dev, T_train, T_dev, T)
        alarmwork_test_acc = evaluate_model(alarmwork_model, X_test, T_test)

        results[T] = {
            'SimpleRNNFromBox': rnn_test_acc,
            'SimpleLSTMFromBox': lstm_model_acc,
            'VectorAlarmworkRNN': alarmwork_test_acc
        }

        logger.info(f'Finished run for sequence length T = {T}: {results[T]}.')

    logger.info(results)


if __name__ == '__main__':
    main()
