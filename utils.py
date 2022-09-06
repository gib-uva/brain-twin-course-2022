import numpy as np
from tabulate import tabulate
import pickle


def print_acc_per_seq(acc_per_seq):
    table_cmd_acc_per_seq = ['Command decoding acc']
    cmd_acc_per_seq = np.char.mod('%.2f', acc_per_seq*100).astype(str).tolist()
    table_cmd_acc_per_seq += cmd_acc_per_seq
    headers = [''] + list(range(1, 16))
    print('\nAccuracy per number of sequences of stimulation:\n')
    print(tabulate([table_cmd_acc_per_seq], headers=headers))


def save_model(filename, clf):
    with open('models/%s.mdl' % filename, 'wb') as f:
        pickle.dump(clf, f)


def load_model(filename):
    with open('models/%s.mdl' % filename, 'rb') as f:
        return pickle.load(f)
