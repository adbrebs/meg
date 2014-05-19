__author__ = 'adeb'

import sys
import ConfigParser

from databasemeg import DataBaseMEG
import brain.nn as nn
from brain.trainer import Trainer


def load_config():
    cf = ConfigParser.ConfigParser()
    if len(sys.argv) == 1:
        cf.read('meg_training.ini')
    else:
        cf.read(str(sys.argv[1]))
    return cf

if __name__ == '__main__':

    ### Load the config file
    training_cf = load_config()

    db = DataBaseMEG()
    db.load_from("train_meg.h5")

    ### Create the network
    # MLP kind network
    net = nn.Network1()
    net.init(db.n_in_features, db.n_out_features)
    # CNN network
    # net = nn.Network2(db.patch_width, db.n_out_features)

    ### Train the network
    t = Trainer(training_cf, net, db, False)
    t.train()

    ### Save the network
    net.save_parameters(training_cf.get("general", "net"))