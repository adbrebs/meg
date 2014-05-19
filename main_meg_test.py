__author__ = 'adeb'

import numpy as np

import brain.nn as nn
import brain.trainer as trainer

from datasetmeg import DatasetMEGUnabelled, DatasetMEGLabelled

if __name__ == '__main__':

    ### Load the network
    net = nn.Network1()
    net.load_parameters("net_meg.net")

    ### Load testing dataset
    ds_test = DatasetMEGUnabelled()
    ds_test.read("test_meg.h5")

    ds_train = DatasetMEGLabelled()
    ds_train.read("train_meg.h5")

    pred_train = net.predict(ds_train.inputs)
    pred_train_2 = np.argmax(pred_train, axis=1)
    err = trainer.Trainer.error_rate(pred_train, ds_train.outputs)
    print err

    pred_test = net.predict(ds_test.inputs)
    pred_test_2 = np.argmax(pred_test, axis=1)

    filename_submission = 'submission.csv'
    f = open(filename_submission,'w')
    print >> f, 'Id,Prediction'
    for i, ids_i in enumerate(ds_test.ids):
        print >> f, str(int(ids_i)) + ',' + str(int(pred_test_2[i]))
    f.close()