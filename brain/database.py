__author__ = 'adeb'

import numpy as np

import theano

from dataset import DatasetBrainParcellation


class DataBase():
    """
    Class responsible for splitting, storing and loading on the GPU the training and testing datasets.
    """
    def __init__(self):

        self.n_in_features = None
        self.n_out_features = None
        self.n_data = None

        self.test_x = None
        self.test_y = None
        self.valid_x = None
        self.valid_y = None
        self.train_x = None
        self.train_y = None

        self.n_train = None
        self.n_valid = None
        self.n_test = None

    def load_from_config(self, config):
        raise NotImplementedError

    def share_data(self, test_x, test_y, valid_x, valid_y, train_x, train_y):
        """
        Store the data in shared variables
        """
        def share_data_object(data, borrow=True):
            shared_data = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=borrow)
            return shared_data

        self.test_x = share_data_object(test_x)
        self.test_y = share_data_object(test_y)
        self.valid_x = share_data_object(valid_x)
        self.valid_y = share_data_object(valid_y)
        self.train_x = share_data_object(train_x)
        self.train_y = share_data_object(train_y)

        self.n_train, self.n_in_features = self.train_x.get_value(borrow=True).shape
        self.n_valid = self.valid_x.get_value(borrow=True).shape[0]
        self.n_test = self.test_x.get_value(borrow=True).shape[0]

        self.n_out_features = self.train_y.get_value(borrow=True).shape[1]
        self.n_data = self.n_train + self.n_valid + self.n_test


class DataBaseBrainParcellation(DataBase):
    def __init__(self):
        DataBase.__init__(self)

        self.patch_width = None
        self.n_patch_per_voxel_testing = None

    def load_from_config(self, config):

        training_data_file = config.get('dataset', 'training_data')
        testing_data_file = config.get('dataset', 'testing_data')

        print '... loading data ' + training_data_file + ' and ' + testing_data_file

        # Load training data
        training_data = DatasetBrainParcellation()
        training_data.read(training_data_file)
        self.n_out_features = training_data.n_out_features
        self.patch_width = training_data.patch_width
        n_data = training_data.n_data

        # Create a validation set
        validatioin_split = int(0.9 * n_data)
        train_x = training_data.inputs[0:validatioin_split-1, :]
        train_y = training_data.outputs[0:validatioin_split-1, :]
        valid_x = training_data.inputs[validatioin_split:n_data, :]
        valid_y = training_data.outputs[validatioin_split:n_data, :]

        # Load testing data
        testing_data = DatasetBrainParcellation()
        testing_data.read(testing_data_file)
        self.n_patch_per_voxel_testing = testing_data.n_patch_per_voxel
        test_x = testing_data.inputs
        test_y = testing_data.outputs

        self.share_data(test_x, test_y, valid_x, valid_y, train_x, train_y)