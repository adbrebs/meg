
__author__ = 'adeb'

import os
import numpy as np
from scipy.io import loadmat

from brain.dataset import Dataset


class DatasetMEG(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.n_time_series = None
        self.len_time_series = None

    @staticmethod
    def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
        """Creation of the feature space:
        - restricting the time window of MEG data to [tmin, tmax]sec.
        - Concatenating the 306 timeseries of each trial in one long
          vector.
        - Normalizing each feature independently (z-scoring).
        """
        print "Applying the desired time window."
        beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
        end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
        XX = XX[:, :, beginning:end].copy()

        print "2D Reshaping: concatenating all 306 timeseries."
        XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

        print "Features Normalization."
        XX -= XX.mean(0)
        XX = np.nan_to_num(XX / XX.std(0))

        return XX

    def write_virtual(self, h5file):
        h5file.attrs['n_time_series'] = self.n_time_series
        h5file.attrs['len_time_series'] = self.len_time_series

    def read_virtual(self, h5file):
        self.n_time_series = int(h5file.attrs["n_time_series"])
        self.len_time_series = int(h5file.attrs["len_time_series"])


class DatasetMEGLabelled(DatasetMEG):
    def __init__(self):
        DatasetMEG.__init__(self)

    def generate(self, subjects_train):

        # We throw away all the MEG data outside the first 0.5sec from when
        # the visual stimulus start:
        tmin = 0.0
        tmax = 0.500
        print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

        inputs = []
        outputs = []

        print
        print "Creating the trainset."
        for subject in subjects_train:
            print os.getcwd()
            filename = './data/meg/train_subject%02d.mat' % subject
            print "Loading", filename
            data = loadmat(filename, squeeze_me=True)
            XX = data['X']
            yy = data['y']
            sfreq = data['sfreq']
            tmin_original = data['tmin']
            print "Dataset summary:"
            print "XX:", XX.shape
            print "sfreq:", sfreq

            XX = self.create_features(XX, tmin, tmax, sfreq)

            inputs.append(XX)
            outputs.append(yy)

        self.inputs = np.vstack(inputs)
        self.n_data, self.n_in_features = self.inputs.shape

        outputs = np.concatenate(outputs)
        self.outputs = np.zeros((self.n_data, 2), dtype=int)
        self.outputs[np.arange(self.n_data), outputs] = 1
        self.n_out_features = self.outputs.shape[1]

        print "Trainset:", self.inputs.shape

        self.permute_data()
        self.is_perm = True

        self.n_time_series = 306
        self.len_time_series = 375


class DatasetMEGUnabelled(DatasetMEG):
    def __init__(self):
        DatasetMEG.__init__(self)
        self.ids = None
        self.outputs = 0
        self.n_out_features = 0

    def generate(self, subjects_train):

        # We throw away all the MEG data outside the first 0.5sec from when
        # the visual stimulus start:
        tmin = 0.0
        tmax = 0.500
        print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

        inputs = []
        ids = []

        print
        print "Creating the trainset."
        for subject in subjects_train:
            print os.getcwd()
            filename = './data/meg/test_subject%02d.mat' % subject
            print "Loading", filename
            data = loadmat(filename, squeeze_me=True)
            XX = data['X']
            iids = data['Id']
            sfreq = data['sfreq']
            tmin_original = data['tmin']
            print "Dataset summary:"
            print "XX:", XX.shape
            print "sfreq:", sfreq

            XX = self.create_features(XX, tmin, tmax, sfreq)

            inputs.append(XX)
            ids.append(iids)

        self.inputs = np.vstack(inputs)
        self.n_data, self.n_in_features = self.inputs.shape

        self.ids = np.concatenate(ids)

        print "Trainset:", self.inputs.shape

        self.is_perm = False

        self.n_time_series = 306
        self.len_time_series = 375

    def permute_data_virtual(self, perm):
        self.ids = self.ids[perm]

    def write_virtual(self, h5file):
        DatasetMEG.write_virtual(self, h5file)
        h5file.create_dataset("ids", data=self.ids, dtype='f')

    def read_virtual(self, h5file):
        DatasetMEG.read_virtual(self, h5file)
        self.ids = h5file["ids"].value

if __name__ == '__main__':
    ds_train = DatasetMEGLabelled()
    ds_train.generate(range(1, 17))
    ds_train.write("train_meg.h5")

    ds_train = DatasetMEGUnabelled()
    ds_train.generate(range(17, 24))
    ds_train.write("test_meg.h5")