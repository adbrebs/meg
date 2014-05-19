__author__ = 'adeb'

import os
import numpy as np
from scipy.io import loadmat

from dataset import Dataset


class DatasetMEG(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.n_time_series = None
        self.len_time_series = None

    def generate(self):

        subjects_train = range(1, 6)  # use range(1, 17) for all subjects

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
            print "yy:", yy.shape
            print "sfreq:", sfreq

            XX = self.create_features(XX, tmin, tmax, sfreq)

            inputs.append(XX)
            outputs.append(yy)

        self.inputs = np.vstack(inputs)
        self.n_data, self.n_in_features = self.inputs.shape

        outputs = np.concatenate(outputs)
        self.outputs = np.zeros((self.n_data, 2), dtype=int)
        self.outputs[np.arange(self.n_data), outputs] = 1
        print "Trainset:", self.inputs.shape

        self.n_out_features = self.outputs.shape[1]

        self.permute_data()
        self.is_perm = True

        self.n_time_series = 306
        self.len_time_series = 375

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

if __name__ == '__main__':
    ds = DatasetMEG()
    ds.write("data_meg.h5")