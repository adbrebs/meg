__author__ = 'adeb'

from database import DataBase
from datasetmeg import DatasetMEG


class DataBaseMEG(DataBase):

    def __init__(self):
        DataBase.__init__(self)

        ds = DatasetMEG()
        ds.generate()

        self.n_out_features = ds.n_out_features
        self.n_data = ds.n_data

        split1 = int(0.15 * self.n_data)
        split2 = split1 + int(0.15 * self.n_data)

        self.test_x = ds.inputs[:split1]
        self.test_y = ds.outputs[:split1]
        self.valid_x = ds.inputs[split1:split2]
        self.valid_y = ds.outputs[split1:split2]
        self.train_x = ds.inputs[split2:]
        self.train_y = ds.outputs[split2:]

        self.share_data(self.test_x, self.test_y, self.valid_x, self.valid_y, self.train_x, self.train_y)
