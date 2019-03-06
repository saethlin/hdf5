import h5py
import numpy as np

with h5py.File('test.hdf5', 'w') as f:
    f.attrs['test_attr'] = 'this_is_a_test_attr'
    f['test_dataset'] = np.arange(100)
    f['test_dataset'].attrs['test'] = 'dataset_attr'
