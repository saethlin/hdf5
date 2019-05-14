import h5py
import numpy as np

with h5py.File('test.hdf5', 'w') as f:
    f.attrs['file_attr_name'] = 'file_attr_contents'
    '''
    f.attrs['test_attr2'] = 'this_is_a_test_attr'
    f.attrs['test_attr3'] = 'this_is_a_test_attr'
    f.attrs['test_attr4'] = 'this_is_a_test_attr'
    f.attrs['test_attr5'] = 'this_is_a_test_attr'
    '''
    f['test_dataset'] = np.arange(100, dtype=np.float64)
    f['test_dataset'].attrs['dset_attr_name'] = b'dset_attr_contents'
    f['test_dataset2'] = np.arange(2, 100, dtype=np.float64)
    g = f.create_group('test_group')
    g['tg_dataset'] = np.arange(10, dtype=np.float64)
