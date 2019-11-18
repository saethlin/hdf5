import h5py
import numpy as np

with h5py.File('test.hdf5', 'w') as f:
    #f.attrs['file_attr'] = "file attr oof"
    '''
    f.attrs['test_attr2'] = 'this_is_a_test_attr'
    f.attrs['test_attr3'] = 'this_is_a_test_attr'
    f.attrs['test_attr4'] = 'this_is_a_test_attr'
    f.attrs['test_attr5'] = 'this_is_a_test_attr'
    '''
    f['dataset'] = np.arange(100, dtype=np.float64)
    '''
    f['dataset'].attrs['dset_attr_name'] = "oooooooooooofs can be found here"
    g = f.create_group('group')
    g['group_dataset'] = np.arange(10, dtype=np.float64)
    '''
