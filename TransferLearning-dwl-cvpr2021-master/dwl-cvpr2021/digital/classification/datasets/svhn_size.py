from scipy.io import loadmat
import numpy as np
import sys

#sys.path.append('../utils/')
#from utils.utils import dense_to_one_hot

svhn_train = loadmat('./data/svhn/train_32x32.mat')
svhn_test = loadmat('./data/svhn/test_32x32.mat')
svhn_train_im = svhn_train['X']
print(np.shape(svhn_train_im))
