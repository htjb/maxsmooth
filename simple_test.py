import numpy as np
from maxsmooth.settings import setting
from maxsmooth.msf import smooth

setting = setting()
setting.data_save = True

x = np.load('Data/x.npy')
y = np.load('Data/y.npy')

N = [3, 4, 5, 6, 7, 8, 9, 10, 11]

result = smooth(x, y, N, setting)
print('Objective Funtion Evaluations:\n', result.Optimum_chi)
print('RMS:\n',result.rms)
# print('Parameters:\n',result.Optimum_params[2])
# print('Fitted y:\n',result.y_fit)
# print('Sign Combinations:\n',result.Optimum_signs)
# print('Derivatives:\n',result.derivatives)
