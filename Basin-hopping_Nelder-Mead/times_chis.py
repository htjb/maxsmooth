import numpy as np
import matplotlib.pyplot as plt
from maxsmooth.DCF import smooth
from BHNM.msf_fit import max_fit_BH, max_fit_NM
import time


def fit(params, mid_point, x, y, N):
    y_sum = y[mid_point]*np.sum([
            params[i]*(x/x[mid_point])**i
            for i in range(N)], axis=0)
    return y_sum


"""x = np.linspace(1, 5, 100)
noise = np.random.normal(0,0.5,len(x))
y = 0.6+2*x+4*x**(3)+9*x**(4)+noise"""

x = np.loadtxt('x_data_for_comp.txt')
y = np.loadtxt('y_data_for_comp.txt')

N = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Testing all signs
times_qp = []
chi_qp = []
for i in range(len(N)):
    s = time.time()
    result = smooth(
        x, y, N[i], model_type='normalised_polynomial', fit_type='qp')
    e = time.time()
    times_qp.append(e-s)
    chi_qp.append(result.optimum_chi)

# Using the maxsmooth sign navigating algorithm
times_qpsf = []
chi_qpsf = []
for i in range(len(N)):
    s = time.time()
    result = smooth(x, y, N[i], model_type='normalised_polynomial')
    e = time.time()
    times_qpsf.append(e-s)
    chi_qpsf.append(result.optimum_chi)

np.savetxt('times_qp_for_comp.txt', times_qp)
np.savetxt('times_qpsf_for_comp.txt', times_qpsf)
np.savetxt('chi_qp_for_comp.txt', chi_qp)
np.savetxt('chi_qpsf_for_comp.txt', chi_qpsf)

# Basin-hopping followed by a Nelder-Mead routine
N_BHNM = np.arange(3, 8, 1)
mid_point = len(x)//2
BHNM_chis_defualt = []
BHNM_times_defualt = []
h = []
for i in range(len(N_BHNM)):
    s = time.time()
    BH = max_fit_BH(
        x, y, N_BHNM[i], mid_point, step_size=10*N_BHNM[i],
        interval=50, temp=5 * N_BHNM[i])
    NM = max_fit_NM(x, y, N_BHNM[i], BH.fit_result, mid_point)
    fitted_y = fit(NM.fit_result, mid_point, x, y, N_BHNM[i])
    e = time.time()
    print(
        'N:', N_BHNM[i], 'TIME:', e-s, 'CHI_RATIO:',
        np.sum((y-fitted_y)**2)/chi_qpsf[i])
    print('CHI_BHNM:', np.sum((y-fitted_y)**2), 'CHI_QP:', chi_qpsf[i])
    chi = np.sum((y-fitted_y)**2)
    BHNM_chis_defualt.append(chi)
    BHNM_times_defualt.append(e-s)

np.save('BHNM_times.npy', BHNM_times_defualt)
np.save('BHNM_chis.npy', BHNM_chis_defualt)

chi_qp_plotting = []
for i in range(len(N)):
    if N[i] <= N_BHNM.max():
        chi_qp_plotting.append(chi_qpsf[i])

ratio = []
for i in range(len(BHNM_chis_defualt)):
    ratio.append(BHNM_chis_defualt[i]/chi_qp_plotting[i])

plt.subplot(111)
plt.plot(N_BHNM, ratio, ls='-', c='k')
plt.xlabel('N', fontsize=12)
plt.ylabel(
    r'$\frac{X^2_{Basinhopping + Nelder-Mead}}{X^2_{QP Sampling Sign Space}}$',
    fontsize=12)
plt.tight_layout()
plt.savefig('chi_ratio.pdf')
plt.close()

plt.subplot(111)
plt.plot(N, times_qp, ls='-', label='QP Testing All Sign Combinations', c='g')
plt.plot(N, times_qpsf, ls='-', label='QP Sampling Sign Space', c='k')
plt.plot(
    N_BHNM, BHNM_times_defualt,
    ls='-', label='Basinhopping + Nelder-Mead\n', c='r')
plt.xlabel('N', fontsize=12)
plt.ylabel(r'$t$ [s]', fontsize=12)
plt.yscale('log')
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('Times.pdf')
plt.close()

plt.subplot(111)
plt.plot(N, chi_qp, ls='-', label='QP Testing All Sign Combinations', c='g')
plt.plot(N, chi_qpsf, ls='-', label='QP Sampling Sign Space', c='k')
plt.plot(
    N_BHNM, BHNM_chis_defualt,
    ls='-', label='Basinhopping + Nelder-Mead', c='r')
plt.xlabel('N', fontsize=12)
plt.ylabel(r'$X^2$', fontsize=12)
plt.yscale('log')
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('chi.pdf')
plt.close()
