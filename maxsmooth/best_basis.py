import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from maxsmooth.DCF import smooth

class basis_test(object):
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y

        for keys, values in kwargs.items():
            if keys not in set(['fit_type', 'base_dir', 'N']):
                print("Error: Unexpected keyword argument in basis_test().")
                sys.exit(1)

        self.fit_type = kwargs.pop('fit_type', 'qp-sign_flipping')
        if self.fit_type not in set(['qp', 'qp-sign_flipping']):
            print("Error: Invalid 'fit_type'. Valid entries include 'qp'\n" +
                "'qp-sign_flipping'")
            sys.exit(1)

        self.base_dir = kwargs.pop('base_dir', 'Fitted_Output/')

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        self.N = kwargs.pop('N', np.arange(3, 14, 1))
        if np.any(self.N%1 != 0):
            print('Error: N must be an array or list of integers.')
            sys.exit(1)

        self.test()

    def test(self):

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        def fit(model_type):

            chi = []
            N_passed = []
            for i in range(len(self.N)):
                try:

                    result = smooth(self.x, self.y, self.N[i], model_type=model_type,
                        fit_type=self.fit_type)

                    if model_type == 'loglog_polynomial':
                        chi.append(np.sum((self.y - result.y_fit)**2))
                    else:
                        chi.append(result.optimum_chi)
                    N_passed.append(self.N[i])
                except:
                    pass
            chi = np.array(chi)
            min_chi = chi.min()
            for i in range(len(chi)):
                if chi[i] == chi.min():
                    best_N = self.N[i]

            return chi, min_chi, N_passed, best_N

        chi_poly, min_chi_poly, N_poly, best_N_poly = fit('polynomial')
        chi_np, min_chi_np, N_np, best_N_np = fit('normalised_polynomial')
        chi_log, min_chi_log, N_log, best_N_log = fit('log_polynomial')
        chi_loglog, min_chi_loglog, N_loglog, best_N_loglog = fit('loglog_polynomial')
        chi_leg, min_chi_leg, N_leg, best_N_leg = fit('legendre')
        chi_exp, min_chi_exp, N_exp, best_N_exp = fit('exponential')
        chi_dif, min_chi_dif, N_dif, best_N_dif = fit('difference_polynomial')

        plt.figure(figsize=(3.15, 2.5))
        plt.plot(N_np, chi_np, label='Normalised Polynomial', c='b')
        plt.plot(N_poly, chi_poly, label='Polynomial', c='k')
        plt.plot(N_dif, chi_dif, label='Difference Polynomial', c='r')
        plt.plot(N_log, chi_log, label=r'Log Polynomial', c='g')
        plt.plot(N_loglog, chi_loglog, label=r'Log Log Polynomial', c='orange')
        plt.plot(N_leg, chi_leg, label='Legendre', c='purple')
        plt.plot(N_exp, chi_exp, label='Exponential', c='magenta')
        plt.legend()
        plt.xlabel(r'N')
        plt.xticks(self.N)
        plt.ylabel(r'$\chi^2$')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(self.base_dir + 'Basis_functions.pdf')
        plt.show()
