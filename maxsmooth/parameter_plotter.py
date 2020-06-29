import warnings
import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
import progressbar
from itertools import product
from maxsmooth import Models
from maxsmooth.derivatives import derivative_class
import os

class param_plotter(object):
    def __init__(self, best_params, optimum_signs, x, y, N, **kwargs):
        self.best_params = best_params
        self.optimum_signs = optimum_signs
        self.x = x
        self.y = y

        self.N = N
        if self.N%1!=0:
            raise ValueError('N must be an integer or whole number float.')

        for keys, values in kwargs.items():
            if keys not in set(['fit_type', 'model_type', 'base_dir',
                'ifp_list', 'constraints',
                'basis_functions','der_pres', 'model',
                'derivatives', 'args', 'pivot_point', 'samples',
                'width', 'warnings', 'gridlines']):
                raise KeyError("Unexpected keyword argument in parameter plotter.")

        self.fit_type = kwargs.pop('fit_type', 'qp-sign_flipping')
        if self.fit_type not in set(['qp', 'qp-sign_flipping']):
            raise KeyError("Invalid 'fit_type'. Valid entries include 'qp'\n" +
                "'qp-sign_flipping'")

        self.model_type = kwargs.pop('model_type', 'difference_polynomial')
        if self.model_type not in set(['normalised_polynomial', 'polynomial',
            'log_polynomial', 'loglog_polynomial', 'difference_polynomial',
            'exponential', 'legendre']):
            raise KeyError("Invalid 'model_type'. See documentation for built" +
                "in models.")

        self.basis_functions = kwargs.pop('basis_functions', None)
        self.der_pres = kwargs.pop('der_pres', None)
        self.model = kwargs.pop('model', None)
        self.derivatives_function = kwargs.pop('derivatives', None)
        self.args = kwargs.pop('args', None)

        self.new_basis = {'basis_function':
            self.basis_functions, 'der_pres': self.der_pres,
            'derivatives_function': self.derivatives_function,
            'model': self.model, 'args': self.args}
        if np.all(value is None for value in self.new_basis.values()):
            pass
        else:
            count = 0
            for key, value in self.new_basis.items():
                if value is None and key != 'args':
                    raise KeyError(
                        'Attempt to change basis functions failed.' +
                        ' One or more functions not defined.' +
                        ' Please consult documentation.')
                if value is None and key == 'args':
                    warn('Warning: No additional arguments passed to new basis' +
                        'functions')
                count += 1
            if count == len(self.new_basis):
                self.model_type = 'user_defined'

        self.pivot_point = kwargs.pop('pivot_point', len(self.x)//2)
        if type(self.pivot_point) is not int:
            raise TypeError('Pivot point is not an integer index.')
        elif self.pivot_point >= len(self.x) or self.pivot_point < -len(self.x):
            raise ValueError('Pivot point must be in the range -len(x) - len(x).')

        self.base_dir = kwargs.pop('base_dir', 'Fitted_Output/')
        if type(self.base_dir) is not str:
            raise KeyError("'base_dir' must be a string ending in '/'.")
        elif self.base_dir.endswith('/') is False:
            raise KeyError("'base_dir' must end in '/'.")

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        self.constraints = kwargs.pop('constraints', 2)
        if type(self.constraints) is not int:
            raise TypeError("'constraints' is not an integer")
        if self.constraints > self.N-1:
            raise ValueError("'constraints' exceeds the number of derivatives.")

        self.ifp_list = kwargs.pop('ifp_list', None)
        if self.ifp_list is not None:
            for i in range(len(self.ifp_list)):
                if type(self.ifp_list[i]) is not int:
                    raise TypeError("Entries in 'ifp_list' are not integer.")
                if self.ifp_list[i] < self.constraints:
                    raise ValueError('One or more specified derivatives for' +
                        ' inflection points is less than the minimum constrained' +
                        ' derivative.\n ifp_list = ' + str(self.ifp_list) + '\n' +
                        ' Minimum Constrained Derivative = ' + str(self.constraints))

        self.samples = kwargs.pop('samples', 51)
        if self.samples%1 != 0:
            raise ValueError('Error: Samples must be a whole number.')

        self.width = kwargs.pop('width', 0.5)
        if type(self.width) is not int:
            if type(self.width) is not float:
                raise ValueError('Width must be an integer or a float.')

        self.warnings = kwargs.pop('warnings', True)
        self.gridlines = kwargs.pop('gridlines', False)
        boolean_kwargs = [self.warnings, self.gridlines]
        for i in range(len(boolean_kwargs)):
            if type(boolean_kwargs[i]) is not bool:
                raise TypeError("Boolean keyword argument with value "
                    + str(boolean_kwargs[i]) +
                    " is not True or False.")

        self.plot()


    def plot(self):

        def chi_squared(parameters):
            y_sum = Models.Models_class(parameters, self.x, self.y,
                self.N, self.pivot_point, self.model_type,
                self.new_basis).y_sum
            if self.model_type == 'loglog_polynomial':
                chi = np.sum((np.log10(self.y) - np.log10(y_sum))**2)
            else:
                chi = np.sum((self.y - y_sum)**2)
            return chi

        def plot_formatting(xpos, ypos):
            ypos -= 1
            if xpos == 0:
                axes[ypos, xpos].set_ylabel(r'$a_{%2d}$' %i1, fontsize=12)
            if xpos != 0:
                axes[ypos, xpos].set_yticklabels([])
            if ypos == self.N-2:
                axes[ypos, xpos].set_xlabel(r'$a_{%2d}$' %i2, fontsize=12)
            if ypos != self.N-2:
                axes[ypos, xpos].set_xticklabels([])
            for label in axes[ypos, xpos].get_xticklabels():
                label.set_rotation(90)

        def signs_array(nums):
            return np.array(list(product(*((x, -x) for x in nums))))

        if self.ifp_list is not None:
            available_signs = signs_array([1]*(self.N-self.constraints-len(self.ifp_list)))
        else:
            available_signs = signs_array([1]*(self.N-self.constraints))

        indices = np.array([np.arange(0, self.N, 1), np.arange(0, self.N, 1)])
        combinations = list(itertools.product(*indices))

        combis = []
        for i in range(len(combinations)):
            if combinations[i][0] != combinations[i][1]:
                combis.append(combinations[i])
                for j in range(len(combis)):
                    if combis[-1] == tuple(sorted(combis[j])):
                        combis.remove(combis[-1])

        bar = progressbar.ProgressBar(maxval=len(combis), \
            widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        mapped_colours = []
        cp_array = []
        sign_combinations = []
        warnings_count = 0
        fig, axes = plt.subplots(figsize=(15, 15), nrows=self.N-1, ncols=self.N-1)

        for n in range(self.N-1):
            for m in range(self.N-1):
                if n < m:
                    axes[n, m].axis('off')
        for f in range(len(combis)):
                bar.update(f+1)
                i1 = combis[f][0]
                i2 = combis[f][1]
                p = []
                for i in range(self.N):
                    if i == i1 or i == i2 :
                        p.append(np.linspace(self.best_params[i]*(1 -self.width),
                            self.best_params[i]*(1 + self.width), self.samples))
                p = np.array(p).T[0]

                comb, id = [], []
                for l in range(self.N):
                    if l != i1 and l != i2:
                        id.append(l)
                        comb.append(self.best_params[l])
                comb, id = np.array(comb).T[0], np.array(id)

                X, Y = np.meshgrid(p[:, 0], p[:, 1])

                chi = np.empty(X.shape)
                pf = np.empty(X.shape)
                if self.N <= 5:
                    s = np.empty(X.shape)
                    for i in range(s.shape[0]):
                        for l in range(s.shape[1]):
                            s[i, l] = len(available_signs)+10

                for j in range(X.shape[0]):
                    for a in range(X.shape[1]):
                        parameters = np.empty(self.N)
                        parameters[i1] = Y[j, a].copy()
                        parameters[i2] = X[j, a].copy()
                        for h in range(len(id)):
                            parameters[id[h]] = comb[h]
                        parameters = np.array(parameters)
                        chi[j, a] = chi_squared(parameters)

                        if self.model_type == 'legendre':
                            parameters = np.array([parameters]).T

                        der = derivative_class(self.x, self.y,
                            parameters, self.N,
                            self.pivot_point, self.model_type,
                            self.ifp_list,, self.constraints,
                            self.new_basis, call_type='plotter')

                        derivatives, pass_fail = der.derivatives, der.pass_fail

                        if np.any(pass_fail == 0):
                            pf[j, a] = 0
                        else:
                            pf[j, a] = 1

                        if self.N <=5:
                            signs = []
                            for i in range(derivatives.shape[0]):
                                if (np.all(derivatives[i, :] >= -1e-6)) and \
                                    (np.all(derivatives[i, :] <= 1e-6)):
                                    signs.append(self.optimum_signs*-1)
                                    if self.warnings is True and \
                                        warnings_count == 0:
                                        print('Warning: One or more derivatives'
                                        + ' equals 0 across the band. Optimum'
                                        + ' derivative signs from maxsmooth'
                                        + ' assumed for these derivatives'
                                        + ' which may cause inconsistencies'
                                        + ' in the parameter plot.')
                                        warnings_count += 1
                                elif (np.all(derivatives[i, :] >= -1e-6)):
                                    signs.append(1)
                                elif (np.all(derivatives[i, :] <= 1e-6)):
                                    signs.append(-1)

                            for i in range(len(available_signs)):
                                if np.all(signs == available_signs[i]) and pf[j, a] !=0:
                                    s[j, a] = i

                chi_masked = np.ma.masked_where(pf == 0, chi)
                chi_fail_masked = np.ma.masked_where(pf==1, chi)

                if self.N <= 5:
                    chi_array = []
                    for i in range(len(available_signs)):
                        array = chi_masked.copy()
                        chi_array.append(np.ma.masked_where(s != i, array))

                plot_formatting(i2, i1)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if self.N > 5 or self.constraints != 2:
                        cp = axes[i1 - 1, i2].contourf(X, Y, chi_masked,
                            np.linspace(chi.min(), chi.max(), 10),
                            cmap='autumn')
                        if f == len(combis) - 1:
                            cbax = fig.add_axes([0.61 , 0.8, 0.3, 0.02])
                            clb = plt.colorbar(cp, cax = cbax,
                                orientation='horizontal')
                            clb.ax.set_ylabel(r'Valid Region',rotation=0, fontsize=10)
                            clb.ax.yaxis.set_label_coords(-0.2, 0.1)
                            clb.ax.tick_params(rotation=90)
                    else:
                        cps, mapped_colours_combi, mapped_sign_combinations= [], [], []
                        cmaps=['autumn', 'winter', 'summer', 'spring', 'Greens', 'cool','pink', 'ocean']
                        for i in range(len(available_signs)):
                            if np.all(chi_array[i].mask == True):
                                pass
                            else:
                                cp = axes[i1 - 1, i2].contourf(X, Y, chi_array[i],
                                    np.linspace(chi.min(), chi.max(), 10),
                                    cmap=cmaps[i])
                                cps.append(cp)
                                mapped_colours_combi.append(cmaps[i])
                                mapped_sign_combinations.append(available_signs[i])
                        cp_array.append(cps)
                        mapped_colours.append(mapped_colours_combi)
                        sign_combinations.append(mapped_sign_combinations)
                    cp_fail = axes[i1 - 1, i2].contourf(X, Y, chi_fail_masked,
                        np.linspace(chi.min(), chi.max(), 10),
                        cmap='gray')
                    if f == len(combis) - 1:
                        cbax = fig.add_axes([0.61 , 0.8+0.02, 0.3, 0.02])
                        clb = plt.colorbar(cp_fail, cax = cbax,
                            orientation='horizontal')
                        clb.ax.set_title(r'$\chi^2$')
                        clb.ax.set_ylabel(r'Invalid Region',rotation=0, fontsize=10)
                        clb.ax.yaxis.set_label_coords(-0.2, 0.1)
                        clb.ax.tick_params(labelcolor="none", bottom=False, left=False)

                if self.gridlines is True:
                    axes[i1 - 1, i2].vlines(self.best_params[i2], p[:,1].min(), p[:,1].max(), color='w', ls='--')
                    axes[i1 - 1, i2].hlines(self.best_params[i1], p[:,0].min(), p[:,0].max(), color='w', ls='--')
        bar.finish()

        cbaxes = []
        height = 0.8

        if self.N <= 5:
            mapped_colours = list(itertools.chain.from_iterable(mapped_colours))
            cp_array = list(itertools.chain.from_iterable(cp_array))
            sign_combinations = list(itertools.chain.from_iterable(sign_combinations))
            unique_mapped_colours, indices = np.unique(np.array(mapped_colours), return_index=True)
            for i in range(len(unique_mapped_colours)):
                    if i > 0:
                        height -= 0.02
                    cbaxes.append(fig.add_axes([0.61 , height, 0.3, 0.02]))
            count = 0
            for i in range(len(cp_array)):
                if i in set(indices):
                    clb = plt.colorbar(cp_array[i], cax = cbaxes[count],
                        orientation='horizontal')
                    clb.ax.set_ylabel(r'Signs = ' + str(sign_combinations[i]),
                        rotation=0, fontsize=10)
                    clb.ax.yaxis.set_label_coords(-0.2, 0.1)
                    if i != indices.max():
                        clb.ax.tick_params(labelcolor="none", bottom=False, left=False)
                    else:
                        clb.ax.tick_params(rotation=90)
                    count +=1

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(self.base_dir + 'Parameter_plot.pdf')
        plt.close()
