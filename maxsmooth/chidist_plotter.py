"""

This function allows the user to produce plots of the :math:`{\chi^2}`
distribution as a function of the available discrete sign spaces for the
constrained derivatives. This can be used to identify whether or not the
problem is `ill defined`, see the ``maxsmooth`` paper for a definition,
and if it can be solved using the sign sampling approach.

It can also be used to determine whether or not the 'cap' and maximum allowed
increase on the value of :math:`{\chi^2}` during the directional exploration
are sufficient to identify the global minimum for the problem.

The function is reliant on the output of the ``maxsmooth`` smooth() function.
The required outputs can be saved when running smooth()
using the 'data_save = True' kwarg.
"""

import numpy as np
import os
from itertools import product
import matplotlib.pyplot as plt


class chi_plotter(object):

    r"""

    **Parameters:**

        N: **int**
            | The number of terms in the DCF.

    **Kwargs:**

        fit_type: **Default = 'qp-sign_flipping'**
            | This kwarg is the same as for the smooth() function.
                Here it allows the files to be read from the base
                directory.

        base_dir: **Default = 'Fitted_Output/'**
            | The location of the outputted
                data from ``maxsmooth``. This must be a string and end in '/'
                and must contain the files 'Output_Evaluations/' and
                'Output_Signs/' which can be obtained by running smooth() with
                data_save=True.

        chi: **Default = None else list or numpy array**
            | A list of
                :math:`{\chi^2}` evaluations. If provided then this is used
                over outputted data in the base directory. It must have the
                same length as the ouputted signs in the file 'Output_Signs/'
                in the base directory. It must also be ordered correctly
                otherwise the returned graph will not be correct. A correct
                ordering is one for which each entry in the array corresponds
                to the correct sign combination in 'Output_Signs/'.
                Typically this will not be needed but if the :math:`{\chi^2}`
                evaluation in 'Output_Evaluations/' in the base directory
                is not in the desired parameter space this can be useful.
                For example the built in logarithmic model calculates
                :math:`{\chi^2}` in logarithmic space. To plot the distribution
                in linear space we can calculate
                :math:`{\chi^2}` in linear space using a function for the model
                and the tested parameters which are found in
                'Output_Parameters/' in the base directory.

        **constraints: Default = 2 else an integer less than or equal**
        **to N - 1**
            | The minimum constrained derivative order which is set by default
                to 2 for a Maximally Smooth Function. Used here to determine
                the number of possible sign combinations available.

        zero_crossings: **Default = None else list of integers**
            | Allows you to
                specify if the conditions should be relaxed on any
                of the derivatives between constraints and the highest order
                derivative. e.g. a 6th order fit with just a constrained 2nd
                and 3rd order derivative would have a zero_crossings = [4, 5].
                Again this is used in determining the possible sign
                combinations available.

        plot_limits: **Default = False**
            | Determines whether the limits on
                the directional exploration are plotted on top of the
                :math:`{\chi^2}` distribution.

        cap: **Default = (len(available_signs)//N) + N else an integer**
            | Determines the maximum number of signs explored either side of
                the minimum :math:`{\chi^2}` value found after the
                decent algorithm has terminated when running smooth(). Here
                it is used when plot_limits=True.

        chi_squared_limit: **Default = 2 else float or int**
            | The prefactor on the maximum allowed increase in :math:`{\chi^2}`
                during the directional exploration which is defaulted at 2.
                If this value multiplied by the minimum :math:`{\chi^2}`
                value found after the descent algorithm is exceeded then the
                exploration in one direction is stopped and started in the
                other. For more details on this and 'cap' see the ``maxsmooth``
                paper. Again this is used here
                when plot_limits=True.

    """
    def __init__(self, N, **kwargs):

        self.N = N
        if self.N % 1 != 0:
            raise ValueError('N must be an integer or whole number float.')

        for keys, values in kwargs.items():
            if keys not in set([
                    'chi', 'base_dir',
                    'zero_crossings', 'constraints',
                    'fit_type', 'chi_squared_limit', 'cap', 'plot_limits']):
                raise KeyError("Unexpected keyword argument in chi_plotter().")

        self.base_dir = kwargs.pop('base_dir', 'Fitted_Output/')
        if type(self.base_dir) is not str:
            raise KeyError("'base_dir' must be a string ending in '/'.")
        elif self.base_dir.endswith('/') is False:
            raise KeyError("'base_dir' must end in '/'.")

        if not os.path.exists(self.base_dir):
            raise Exception(
                "'base_dir' must exist and contain the outputted"
                + " evaluations and sign combinations from a maxsmooth fit."
                + " These can be obtained by running maxsmooth with"
                + " 'data_save=True'.")
        else:
            if not os.path.exists(self.base_dir + 'Output_Evaluation/'):
                raise Exception(
                    "No 'Output_Evaluation/' directory found in"
                    + " 'base_dir'.")
            if not os.path.exists(self.base_dir + 'Output_Signs/'):
                raise Exception(
                    "No 'Output_Signs/' directory found in"
                    + " 'base_dir'.")

        self.chi = kwargs.pop('chi', None)

        self.constraints = kwargs.pop('constraints', 2)
        if type(self.constraints) is not int:
            raise TypeError("'constraints' is not an integer")
        if self.constraints > self.N-1:
            raise ValueError(
                "'constraints' exceeds the number" +
                " of derivatives.")

        self.zero_crossings = kwargs.pop('zero_crossings', None)
        if self.zero_crossings is not None:
            for i in range(len(self.zero_crossings)):
                if type(self.zero_crossings[i]) is not int:
                    raise TypeError(
                        "Entries in 'zero_crossings'" +
                        " are not integer.")
                if self.zero_crossings[i] < self.constraints:
                    raise ValueError(
                        'One or more specified derivatives for' +
                        ' zero crossings is less than the minimum' +
                        ' constrained' +
                        ' derivative.\n zero_crossings = '
                        + str(self.zero_crossings)
                        + '\n' + ' Minimum Constrained Derivative = '
                        + str(self.constraints))

        self.fit_type = kwargs.pop('fit_type', 'qp-sign_flipping')
        if self.fit_type not in set(['qp', 'qp-sign_flipping']):
            raise KeyError(
                "Invalid 'fit_type'. Valid entries include 'qp'\n" +
                "'qp-sign_flipping'")

        self.chi_squared_limit = kwargs.pop('chi_squared_limit', None)
        self.cap = kwargs.pop('cap', None)
        if self.chi_squared_limit is not None:
            if isinstance(self.chi_squared_limit, int) or \
                    isinstance(self.chi_squared_limit, float):
                pass
            else:
                raise TypeError(
                    "Limit on maximum allowed increase in chi squared" +
                    ", 'chi_squared_limit', is not an integer or float.")
        if self.cap is not None:
            if type(self.cap) is not int:
                raise TypeError(
                    "The cap on directional exploration" +
                    ", 'cap', is not an integer.")

        self.plot_limits = kwargs.pop('plot_limits', False)
        if type(self.plot_limits) is not bool:
            raise TypeError(
                "Boolean keyword argument with value "
                + " 'plot_limits' is not True or False.")

        self.plot()

    def plot(self):

        def signs_array(nums):
            return np.array(list(product(*((x, -x) for x in nums))))

        if self.zero_crossings is not None:
            possible_signs = signs_array([1]*(
                self.N-self.constraints-len(self.zero_crossings)))
        else:
            possible_signs = signs_array([1]*(self.N-self.constraints))

        plt.figure()
        j = np.arange(0, len(possible_signs), 1)
        if self.chi is None:
            chi = np.loadtxt(
                self.base_dir + 'Output_Evaluation/'
                + str(self.N) + '_' + str(self.fit_type) + '.txt')
            signs = np.loadtxt(
                self.base_dir + 'Output_Signs/'
                + str(self.N) + '_' + str(self.fit_type) + '.txt')
            if len(signs) != len(possible_signs):
                index = []
                for p in range(len(signs)):
                    for i in range(len(possible_signs)):
                        if np.all(signs[p] == possible_signs[i]):
                            index.append(i)
                index, chi = zip(*sorted(zip(index, chi)))
                plt.plot(index, chi, ls='-')
            else:
                plt.plot(j, chi, marker='.', ls='-')
        else:
            chi = self.chi
            signs = np.loadtxt(
                self.base_dir + 'Output_Signs/'
                + str(self.N) + '_' + str(self.fit_type) + '.txt')
            if len(signs) != len(possible_signs):
                index = []
                for p in range(len(signs)):
                    for i in range(len(possible_signs)):
                        if np.all(signs[p] == possible_signs[i]):
                            index.append(i)
                index, chi = zip(*sorted(zip(index, chi)))
                plt.plot(index, chi, marker='.', ls='-')
            else:
                plt.plot(j, chi, marker='.', ls='-')

        if self.cap is None:
            self.cap = (len(possible_signs)//self.N) + self.N
        if self.chi_squared_limit is None:
            self.chi_squared_limit = 2*min(chi)

        for i in range(len(chi)):
            if chi[i] == min(chi):
                plt.plot(i, chi[i], marker='*')
                if self.plot_limits is True:
                    plt.vlines(
                        i + self.cap, min(chi), max(chi), ls='--',
                        label='Cap On Exp.', color='k', alpha=0.5)
                    plt.vlines(
                        i - self.cap,  min(chi), max(chi),
                        ls='--', color='k', alpha=0.5)
        if self.plot_limits is True:
            min_chi = np.load(self.base_dir + str(self.N) +
                '_'+self.fit_type+'_minimum_chi_post_descent.npy')
            plt.hlines(
                self.chi_squared_limit*min_chi, 0, len(possible_signs),
                ls='-.', label=r'Max. Increase\n' + ' in $\chi^2$',
                color='k', alpha=0.5)
        plt.xlim([j[0], j[-1]])
        plt.grid()
        plt.yscale('log')
        plt.ylabel(r'$\chi^2$')
        plt.xlabel('Sign Combination')
        plt.tight_layout()
        plt.savefig(self.base_dir + 'chi_distribution.pdf')
        plt.close()
