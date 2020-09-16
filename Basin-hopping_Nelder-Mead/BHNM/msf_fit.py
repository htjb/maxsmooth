import numpy as np
from scipy.optimize import minimize, basinhopping
import sys


class max_fit_BH(object):
    def __init__(self, x, y, N, mid_point, **kwargs):

        self.step_size = kwargs.pop('step_size', 0.5)
        self.temp = kwargs.pop('temp', 1)
        self.interval = kwargs.pop('interval', 50)
        self.normalisation = kwargs.pop('data_norm', True)

        if self.normalisation is True:
            self.x = x/x.max()
            self.true_x = x
            self.y = y/y.std() + y.mean() / y.std()
            self.true_y = y
        else:
            self.x = x
            self.y = y

        self.N = N
        self.mid_point = mid_point

        self.fit_result = self.fit()

    def fit(self):

        def func1(params):
            y_sum = self.y[self.mid_point] * np.sum([
                    params[i] * (self.x / self.x[self.mid_point])**i
                    for i in range(self.N)], axis=0)
            message = 'test'
            const = constraint(params, message)
            if const == 1:
                return np.sum((self.y - y_sum)**2)
            if const == -1:
                return np.sum((self.y - y_sum)**2)+1e80

        def constraint(params, message):

            m = np.arange(0, self.N, 1)
            deriv = []
            for j in range(len(m)):
                if m[j] >= 2:
                    dif = []
                    for i in range(self.N - m[j]):
                        if i <= (self.N - m[j]):
                            dif_m_bit = (
                                self.y[self.mid_point] /
                                self.x[self.mid_point]) * \
                                np.math.factorial(m[j]+i) / \
                                np.math.factorial(i) * \
                                params[m[j]+i]*(self.x)**i / \
                                (self.x[self.mid_point])**(i+1)
                            dif.append(dif_m_bit)
                    dif = np.array(dif)
                    derivative = dif.sum(axis=0)
                    deriv.append([m[j], derivative])
            deriv = np.array(deriv)
            derive = []
            for i in range(deriv.shape[0]):
                derive.append(deriv[i, 1])
            derive = np.array(derive)
            pass_fail = []  # 1==pass, 0==fail
            for i in range(derive.shape[0]):
                if np.any(derive[i, :] == 1e-7):
                    pass_fail.append(0)
                elif np.any(derive[i, :] > -1e-7) and \
                        np.any(derive[i, :] < 1e-7):
                    pass_fail.append(0)
                else:
                    pass_fail.append(1)
            pass_fail = np.array(pass_fail)

            if np.any(pass_fail == 0):
                const = -1  # failed
            else:
                const = 1  # satisfied

            return const

        params0 = [(self.y[-1]-self.y[0])/2]*(self.N)
        res = basinhopping(
            func1, params0, niter=10000, niter_success=100*self.N,
            stepsize=self.step_size, T=self.temp, interval=self.interval,
            seed=1)
        print('msf fit params', res)

        parameters = res.x.copy()
        if self.normalisation is True:
            for i in range(len(parameters)):
                if i == 0:
                    parameters[i] = parameters[i] * \
                        (1+self.true_y.mean()/self.true_y[self.mid_point]) \
                        - self.true_y.mean()/(self.true_y[self.mid_point])
                else:
                    parameters[i] = parameters[i] * \
                        (1+self.true_y.mean()/self.true_y[self.mid_point])

        message = 'summary'
        const = constraint(parameters, message)
        if const != 1:
            print('Error: condition violated')
            sys.exit(1)

        def fitting(params):
            if self.normalisation is True:
                y_sum = self.true_y[self.mid_point]*np.sum([
                        params[i]*(self.true_x/self.true_x[self.mid_point])**i
                        for i in range(self.N)], axis=0)
            else:
                y_sum = self.y[self.mid_point]*np.sum([
                        params[i]*(self.x/self.x[self.mid_point])**i
                        for i in range(self.N)], axis=0)
            return y_sum

        fitted_y = fitting(parameters)
        if self.normalisation is True:
            print('chi BH', np.sum((self.true_y-fitted_y)**2))
        else:
            print('chi BH', np.sum((self.y-fitted_y)**2))

        return res.x


class max_fit_NM(object):
    def __init__(self, x, y, N, BH_params, mid_point, **kwargs):

        self.normalisation = kwargs.pop('data_norm', True)

        if self.normalisation is True:
            self.x = x/x.max()
            self.true_x = x
            self.y = y/y.std() + y.mean()/y.std()
            self.true_y = y
        else:
            self.x = x
            self.y = y

        self.N = N
        self.BH_params = BH_params
        self.mid_point = mid_point
        self.fit_result, self.fit_h, self.chi = self.fit()

    def fit(self):
        print('-------------------------------------------------------------')

        def func1(params):
            y_sum = self.y[self.mid_point]*np.sum([
                    params[i]*(self.x/self.x[self.mid_point])**i
                    for i in range(self.N)], axis=0)
            message = 'test'
            const, h = constraint(params, message)
            if const == 1:
                return np.sum((self.y-y_sum)**2)
            if const == -1:
                return np.sum((self.y-y_sum)**2)+1e80

        def constraint(params, message):

            m = np.arange(0, self.N, 1)
            deriv = []
            for j in range(len(m)):
                if m[j] >= 2:
                    dif = []
                    for i in range(self.N-m[j]):
                        if i <= (self.N-m[j]):
                            dif_m_bit = (
                                self.y[self.mid_point] /
                                self.x[self.mid_point]) * \
                                np.math.factorial(m[j]+i) / \
                                np.math.factorial(i) * \
                                params[m[j]+i]*(self.x)**i / \
                                (self.x[self.mid_point])**(i+1)
                            dif.append(dif_m_bit)
                    dif = np.array(dif)
                    derivative = dif.sum(axis=0)
                    deriv.append([m[j], derivative])
            deriv = np.array(deriv)
            derive = []
            for i in range(deriv.shape[0]):
                derive.append(deriv[i, 1])
            derive = np.array(derive)
            if message == 'summary':
                print('min derivative', derive.min())
                h = np.abs(derive).min()/2
            if message == 'test':
                h = None
                pass
            pass_fail = []  # 1==pass, 0==fail
            for i in range(derive.shape[0]):
                if np.any(derive[i, :] == 1e-7):
                    pass_fail.append(0)
                elif np.any(derive[i, :] > -1e-7) and \
                        np.any(derive[i, :] < 1e-7):
                    pass_fail.append(0)
                else:
                    pass_fail.append(1)
            pass_fail = np.array(pass_fail)

            if np.any(pass_fail == 0):
                const = -1  # failed
            else:
                const = 1  # satisfied

            return const, h

        def fitting(params):
            if self.normalisation is True:
                y_sum = self.true_y[self.mid_point]*np.sum([
                        params[i]*(self.true_x/self.true_x[self.mid_point])**i
                        for i in range(self.N)], axis=0)
            else:
                y_sum = self.y[self.mid_point]*np.sum([
                        params[i]*(self.x/self.x[self.mid_point])**i
                        for i in range(self.N)], axis=0)
            return y_sum

        params0 = self.BH_params

        res = minimize(
            func1, params0,
            options={
                'maxiter': 100000, 'adaptive': True, 'fatol': 1e-7,
                'xatol': 1e-7}, method='Nelder-Mead')
        print(res)

        parameters = res.x
        if self.normalisation is True:
            for i in range(len(parameters)):
                if i == 0:
                    parameters[i] = parameters[i] * \
                        (1+self.true_y.mean()/self.true_y[self.mid_point]) \
                        - self.true_y.mean()/(self.true_y[self.mid_point])
                else:
                    parameters[i] = parameters[i] * \
                        (1+self.true_y.mean()/self.true_y[self.mid_point])

        fitted_y = fitting(parameters)
        if self.normalisation is True:
            print('chi NM', np.sum((self.true_y-fitted_y)**2))
            chi = np.sum((self.true_y-fitted_y)**2)
        else:
            print('chi NM', np.sum((self.y-fitted_y)**2))
            chi = np.sum((self.y-fitted_y)**2)

        message = 'summary'
        const, h = constraint(parameters, message)
        if const != 1:
            print('Error: condition violated')
            sys.exit(1)

        return parameters, h, chi
