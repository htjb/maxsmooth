import numpy as np
from scipy.special import legendre


class Models_class(object):
    def __init__(self, params, x, y, N, pivot_point, model_type, new_basis):
        self.x = x
        self.y = y
        self.N = N
        self.params = params
        self.pivot_point = pivot_point
        self.model_type = model_type
        self.model = new_basis['model']
        self.args = new_basis['args']
        self.y_sum = self.fit()

    def fit(self):

        if self.model is None:
            if self.model_type == 'normalised_polynomial':

                y_sum = self.y[self.pivot_point]*np.sum([
                    self.params[i]*(self.x/self.x[self.pivot_point])**i
                    for i in range(self.N)], axis=0)

            if self.model_type == 'polynomial':

                y_sum = np.sum(
                    [self.params[i]*(self.x)**i for i in range(self.N)],
                    axis=0)

            if self.model_type == 'loglog_polynomial':

                y_sum = 10**(np.sum([
                    self.params[i]*np.log10(self.x)**i
                    for i in range(self.N)],
                    axis=0))

            if self.model_type == 'exponential':

                y_sum = self.y[self.pivot_point]*np.sum([
                    self.params[i] *
                    np.exp(-i*self.x/self.x[self.pivot_point])
                    for i in range(self.N)],
                    axis=0)

            if self.model_type == 'log_polynomial':

                y_sum = np.sum([
                    self.params[i] *
                    np.log10(self.x/self.x[self.pivot_point])**i
                    for i in range(self.N)],
                    axis=0)

            if self.model_type == 'difference_polynomial':

                y_sum = np.sum([
                    self.params[i]*(self.x-self.x[self.pivot_point])**i
                    for i in range(self.N)], axis=0)

            if self.model_type == 'legendre':

                interval = np.linspace(-0.999, 0.999, len(self.x))
                lps = []
                for n in range(self.N):
                    P = legendre(n)
                    lps.append(P(interval))
                lps = np.array(lps)
                y_sum = np.sum([
                    self.params[i] * lps[i] for i in range(self.N)], axis=0)

        if self.model is not None:
            if self.args is None:
                y_sum = self.model(
                    self.x, self.y, self.pivot_point, self.N, self.params)
            if self.args is not None:
                y_sum = self.model(
                    self.x, self.y, self.pivot_point, self.N, self.params,
                    *self.args)
        return y_sum
