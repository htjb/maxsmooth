import numpy as np
import pylab as pl
from maxsmooth.settings import setting
from maxsmooth.msf import msf_fit
from cvxopt import matrix

setting=setting()

#EDGES Data
freq,weight,Tsky,Tres1,Tres2,Tmodel,T21=np.loadtxt('EDGES_Data/figure1_plotdata.csv',delimiter=',',unpack=True,usecols=[0,1,2,3,4,5,6],skiprows=1)
x=[]
y=[]
for i in range(len(Tsky)):
    if Tsky[i]!=0:
        y.append(Tsky[i])
        x.append(freq[i])
x=np.array(x)
y=np.array(y)

N=[3]

#setting.model_type='normalised_polynomial'
#setting.all_output=True
setting.data_save=False

def derivative_pre(m, i, x, y, mid_point):

    mth_order_derivative_term = (
        y[mid_point] /
        x[mid_point]) \
        * np.math.factorial(m + i) \
        / np.math.factorial(i) * \
        (x)**i/(x[mid_point])**(i + 1)

    return mth_order_derivative_term

params0 = [y[0]]*(N[0])

def basis_functions(x, y, mid_point, N):

    # This function calculates the basis function values.

    A = np.empty([len(x), N])
    for h in range(len(x)):
        for i in range(N):
            A[h, i] = y[mid_point] * (
                x[h] / x[mid_point])**i
    A = matrix(A)

    return A

b = matrix(y, (len(y), 1), 'd')

def model(x, y, mid_point, N, params, *args):

    # This function takes the optimal parameters and calculates the
    # fitted y as a function of x.

    a=args[0]

    y_sum = a*y[mid_point]*np.sum(
        [params[i]*(x/x[mid_point])**i
        for i in range(N)], axis=0)

    return y_sum

def derivative(m, i, x, y, mid_point, params, *args):

    # This function calculates the terms of the mth order derivatives
    # to be summed up by maxsmooth. This is for plotting and checking
    # that the constraints are met.
    b = args[1]
    mth_order_derivative_term = (
        y[mid_point]/x[mid_point]) * \
        np.math.factorial(m+i) / \
        np.math.factorial(i) * \
        params[int(m)+i]*(x)**i / \
        (x[mid_point])**(i+1)*b
    return mth_order_derivative_term

a=10
b=100
arguments=[a,b]

result=msf_fit(x,y,N,setting, derivatives_function=derivative, model=model, args=arguments)# data_matrix=b)#basis_functions=basis_functions)# der_pres=derivative_pre)#, initial_params=params0)
print('Objective Funtion Evaluations:\n', result.Optimum_chi)
#print('Parameters:\n',result.Optimum_params[2])
print('RMS:\n',result.rms)
#print('Fitted Temps:\n',result.y_fit)
#print('Sign Combinations:\n',result.Optimum_signs)
#print('Derivatives:\n',result.derivatives)

#pl.subplot(111)
#pl.plot(N,result.rms)
#pl.ylim([0,1])
#pl.show()
