import numpy as np
import pylab as pl
#from cvxopt.solvers import qp
from cvxopt import matrix, solvers
from maxsmooth.fit_model import function_fit
from maxsmooth.derivatives import derivative_class
import sys
import warnings

warnings.simplefilter('always', UserWarning)


class max_fit_qp(object):
    def __init__(self,x,y,N,signs,mid_point,model_type,cvxopt_maxiter,filtering,all_output,ifp,ifp_list):
        self.x=x
        self.y=y
        self.N=N
        self.signs=signs
        self.mid_point=mid_point
        self.model_type=model_type
        self.cvxopt_maxiter=cvxopt_maxiter
        self.filtering=filtering
        self.all_output=all_output
        self.ifp=ifp
        self.ifp_list=ifp_list
        self.parameters, self.obj_func,self.pass_fail=self.fit()

    def fit(self):

        solvers.options['maxiters']=self.cvxopt_maxiter
        solvers.options['show_progress']=False

        def constraint(m):

            deriv=[]
            #sign=np.random.randint(0,2)
            #if sign == 0:
            #    sign =-1
            if m>=2:
                for i in range(self.N):
                    if i<=m-1:
                        deriv.append([0]*len(self.x))
                for i in range(self.N-m):
                    if i<(self.N-m) or i==(self.N-m):
                        if self.model_type=='normalised_polynomial':
                            dif_m_bit=(self.y[self.mid_point]/self.x[self.mid_point])*np.math.factorial(m+i)/np.math.factorial(i)*(self.x)**i/(self.x[self.mid_point])**(i+1)
                            deriv.append(dif_m_bit)
                        if self.model_type=='polynomial':
                            dif_m_bit=np.math.factorial(m+i)/np.math.factorial(i)*(self.x)**i
                            deriv.append(dif_m_bit)
                        if self.model_type=='MSF_2017_polynomial':
                            dif_m_bit=np.math.factorial(m+i)/np.math.factorial(i)*(self.x-self.x[self.mid_point])**i
                            deriv.append(dif_m_bit)
                        if self.model_type=='logarithmic_polynomial':
                            dif_m_bit=np.math.factorial(m+i)/np.math.factorial(i)*(np.log10(self.x))**i
                            deriv.append(dif_m_bit)
            deriv=np.array(deriv).astype(np.double)
            deriv=matrix(deriv)
            derivatives=deriv.T
            #print(derivatives)
            return derivatives # , sign

        m=np.arange(0,self.N,1)
        derive=[]
        signs=matrix(self.signs)
        max_derive_prefactors=[]
        for i in range(len(m)):
            if m[i]>=2:
                derivative_prefactors=constraint(m[i])
                if derivative_prefactors != []:
                    derive.append(signs[i-2]*derivative_prefactors)

        G=matrix(derive)
        #print(G)

        A=np.empty([len(self.x),self.N])
        for h in range(len(self.x)):
            for i in range(self.N):
                if self.model_type=='normalised_polynomial':
                    A[h,i]=self.y[self.mid_point]*(self.x[h]/self.x[self.mid_point])**i
                if self.model_type=='polynomial':
                    A[h,i]=(self.x[h])**i
                if self.model_type=='MSF_2017_polynomial':
                    A[h,i]=(self.x[h]-self.x[self.mid_point])**i
                if self.model_type=='logarithmic_polynomial':
                    A[h,i]=np.log10(self.x[h])**i
        A=matrix(A)
        if self.model_type=='logarithmic_polynomial':
            b=matrix(np.log10(self.y),(len(self.y),1),'d')
        else:
            b=matrix(self.y,(len(self.y),1),'d')

        if self.ifp==False:
            h = matrix(-1e-7,((self.N-2)*len(self.x),1),'d')
        if self.ifp==True:
            if self.ifp_list == 'None':
                print('ERROR: setting.ifp set to True but no derivatives selected. Please state which derivatives you would like to allow inflection points in by setting ifp_list to a list of derivative orders(see settings.py for more information). ')
                sys.exit(1)
            else:
                h_ifp_bit=[]
                ifp_list=np.array(self.ifp_list)
                for i in range(self.N-2):
                    if np.any(ifp_list-2 == i):
                        h_ifp_bit.append([1e20]*(len(self.x)))
                    else:
                        h_ifp_bit.append([-1e-7]*(len(self.x)))
                h_ifp=np.array(h_ifp_bit)
                for i in range(h_ifp.shape[0]):
                    if i == 0:
                        hifp=np.array(h_ifp[i])
                    else:
                        hifp=np.concatenate([hifp,h_ifp[i]])
                h=matrix(hifp.T)
        Q=A.T*A
        q= -A.T*b
        if self.model_type=='logarithmic_polynomial':
            params0=[(np.log10(self.y[-1])-np.log10(self.y[0]))/2]*(self.N)
        else:
            params0=[(self.y[-1]-self.y[0])/2]*(self.N)
        qpfit = solvers.coneqp(Q,q, G, h, initvals=params0)#['x']# dims)['x']


        parameters=qpfit['x']
        y=function_fit(parameters,self.x,self.y,self.N,self.mid_point,self.model_type).y_sum
        der=derivative_class(self.x,self.y,parameters,self.N,self.signs,self.mid_point,self.model_type,self.ifp)
        derive=der.derive
        pass_fail=der.pass_fail
        if 'unknown' in qpfit['status']:
            if qpfit['iterations']==self.cvxopt_maxiter:
                print('ERROR: "Maximum number of iterations reached in cvxopt routine." Increase value of setting.cvxopt_maxiter')
                sys.exit(1)
            else:
                if self.filtering==False:
                    print('ERROR: "Terminated (singular KKT matrix)". Problem is infeasible with this sign combination set setting.filtering==True to filter out this and any other incidences.')
                    sys.exit(1)
                if self.filtering==True:
                    pass_fail=[]
                    if self.all_output==True:
                        warnings.warn('"Terminated (singular KKT matrix)". Problem infeasable with the following sign combination, therefore sign combination will be excluded when identifying the best solution.',stacklevel=2)


        obj_func=np.sum((self.y-y)**2)
        parameters=np.array(parameters)

        return parameters, obj_func, pass_fail
