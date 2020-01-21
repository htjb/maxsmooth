import numpy as np
import pylab as pl
import sys
import warnings

warnings.simplefilter('always', UserWarning)

class derivative_class(object):
    def __init__(self,x,y,params,N,signs,mid_point,model_type,ifp):
        self.signs=signs
        self.x=x
        self.y=y
        self.N=N
        self.params=params
        self.mid_point=mid_point
        self.model_type=model_type
        self.ifp=ifp
        self.derive,self.pass_fail=self.derivatives_func()

    def derivatives_func(self):#,signs):
        m=np.arange(1,self.N+1,1)
        deriv=[]
        for j in range(len(m)-1):
            if m[j]>=2:
                dif=[]
                for i in range(self.N-m[j]):
                    if i<=(self.N-m[j]):
                        if self.model_type=='normalised_polynomial':
                            dif_m_bit=(self.y[self.mid_point]/self.x[self.mid_point])*np.math.factorial(m[j]+i)/np.math.factorial(i)*self.params[int(m[j])+i]*(self.x)**i/(self.x[self.mid_point])**(i+1)
                            dif.append(dif_m_bit)
                        if self.model_type=='polynomial':
                            dif_m_bit=np.math.factorial(m[j]+i)/np.math.factorial(i)*self.params[int(m[j])+i]*(self.x)**i
                            dif.append(dif_m_bit)
                        if self.model_type=='MSF_2017_polynomial':
                            dif_m_bit=np.math.factorial(m[j]+i)/np.math.factorial(i)*self.params[int(m[j])+i]*(self.x-self.x[self.mid_point])**i
                            dif.append(dif_m_bit)
                        if self.model_type=='logarithmic_polynomial':
                            dif_m_bit=np.math.factorial(m[j]+i)/np.math.factorial(i)*self.params[int(m[j])+i]*np.log10(self.x)**i
                            dif.append(dif_m_bit)
                dif=np.array(dif)
                derivative=dif.sum(axis=0)
                deriv.append([m[j],derivative])
        deriv=np.array(deriv)
        derive=[]
        for i in range(deriv.shape[0]):
            derive.append(deriv[i,1])
        derive=np.array(derive)
        #array_for_checking=np.array([0]*len(self.x))
        #print(derive[0,:].max()-1e-4)
        pass_fail=[] #1==pass, 0==fail
        for i in range(derive.shape[0]):
            if np.all(derive[i,:]>0) or np.all(derive[i,:]<0):
                pass_fail.append(1)
            else:
                pass_fail.append(0)

        pass_fail=np.array(pass_fail)

        #print('derive',derive)
        if np.any(pass_fail == 0):
            if self.ifp==True:
                #print('Pass or Fail',pass_fail)
                warnings.warn('WARNING: setting.ipf = True has lead to derivatives including inflection points.',stacklevel=2)
            if self.ifp==False:
                print('Pass or fail',pass_fail)
                #pl.subplot(111)
                #[pl.plot(self.x,derive[i,:]) for i in range(derive.shape[0])]
                #pl.show()
                #pl.close()
                print('ERROR: "Condition Violated" Derivatives feature crossing points.')
                sys.exit(1)

        return derive, pass_fail
