import numpy as np

class Models_class(object):
    def __init__(self,params,x,y,N,mid_point,model_type):
        self.x=x
        self.y=y
        self.N=N
        self.params=params
        self.mid_point=mid_point
        self.model_type=model_type
        self.y_sum=self.fit()

    def fit(self):

        if self.model_type=='normalised_polynomial':
            y=[]
            append_y=y.append
            for i in range(self.N):
                y_bit=self.params[i]*(self.x/self.x[self.mid_point])**i
                append_y(y_bit)
            y=np.array(y)
            y_sum=y.sum(axis=0)*self.y[self.mid_point]

        if self.model_type=='polynomial':
            y=[]
            append_y=y.append
            for i in range(self.N):
                y_bit=self.params[i]*(self.x)**i
                append_y(y_bit)
            y=np.array(y)
            y_sum=y.sum(axis=0)

        if self.model_type=='MSF_2017_polynomial':
            y=[]
            append_y=y.append
            for i in range(self.N):
                y_bit=self.params[i]*(self.x-self.x[self.mid_point])**i
                append_y(y_bit)
            y=np.array(y)
            y_sum=y.sum(axis=0)

        if self.model_type=='logarithmic_polynomial':
            logy=[]
            append_logy=logy.append
            for i in range(self.N):
                logy_bit=self.params[i]*(np.log10(self.x))**i
                append_logy(logy_bit)
            logy=np.array(logy)
            y_sum=10**logy.sum(axis=0)
            
        return y_sum
