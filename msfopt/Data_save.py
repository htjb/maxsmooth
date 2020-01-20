import os
import numpy as np

class save(object):
    def __init__(self,base_dir,params,obj_func,signs,N,fit_type):
        self.base_dir=base_dir
        self.params=params
        self.obj_func=obj_func
        self.signs=signs
        self.N=N
        self.fit_type=fit_type

        if not os.path.exists( self.base_dir+'Output_Parameters/'):
            os.mkdir( self.base_dir+'Output_Parameters/')

        f=open(self.base_dir+'Output_Parameters/'+str(self.N)+'_'+self.fit_type+'.txt','a')
        f.write(str(np.array(self.params).T)+'\n')
        f.close()

        if not os.path.exists( self.base_dir+'Output_Signs/'):
            os.mkdir( self.base_dir+'Output_Signs/')

        f=open(self.base_dir+'Output_Signs/'+str(self.N)+'_'+self.fit_type+'.txt','a')
        f.write(str(self.signs)+'\n')
        f.close()

        if not os.path.exists( self.base_dir+'Output_Evaluation/'):
            os.mkdir( self.base_dir+'Output_Evaluation/')

        f=open(self.base_dir+'Output_Evaluation/'+str(self.N)+'_'+self.fit_type+'.txt','a')
        f.write(str(self.obj_func)+'\n')
        f.close()

class save_optimum(object):
    def __init__(self,base_dir,time,N,best_signs,best_obj_func,best_params,fit_type,model_type,ifp,ifp_list,best_pass_fail):
        self.base_dir=base_dir
        self.time=time
        self.N=N
        self.best_signs=best_signs
        self.best_obj_func=best_obj_func
        self.best_params=best_params
        self.fit_type=fit_type
        self.model_type=model_type
        self.ifp=ifp
        self.ifp_list=ifp_list
        self.best_pass_fail=best_pass_fail

        f=open(self.base_dir + 'Optimal_Results_'+self.fit_type+'_'+str(N)+'.txt','w')
        f.write('Time:\n')
        f.write(str(self.time)+'\n')
        f.write('Polynomial Order:\n')
        f.write(str(self.N)+'\n')
        f.write('Number of Derivatives:\n')
        f.write(str(self.N-2)+'\n')
        f.write('Signs:\n')
        f.write(str(self.best_signs)+'\n')
        f.write('Objective Function Value:\n')
        f.write(str(self.best_obj_func)+'\n')
        f.write('Parameters:\n')
        f.write(str(self.best_params.T)+'\n')
        f.write('Method:\n')
        f.write(self.fit_type+'\n')
        f.write('Model:\n')
        f.write(self.model_type+'\n')
        f.write('Inflection Points?:\n')
        f.write(str(self.ifp)+'\n')
        if self.ifp==True:
            f.write('Inflection Point Derivatives:\n')
            f.write(str(self.ifp_list)+'\n')
            f.write('Inflection Points Used? (0 signifies Yes):\n')
            f.write(str(self.best_pass_fail)+'\n')
        f.close()
