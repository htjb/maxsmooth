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

        if os.path.isfile( self.base_dir+'Output_Parameters/'):
            os.remove( self.base_dir+'Output_Parameters/')

        if not os.path.exists( self.base_dir+'Output_Parameters/'):
            os.mkdir( self.base_dir+'Output_Parameters/')

        with open(self.base_dir+'Output_Parameters/'+str(self.N)+'_'+self.fit_type+'.txt','a') as f:
            f.write('\n')
            np.savetxt(f,np.array(self.params).T)
            f.close()

        if os.path.isfile( self.base_dir+'Output_Signs/'):
            os.remove( self.base_dir+'Output_Signs/')

        if not os.path.exists( self.base_dir+'Output_Signs/'):
            os.mkdir( self.base_dir+'Output_Signs/')

        with open(self.base_dir+'Output_Signs/'+str(self.N)+'_'+self.fit_type+'.txt','a') as f:
            f.write('\n')
            np.savetxt(f,np.array(self.signs))
            f.close()

        if os.path.isfile( self.base_dir+'Output_Evaluation/'):
            os.remove( self.base_dir+'Output_Evaluation/')

        if not os.path.exists( self.base_dir+'Output_Evaluation/'):
            os.mkdir( self.base_dir+'Output_Evaluation/')

        with open(self.base_dir+'Output_Evaluation/'+str(self.N)+'_'+self.fit_type+'.txt','a') as f:
            #f.write('\n')
            np.savetxt(f,np.array([self.obj_func]))
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
        np.savetxt(f,np.array([self.time]))
        f.write('Polynomial Order:\n')
        np.savetxt(f,np.array([self.N]))
        f.write('Number of Derivatives:\n')
        np.savetxt(f,np.array([self.N-2]))
        f.write('Signs:\n')
        np.savetxt(f,self.best_signs)
        f.write('Objective Function Value:\n')
        np.savetxt(f,np.array([self.best_obj_func]))
        f.write('Parameters:\n')
        np.savetxt(f,self.best_params)
        f.write('Method:\n')
        f.write(self.fit_type+'\n')
        f.write('Model:\n')
        f.write(self.model_type+'\n')
        f.write('Inflection Points?:\n')
        f.write(str(self.ifp)+'\n')
        if self.ifp==True:
            f.write('Inflection Point Derivatives:\n')
            np.savetxt(f,self.ifp_list)
            f.write('Inflection Points Used? (0 signifies Yes):\n')
            np.savetxt(f,self.best_pass_fail)
        f.close()
