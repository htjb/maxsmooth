import numpy as np
import pylab as pl
from maxsmooth.msf_qp import max_fit_qp
import time
from itertools import product
import os
import sys
from maxsmooth.fit_model import function_fit
from maxsmooth.derivatives import derivative_class
from maxsmooth.Data_save import save,save_optimum

class msf_qp_class(object):
    def __init__(self,x,y,N,setting):
        self.x=x
        self.y=y
        self.N=N
        self.fit_type,self.base_dir,self.model_type,self.filtering,self.all_output,self.cvxopt_maxiter,self.ifp,self.ifp_list=\
            setting.fit_type,setting.base_dir,setting.model_type,setting.filtering,setting.all_output,setting.cvxopt_maxiter,setting.ifp,setting.ifp_list
        self.y_fit, self.signs_results, self.parameters, self.derivatives, self.objective,self.rms=self.fitting()

    def fitting(self):

        def signs_array(nums):
            return np.array(list(product(*((x,-x) for x in nums))))

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        def qp(x,y,N,mid_point):
            print('#############################################################')
            start=time.time()
            if not os.path.exists( self.base_dir+'MSF_Order_'+str(N)+'_'+self.fit_type+'/'):
                os.mkdir(self.base_dir+'MSF_Order_'+str(N)+'_'+self.fit_type+'/')

            signs=signs_array([1]*(N-2))

            params, objective_func, passed_failed,passed_signs=[],[],[],[]
            append_params,append_obj,append_pf,append_passed_signs=params.append,objective_func.append,passed_failed.append,passed_signs.append
            for j in range(signs.shape[0]):
                fit = max_fit_qp(x,y,N,signs[j,:],mid_point,self.model_type,self.cvxopt_maxiter,self.filtering,self.all_output,self.ifp,self.ifp_list)
                parameters, obj_func,pass_fail=fit.parameters,fit.obj_func,fit.pass_fail
                if self.all_output==True:
                    print('----------------------------------------------------')
                    print('Polynomial Order:', N)
                    print('Number of Derivatives:', N-2)
                    print('Signs :',signs[j,:])
                    print('Objective Function Value:',obj_func)
                    print('Parameters:', parameters.T)
                    print('Method:',self.fit_type)
                    print('Model:',self.model_type)
                    print('Inflection Points?:', self.ifp)
                    if self.ifp==True:
                        print('Inflection Point Derivatives:', self.ifp_list)
                        print('Inflection Points Used? (0 signifies Yes):', pass_fail)
                    print('----------------------------------------------------')

                if self.filtering==False:
                    append_params(parameters)
                    append_obj(obj_func)
                    append_pf(pass_fail)
                    append_passed_signs(signs[j,:])
                if self.filtering==True:
                    if pass_fail == []:
                        pass
                    else:
                        append_params(parameters)
                        append_obj(obj_func)
                        append_pf(pass_fail)
                        append_passed_signs(signs[j,:])
                        save(self.base_dir,parameters,obj_func,signs[j,:],N,self.fit_type)
            params,objective_func,passed_failed,passed_signs=np.array(params),np.array(objective_func),np.array(passed_failed),np.array(passed_signs)
            for f in range(len(objective_func)):
                if objective_func[f]==objective_func.min():
                    best_params=(params[f,:])
                    best_sign_combination=passed_signs[f,:]
                    best_objective_func=objective_func.min()
            #np.save(self.base_dir+'Objective_func_'+self.fit_type+'_'+str(N)+'.npy',best_objective_func)
            y_fit=function_fit(best_params,x,y,N,mid_point,self.model_type).y_sum
            derivatives=derivative_class(x,y,best_params,N,best_sign_combination,mid_point,self.model_type,self.ifp)
            derive,best_pass_fail=derivatives.derive,derivatives.pass_fail

            end=time.time()
            #np.save(self.base_dir+'Time_'+self.fit_type+'_'+str(N)+'.npy',end-start)
            print('#############################################################')
            print('----------------------OPTIMUM RESULT-------------------------')
            print('Time:',end-start)
            print('Polynomial Order:', N)
            print('Number of Derivatives:', N-2)
            print('Signs :',best_sign_combination)
            print('Objective Function Value:',best_objective_func)
            print('Parameters:', best_params.T)
            print('Method:',self.fit_type)
            print('Model:',self.model_type)
            print('Inflection Points?:', self.ifp)
            if self.ifp==True:
                print('Inflection Point Derivatives:', self.ifp_list)
                print('Inflection Points Used? (0 signifies Yes):', best_pass_fail)
            print('-------------------------------------------------------------')
            print('#############################################################')

            save_optimum(self.base_dir,end-start,N,best_sign_combination,best_objective_func,best_params,self.fit_type,self.model_type,self.ifp,self.ifp_list,best_pass_fail)

            return y_fit,derive,best_objective_func,best_params,best_sign_combination

        def qp_sign_flipping(x,y,N,mid_point):
            print('#############################################################')
            start=time.time()
            if not os.path.exists( self.base_dir+'MSF_Order_'+str(N)+'_'+self.fit_type+'/'):
                os.mkdir(self.base_dir+'MSF_Order_'+str(N)+'_'+self.fit_type+'/')

            runs=np.arange(0,2*N*N,1)
            best_params,best_obj_func,best_signs=[],[],[]
            for k in range(len(runs)):
                if self.filtering == True:
                    pass_fail=[]
                    while pass_fail==[]:
                        signs=[]
                        for l in range(N-2):
                            sign=np.random.randint(0,2)
                            if sign == 0:
                                sign =-1
                            signs.append(sign)
                        signs=np.array(signs)
                        fit = max_fit_qp(x,y,N,signs,mid_point,self.model_type,self.cvxopt_maxiter,self.filtering,self.all_output,self.ifp,self.ifp_list)
                        obj_func_old,params,pass_fail=fit.obj_func,fit.parameters,fit.pass_fail
                        if self.all_output==True:
                            print('----------------------------------------------------')
                            print('Polynomial Order:', N)
                            print('Number of Derivatives:', N-2)
                            print('Signs :',signs)
                            print('Objective Function Value:',obj_func_old)
                            print('Parameters:', params.T)
                            print('Method:',self.fit_type)
                            print('Model:',self.model_type)
                            print('Inflection Points?:', self.ifp)
                            if self.ifp==True:
                                print('Inflection Point Derivatives:', self.ifp_list)
                                print('Inflection Points Used? (0 signifies Yes):', pass_fail)
                            print('----------------------------------------------------')
                        if pass_fail !=[]:
                            save(self.base_dir,params,obj_func_old,signs,N,self.fit_type)
                else:
                    signs=[]
                    for l in range(N-2):
                        sign=np.random.randint(0,2)
                        if sign == 0:
                            sign =-1
                        signs.append(sign)
                    signs=np.array(signs)
                    fit = max_fit_qp(x,y,N,signs,mid_point,self.model_type,self.cvxopt_maxiter,self.filtering,self.all_output,self.ifp,self.ifp_list)
                    obj_func_old,params,pass_fail=fit.obj_func,fit.parameters,fit.pass_fail
                    if self.all_output==True:
                        print('----------------------------------------------------')
                        print('Polynomial Order:', N)
                        print('Number of Derivatives:', N-2)
                        print('Signs :',signs)
                        print('Objective Function Value:',obj_func_old)
                        print('Parameters:', params.T)
                        print('Method:',self.fit_type)
                        print('Model:',self.model_type)
                        print('Inflection Points?:', self.ifp)
                        if self.ifp==True:
                            print('Inflection Point Derivatives:', self.ifp_list)
                            print('Inflection Points Used? (0 signifies Yes):', pass_fail)
                        print('----------------------------------------------------')

                obj_func_new=0
                parameters=[]
                parameters.append(params)
                objective_funcs,signs_tested=[],[]
                signs_tested.append(signs)
                h=0
                while obj_func_old>obj_func_new:
                        if h > 0:
                            if obj_func_new != 0:
                                obj_func_old=obj_func_new
                        if h < len(signs):
                            new_signs=np.empty(len(signs))
                            for m in range(len(signs)):
                                if m == h:
                                    new_signs[h] = signs[h]*-1
                                else:
                                    new_signs[m]=signs[m]
                        fit = max_fit_qp(x,y,N,new_signs,mid_point,self.model_type,self.cvxopt_maxiter,self.filtering,self.all_output,self.ifp,self.ifp_list)
                        obj_func_new,pass_fail,params=fit.obj_func,fit.pass_fail,fit.parameters
                        if self.all_output==True:
                            print('----------------------------------------------------')
                            print('Polynomial Order:', N)
                            print('Number of Derivatives:', N-2)
                            print('Signs :',new_signs)
                            print('Objective Function Value:',obj_func_new)
                            print('Parameters:', params.T)
                            print('Method:',self.fit_type)
                            print('Model:',self.model_type)
                            print('Inflection Points?:', self.ifp)
                            if self.ifp==True:
                                print('Inflection Point Derivatives:', self.ifp_list)
                                print('Inflection Points Used? (0 signifies Yes):', pass_fail)
                            print('----------------------------------------------------')
                        if self.filtering==True:
                            if fit.pass_fail==[]:
                                obj_func_new=0
                            else:
                                parameters.append(fit.parameters)
                                signs_tested.append(new_signs)
                                save(self.base_dir,fit.parameters,obj_func_new,new_signs,N,self.fit_type)
                        if self.filtering==False:
                            parameters.append(fit.parameters)
                            signs_tested.append(new_signs)
                        if h<=len(signs):
                                pass
                        else:
                            print('ERROR: Sign flipping iterations has exceeded len(signs)')
                            sys.exit(1)
                        h+=1
                best_obj_func.append(obj_func_old)
                best_params.append(parameters[-2])
                best_signs.append(signs_tested[-2])
            best_obj_func=np.array(best_obj_func)
            for j in range(len(best_obj_func)):
                if best_obj_func[j]==best_obj_func.min():
                    obj_func_result=best_obj_func[j]
                    params_result=best_params[j]
                    signs_result=best_signs[j]
            #np.save(self.base_dir+'Objective_func_'+self.fit_type+'_'+str(N)+'.npy',obj_func_result)
            y_result=function_fit(params_result,x,y,N,mid_point,self.model_type).y_sum
            derivatives=derivative_class(x,y,params_result,N,signs_result,mid_point,self.model_type,self.ifp)
            derive,best_pass_fail=derivatives.derive,derivatives.pass_fail

            end=time.time()
            #np.save(self.base_dir+'Time_'+self.fit_type+'_'+str(N)+'.npy',end-start)
            print('#############################################################')
            print('----------------------OPTIMUM RESULT-------------------------')
            print('Time:',end-start)
            print('Polynomial Order:', N)
            print('Number of Derivatives:', N-2)
            print('Signs :',signs_result)
            print('Objective Function Value:',obj_func_result)
            print('Parameters:', params_result.T)
            print('Method:',self.fit_type)
            print('Model:',self.model_type)
            print('Inflection Points?:', self.ifp)
            if self.ifp==True:
                print('Inflection Point Derivatives:', self.ifp_list)
                print('Inflection Points Used? (0 signifies Yes):', best_pass_fail)
            print('-------------------------------------------------------------')
            print('#############################################################')

            save_optimum(self.base_dir,end-start,N,signs_result,obj_func_result,params_result,self.fit_type,self.model_type,self.ifp,self.ifp_list,best_pass_fail)

            return y_result,derive,obj_func_result,params_result,signs_result

        def plotting(x,y,N,y_fit,derivatives):
            rms_array=[]
            for i in range(len(N)):
                pl.subplot(111)
                pl.plot(x,y_fit[i,:],c='r',label='Fitted MSF with N = '+ str(N[i]))
                pl.plot(x,y,label='Sky Data')
                pl.legend(loc=0)
                pl.xlabel(r'$\nu$ [Hz]')
                pl.ylabel(r'T [K]')
                pl.tight_layout()
                pl.savefig(self.base_dir + 'MSF_Order_'+str(N[i])+'_'+self.fit_type+'/Fit.pdf')
                #pl.show()
                pl.close()

                pl.subplot(111)
                pl.plot(x,y-y_fit[i,:],label='RMS=%2.2f mK' %(np.sqrt(np.sum((y-y_fit[i,:])**2)/len(y))*1e3))
                pl.fill_between(x,np.array([np.sqrt(np.sum((y-y_fit[i,:])**2)/len(y))]*len(x)),-np.array([np.sqrt(np.sum((y-y_fit[i,:])**2)/len(y))]*len(x)),color='r',alpha=0.5)
                pl.legend(loc=0)
                pl.ylabel('y-y_fit [K]')
                pl.xlabel(r'$\nu$ [Hz]')
                pl.tight_layout()
                pl.savefig(self.base_dir + 'MSF_Order_'+str(N[i])+'_'+self.fit_type+'/RMS.pdf')
                #pl.show()
                pl.close()

                rms=(np.sqrt(np.sum((y-y_fit[i,:])**2)/len(y))*1e3)
                np.save(self.base_dir + 'MSF_Order_'+str(N[i])+'_'+self.fit_type+'/RMS.npy',rms)

                pl.subplot(111)
                #for j in range(derive.shape[0]):
                [pl.plot(x,derivatives[i][j,:],label='M:'+str(j+2)) for j in range(derivatives[i].shape[0])]
                pl.legend(loc=0)
                pl.tight_layout()
                pl.xlabel(r'$\nu$ [Hz]')
                pl.ylabel('M Order Derivatives')
                pl.savefig(self.base_dir + 'MSF_Order_'+str(N[i])+'_'+self.fit_type+'/Derivatives.pdf')
                #pl.show()
                pl.close()
                rms_array.append(rms)
            rms_array=np.array(rms_array)
            return rms_array

        mid_point=len(self.x)//2
        if self.fit_type=='qp':
            y_fit,signs_result,derivatives,parameters,objective=[],[],[],[],[]
            for i in range(len(self.N)):
                y_result,derive,obj,params,signs=qp(self.x,self.y,self.N[i],mid_point)
                y_fit.append(y_result)
                signs_result.append(signs)
                derivatives.append(derive)
                parameters.append(params)
                objective.append(obj)
            y_fit,signs_result,derivatives,parameters,objective=np.array(y_fit),np.array(signs_result),np.array(derivatives),np.array(parameters),np.array(objective)
        if self.fit_type=='qp-sign_flipping':
            y_fit,signs_result,derivatives,parameters,objective=[],[],[],[],[]
            for i in range(len(self.N)):
                if self.N[i]<=10:
                    y_result,derive,obj,params,signs=qp(self.x,self.y,self.N[i],mid_point)
                else:
                    y_result,derive,obj,params,signs=qp_sign_flipping(self.x,self.y,self.N[i],mid_point)
                y_fit.append(y_result)
                derivatives.append(derive)
                signs_result.append(signs)
                parameters.append(params)
                objective.append(obj)
            y_fit,signs_result,derivatives,objective=np.array(y_fit),np.array(signs_result),np.array(derivatives),np.array(objective)
        rms=plotting(self.x,self.y,self.N,y_fit,derivatives)

        return y_fit, signs_result, parameters, derivatives, objective,rms
