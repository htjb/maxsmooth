import numpy as np
import os


class save(object):
    def __init__(self, base_dir, params, chi_squared, signs, N, fit_type):
        self.base_dir = base_dir
        self.params = params
        self.chi_squared = chi_squared
        self.signs = [signs]
        self.N = N
        self.fit_type = fit_type

        if not os.path.exists(self.base_dir+'Output_Parameters/'):
            os.mkdir(self.base_dir+'Output_Parameters/')

        with open(
                self.base_dir+'Output_Parameters/'+str(self.N) +
                '_'+self.fit_type+'.txt', 'a') as f:
            np.savetxt(f, np.array(self.params).T)
            f.close()

        if not os.path.exists(self.base_dir+'Output_Signs/'):
            os.mkdir(self.base_dir+'Output_Signs/')

        with open(
                self.base_dir+'Output_Signs/'+str(self.N) +
                '_'+self.fit_type+'.txt', 'a') as f:
            np.savetxt(f, self.signs)
            f.close()

        if not os.path.exists(self.base_dir+'Output_Evaluation/'):
            os.mkdir(self.base_dir+'Output_Evaluation/')

        with open(
                self.base_dir+'Output_Evaluation/'+str(self.N) +
                '_'+self.fit_type+'.txt', 'a') as f:
            np.savetxt(f, np.array([self.chi_squared]))
            f.close()


class save_optimum(object):
    def __init__(
                self, base_dir, time, N, Optimum_signs, Optimum_chi_squared,
                Optimum_params, fit_type, model_type, zero_crossings,
                Optimum_zc_dict, constraints):
        self.base_dir = base_dir
        self.time = time
        self.N = N
        self.Optimum_signs = Optimum_signs
        self.Optimum_chi_squared = Optimum_chi_squared
        self.Optimum_params = Optimum_params
        self.fit_type = fit_type
        self.model_type = model_type
        self.zero_crossings = zero_crossings
        self.Optimum_zc_dict = Optimum_zc_dict
        self.constraints = constraints

        f = open(
                self.base_dir + 'Optimal_Results_'+self.fit_type +
                '_'+str(N)+'.txt', 'w')
        f.write('Time:\n')
        np.savetxt(f, np.array([self.time]))
        f.write('Polynomial Order:\n')
        np.savetxt(f, np.array([self.N]))
        f.write('Number of Derivatives:\n')
        if self.zero_crossings is None:
            np.savetxt(f, np.array([self.N-self.constraints]))
        else:
            np.savetxt(f, np.array(
                [self.N-self.constraints-len(self.zero_crossings)]))
        f.write('Signs:\n')
        np.savetxt(f, self.Optimum_signs)
        f.write('Objective Function Value:\n')
        np.savetxt(f, np.array([self.Optimum_chi_squared]))
        f.write('Parameters:\n')
        np.savetxt(f, self.Optimum_params)
        f.write('Method:\n')
        f.write(self.fit_type+'\n')
        f.write('Model:\n')
        f.write(self.model_type+'\n')
        f.write('Constraints'+'\n')
        np.savetxt(f, np.array([self.constraints]))
        if self.zero_crossings is None:
            f.write('Zero crossings Used? (0 signifies Yes):\n')
            f.write(str(self.Optimum_zc_dict))
        if self.zero_crossings is not None:
            f.write('Zero Crossing Derivatives:\n')
            np.savetxt(f, self.zero_crossings)
            f.write(
                'Zero crossings Used? (0 signifies' +
                ' Yes\n in derivative order "i"):\n')
            f.write(str(self.Optimum_zc_dict))
        f.close()
