
class setting(object):
    def __init__(self):

        """
        fit_type: (Default=='qp-sign_flipping') The type of fitting routine used to fit the model.
            Accepted options:
                'qp' - Quadratic programming testing every combination of sign on the derivatives.
                    This is a quick process provided the order of the polynomial is small and the data
                    sets being fitted are also small.
                'qp-sign_flipping' - Quadratic Programming testing a sub sample of sign combinations
                    on the derivatives. The algorithm currently generates a random set of signs for
                    the N-2 derivatives. It then flips sucessive signs in the list until it calculates
                    a chi squared smaller than the previous evaluation of the objective function. For
                    example a 4th order polynomial has 2 derivatives with m>=2 which means it has
                    4 sign combinations [1,1],[-1,-1],[-1,1] and [1,-1]. On first random generation we
                    get [-1,1] with which we evaluate the objective function. We then flip the first sign
                    and evaluate again with [1,1]. If need be we then go back to the original list and
                    flip the second sign evaluating with [-1,-1]. The process repeats until the new chi
                    squared is no longer smaller than the previous evaluation.( I don't think this process
                    ever repeats more times than there are signs to flip). We then repeat the entire
                    process a set number of times to ensure we can identify the true minimum. The number
                    of repeats needed is dependent on the polynomial order. High polynomial orders require
                    a larger number of repeats to find the true minimum. Currently the number of repeats
                    is set at 2*N**2. The sucess of this method can be judged by running with the 'qp' method.

        model_type: (Default = 'normalised_polynomial') The type of model used to fit the data.
            Accepted options:
                'normalised_polynomial' - This is a polynomial of the form y=y_0*sum(p_i*(x/x_0)**i). It
                    consistently appears to give the best fit.
                'polynomial' - This is a polynomial of the form y=sum(p_i*(x)**i).
                'MSF_2017_polynomial' - This is a polynomial of the form described in section 4 of
                    doi:10.3847/1538-4357/aa69bd.
                'logarithmic_polynomial' - This is a polynomial model similar to that used with the
                    setting 'polynomial' but solved in log-space. It has the form
                    log_{10}(y)=sum(p_i*(log_{10}(x))**i). NOTE this model will not work if the y values
                    are negative, for example in the case of uncalibrated data.

        base_dir: (Default = 'Data' + '_' + self.model_type+ '/') This is the directory in which the resultant graphs of the fit, derivatives and residuals are saved. When
            testing multiple model types it is recommended to include this in the base directory
            name eg self.base_dir= 'Data_Name_' + self.model_type + '/'.

        cvxopt_maxiter: (Default=1000) The maximum number of iterations for the cvxopt quadratic
            programming routine

        filtering: (Default=True) Generally for high order N there will be combinations of sign for which
            cvxopt cannot find a solution and these terminate with the error
            "Terminated (Singular KKT Matrix)". Setting filtering to True will flag this as
            a warning and exclude these sign combinations when determining the best possible fit.
            Setting filtering to False will cause the program to crash with the error.
            Setting filtering to True will also save the parameters,objective function value and sign
            combinations from each successful run of cvxopt to the base directories in seperate folders.

        all_output: (Default=False) If set to True this will output the results of each run of cvxopt
            to the terminal.

        ifp: (Default = False) Setting equal to True allows for inflection points in the m order derivatives
            listed in ifp_derivatives.
                NOTE: The algorithm will not necessarily return derivatives
                        with inflection points if this is set to True.
                NOTE: Allowing for inflection points will increasese run time.

        ifp_list: (Default = 'None') The list of derivatives you wish to allow to have inflection points
            in(see ifp above). This should be a list of derivative orders eg. if I have a fith order
            polynomial and I wish to allow the the second derivative to have an inflection point then
            ifp_list=[2]. If I wished to allow the second and fourth derivative to have inflection points
            I would write ifp_list=[2,4]. Values in ifp_list cannot exceed N-2.

        """

        self.fit_type='qp-sign_flipping'
        self.model_type='normalised_polynomial'
        self.base_dir='Data' + '_' + self.model_type+ '/'
        self.cvxopt_maxiter = 1000 #needs explination
        self.filtering=True
        self.all_output=False
        self.ifp=False
        self.ifp_list = 'None'
