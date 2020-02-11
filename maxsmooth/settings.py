
"""
The Settings class is used to define options that are passed to maxsmooth.
It should be called by the user before the function ``smooth`` by,

.. code:: bash

    from maxsmooth.settings import setting
    setting = setting()

and changes to the settings can be made before a call to ``smooth``
like so,

.. code:: bash

    setting.model_type = 'polynomial'
"""

class setting(object):
    r"""
    **Attributes**

    **fit_type:** (Default=='qp-sign_flipping')

    The type of fitting routine used to fit the model. There are two options
    designed to explore the sign space of the function.

    *Accepted options:*

    'qp' - Quadratic programming testing every combination of sign
        on the derivatives. This is a quick process provided the
        order of the polynomial is small.

    'qp-sign_flipping' - Needs an explination...


    **model_type:** (Default = 'log_MSF_polynomial')

    The type of model used to fit the data. There is a built in library of
    maximally smooth functions that can be called by the user.

    *Accepted options:*

        'normalised_polynomial' - This is a polynomial of the form,

            .. math::
                {y=y_0 \sum (p_i\bigg(\frac{x}{x_0}\bigg)^i)}.

        'polynomial' - This is a polynomial of the form,
            .. math::

                {y=sum(p_i(x)^i)}.

        'MSF_2017_polynomial' - This is a polynomial of the form
            described in section 4 of
            `Sathyanarayana Rao, 2017
            <https://iopscience.iop.org/article/10.3847/1538-
            4357/aa69bd/meta>`__

        'log_MSF_polynomial' - ...

    **base_dir:** (Default = 'Fitted_Output')
        This is the directory in which the output of the program is saved. If
        the directory does not exist the software will create it in the working
        as long as the files that preceed it also exist. When testing multiple
        model types it is recommended to include this in the base directory
        name eg `self.base_dir= 'Data_Name_' + self.model_type + '/'.`

    **cvxopt_maxiter:** (Default=1000)
        The maximum number of iterations for the cvxopt quadratic
        programming routine. If cvxopt reaches maxiter the fitting routine
        will exit with an error recommending this be increased.

    **all_output:** (Default=False)
        If set to True this will output the results of each run of cvxopt
        to the terminal.

    **ifp:** (Default = False)
        Setting equal to True allows for inflection points in the m order
        derivatives listed in ifp_derivatives.
        *NOTE:* The algorithm will not necessarily return derivatives
                with inflection points if this is set to True.
        *NOTE:* Allowing for inflection points will increasese run time.

    **ifp_list:** (Default = 'None')
        The list of derivatives you wish to allow
        to have inflection points in(see ifp above). This should be a list
        of derivative orders eg. if I have a fith order
        polynomial and I wish to allow the the second derivative to have
        an inflection point then ifp_list=[2]. If I wished to allow the
        second and fourth derivative to have inflection points
        I would write ifp_list=[2,4]. Values in ifp_list cannot exceed the
        number of possible derivatives and cannot equal 1.

    **data_save:** (Default = True)
        Setting data_save to True will save sample
        graphs of the derivatives, fit and residuals. The inputs to
        produce these graphs are all outputted from the *smooth* function
        and they can be reproduced with more specific axis labels/units in
        the users code. If filtering is also set to True, which it is by
        default, then parameters, objective function values and sign
        combinations from each successful run of cvxopt will be saved
        to the base directories in seperate folders. The condition on
        filtering prevents saving data from runs of cvxopt that did not
        find solutions and terminated with a singular KKT matrix.

    **warnings:** (Default = False)
        Setting to False will prevent warnings showing in the terminal.
        Setting to True will show all warnings.

    """
    def __init__(self):

        self.fit_type = 'qp-sign_flipping'
        self.model_type = 'log_MSF_polynomial'
        self.base_dir = 'Fitted_Output/'
        self.cvxopt_maxiter = 1000
        self.all_output = False
        self.ifp = False
        self.ifp_list = 'None'
        self.data_save = False
        self.warnings = False
        self.cvxopt_feastol = 'Default' # needs documenting
