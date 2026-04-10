#!/usr/bin/env python3
import os
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
from utils import DotDict

__all__ = [
    "lasso_glm_params",
]

class lasso_glm_params(DotDict):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `lasso_glm` on initialization.
    """
    # Internal macro parameter.
    _precision = "float32"

    def __init__(self, dataset="eeg"):
        """
        Initialize `lasso_glm_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(lasso_glm_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = lasso_glm_params._gen_model_params()
        # -- Train parameters
        self.train = lasso_glm_params._gen_train_params(dataset)

    """
    generate funcs
    """
    ## def _gen_model_* funcs
    # def _gen_model_params func
    @staticmethod
    def _gen_model_params():
        """
        Generate model parameters.
        """
        # Initialize model_params.
        model_params = DotDict()

        ## -- Normal parameters
        # The L1 penalty of lasso_glm.
        model_params.l1_penalty = 5e-3
        # The L2 penalty of lasso_glm.
        model_params.l2_penalty = 0.
        # The mode of accuracy calculation.
        model_params.acc_mode = ["default"][0]

        # Return the final `model_params`.
        return model_params

    ## def _gen_train_* funcs
    # def _gen_train_params func
    @staticmethod
    def _gen_train_params(dataset):
        """
        Generate train parameters.
        """
        # Initialize train parameters.
        train_params = DotDict()

        ## -- Normal parameters
        # The type of dataset.
        train_params.dataset = dataset
        # The ratio of train dataset. The rest is test dataset.
        train_params.train_ratio = 0.8
        # The number of training runs for each expriments.
        train_params.n_runs = 30

        # Return the final `train_params`.
        return train_params

if __name__ == "__main__":
    # Instantiate lasso_glm_params.
    lasso_glm_params_inst = lasso_glm_params(dataset="eeg")

