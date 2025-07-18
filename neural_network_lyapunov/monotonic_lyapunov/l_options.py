import torch
import numpy as np


class LOptions:
    """
    When search for the Lyapunov function, we use the 1-norm of |R*(x-x*)|₁.
    This class specificies the options to search for R.
    """
    def __init__(self):
        pass

    def set_variable_value(self, L_val: np.ndarray):
        pass

    def L(self) -> torch.Tensor:
        pass

    def variables(self) -> list:
        pass

    @property
    def fixed_L(self) -> bool:
        pass

    def extract_params(self):
        return dict()



class FixedLOptions(LOptions):
    """
    When search for the Lyapunov function, we use the 1-norm of |R*(x-x*)|₁.
    This class specificies that R is fixed.
    R should be fixed to a full column rank matrix.
    """
    def __init__(self, L: torch.Tensor):
        super(FixedLOptions, self).__init__()
        assert (isinstance(L, torch.Tensor))
        self._L = L

    def L(self):
        return self._L

    def variables(self):
        return []

    def __str__(self):
        return f"Fixed R to \n {self._L}"

    @property
    def fixed_L(self):
        return True


class SearchLfreeOptions(LOptions):
    def __init__(self, L_size: tuple):
        super(SearchLfreeOptions, self).__init__()
        assert (isinstance(L_size, tuple))
        assert (len(L_size) == 1)
        self.L_size = L_size
        self._variables = torch.empty(L_size,
                                      dtype=torch.float64,
                                      requires_grad=True)

    def set_variable_value(self, L_val: np.ndarray):
        assert (isinstance(L_val, np.ndarray))
        assert (L_val.shape == self.L_size)
        self._variables = torch.from_numpy(L_val)
        self._variables.requires_grad = True

    def set_variable_value_directly(self, variable_val: np.ndarray):
        assert (isinstance(variable_val, np.ndarray))
        assert (variable_val.shape == self.L_size)
        self._variables = torch.from_numpy(variable_val)
        self._variables.requires_grad = True

    def L(self):
        return self._variables

    def variables(self) -> list:
        return [self._variables]

    def __str__(self):
        return f"Search L freely. Size {self.L_size}"

    @property
    def fixed_L(self):
        return False

    def extract_params(self):
        return {"L_size": self.L_size}
