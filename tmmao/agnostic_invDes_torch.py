import torch
import numpy as np
from copy import deepcopy
from random import random, uniform, randint, choice
from scipy.optimize import minimize
from agnostic_linear_adjoint_torch import agnostic_linear_adjoint
from datetime import datetime
import warnings

class termOpt(Exception):
    """Nothing to see here. Move along.

    Since I have very little control of scipy.minimize via the api, I had to get creative to allow for dynamic scheduling/parameter mutation during optimization. This is a special
    exception defined for that purpose. You as a user should never see it raised.
    """
    pass

class agnostic_invDes(agnostic_linear_adjoint):
    """Runs an optimization with PyTorch tensor support and device placement

    Subclasses:
        agnostic_linear_adjoint: Computes the adjoint fields and gradients.
    """
    def __init__(self, costFunction, costFunction_gradPhi, costFunction_gradE, globalBoundaries, transferMatrices, transferMatrices_gradPhi, scheduler='', paramPruner='', device=None):
        """Initializes instance of agnostic_invDes with device support"""
        self.device = device if device is not None else torch.device('cpu')

        # Initialize physics package
        self.physics = None  # Will be set by updatePhysicsFunctions

        # Material parameter functions - get from physics package if available
        self.mat1 = lambda x: None
        self.mat2 = lambda x: None
        self.param0 = lambda x: None
        self.paramN = lambda x: None

        # Initialize core functions
        self.costFunction = costFunction
        self.costFunction_gradPhi = costFunction_gradPhi
        self.costFunction_gradE = costFunction_gradE
        self.globalBoundaries = globalBoundaries
        self.transferMatrices = transferMatrices
        self.transferMatrices_gradPhi = transferMatrices_gradPhi
        self.scheduler = scheduler if scheduler != '' else lambda x: 1.0
        self.paramPruner = paramPruner
        self.cfFactor = lambda x: 1.0  # Initialize cfFactor with default value
        self.customInput = lambda x: None  # Initialize customInput with default value

        # Initialize with PyTorch tensors on specified device
        self.simDict = {
            'fields': [],
            'transferMatrices': [],
            'parameters': [],
            'previousCostFunction': torch.tensor(float('inf'), device=self.device),
            'iteration': 0,
            'simPoint': None,  # Initialize simPoint
            'simType': 'MOE',
            'simRange': None,
            'simResolution': None,
            'device': self.device
        }
        self.res = optimizationResults()
        self.iterations = 0
        self.evo = []
        self.L = torch.tensor(float('inf'), device=self.device)
        self.Lphys = torch.tensor(0.0, device=self.device)
        self.Lreg = torch.tensor(0.0, device=self.device)
        self.debug_verbosity = False
        self.callback = self._default_callback  # Add default callback
        super().__init__(device=self.device)

    def updatePhysicsFunctions(self, physics_package):
        """Update physics functions with device support"""
        self.physics = physics_package
        if physics_package is not None:
            # Update material parameter functions
            self.mat1 = physics_package.get_material1_params if hasattr(physics_package, 'get_material1_params') else lambda x: None
            self.mat2 = physics_package.get_material2_params if hasattr(physics_package, 'get_material2_params') else lambda x: None
            self.param0 = physics_package.get_param0 if hasattr(physics_package, 'get_param0') else lambda x: None
            self.paramN = physics_package.get_paramN if hasattr(physics_package, 'get_paramN') else lambda x: None

            # Update physics functions if available
            if hasattr(physics_package, 'costFunction'):
                self.costFunction = physics_package.costFunction
            if hasattr(physics_package, 'costFunction_gradPhi'):
                self.costFunction_gradPhi = physics_package.costFunction_gradPhi
            if hasattr(physics_package, 'costFunction_gradE'):
                self.costFunction_gradE = physics_package.costFunction_gradE
            if hasattr(physics_package, 'globalBoundaries'):
                self.globalBoundaries = physics_package.globalBoundaries
            if hasattr(physics_package, 'transferMatrices'):
                self.transferMatrices = physics_package.transferMatrices
            if hasattr(physics_package, 'transferMatrices_gradPhi'):
                self.transferMatrices_gradPhi = physics_package.transferMatrices_gradPhi

    def _ensure_tensor(self, x):
        """Ensure input is a tensor on the correct device"""
        if isinstance(x, (list, tuple)):
            # Handle lists/tuples of matrices
            return [self._ensure_tensor(item) for item in x]
        elif isinstance(x, dict):
            # Handle dictionaries
            return {k: self._ensure_tensor(v) for k, v in x.items()}
        elif not torch.is_tensor(x):
            if isinstance(x, (int, float, complex)):
                dtype = torch.complex64 if isinstance(x, complex) else torch.float32
            else:
                dtype = torch.float32
            x = torch.tensor(x, dtype=dtype, device=self.device)
        elif x.device != self.device:
            x = x.to(device=self.device)
        return x

    def simulate(self, simDict):
        """Simulate with proper device and parameter handling"""
        try:
            # Extract parameters
            parameters = simDict.get('parameters')
            if not torch.is_tensor(parameters):
                parameters = torch.tensor(parameters, dtype=torch.float32, device=self.device)
            elif parameters.device != self.device:
                parameters = parameters.to(self.device)

            # Prepare simulation parameters
            sim_params = {
                'physics': simDict.get('physics', 'optics'),
                'callPoint': simDict.get('simPoint'),
                'wavelength': simDict.get('simPoint'),
                'simType': simDict.get('simType', 'MOE'),
                'simRange': simDict.get('simRange'),
                'simResolution': simDict.get('simResolution'),
                'third_vars': simDict.get('third_vars', [{'polarization': 's', 'incidentAngle': 0.0}]),
                'device': self.device
            }

            # Get material parameters
            mat1_params = self.mat1(sim_params)
            mat2_params = self.mat2(sim_params)
            param0_params = self.param0(sim_params)
            paramN_params = self.paramN(sim_params)

            # Calculate fields
            fields = self.physics.transferMatrices(
                parameters=parameters,
                mat1=mat1_params,
                mat2=mat2_params,
                param0=param0_params,
                paramN=paramN_params,
                sim_params=sim_params
            )

            return fields

        except Exception as ex:
            print(f"Error in simulate: {str(ex)}")
            print(f"simDict: {simDict}")
            print(f"parameters shape: {parameters.shape if torch.is_tensor(parameters) else None}")
            raise

    def globalBoundaries(self, simDict):
        """Calculate global boundary conditions with proper device handling"""
        try:
            # Extract parameters
            sim_params = {
                'physics': simDict.get('physics', 'optics'),
                'callPoint': simDict.get('simPoint'),
                'wavelength': simDict.get('simPoint'),
                'simType': simDict.get('simType', 'MOE'),
                'simRange': simDict.get('simRange'),
                'simResolution': simDict.get('simResolution'),
                'third_vars': simDict.get('third_vars', [{'polarization': 's', 'incidentAngle': 0.0}]),
                'device': self.device
            }

            # Get material parameters
            paramN_params = self.paramN(sim_params)

            # Calculate global boundary conditions
            global_bcs = self.physics.global_bcs(
                sim_params=sim_params,
                paramN=paramN_params
            )

            # Convert to list if it's a tensor
            if torch.is_tensor(global_bcs):
                global_bcs = global_bcs.tolist()

            return global_bcs

        except Exception as ex:
            print(f"Error in globalBoundaries: {str(ex)}")
            print(f"simDict: {simDict}")
            print(f"sim_params: {sim_params}")
            raise

    def transferMatrices(self, simDict):
        """Get transfer matrices with proper device and tensor handling"""
        try:
            # Get parameters and ensure they're on the correct device
            x = simDict.get('parameters')
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            elif x.device != self.device:
                x = x.to(self.device)

            # Prepare simulation parameters
            sim_params = {
                'physics': simDict.get('physics', 'optics'),
                'simPoint': simDict.get('simPoint'),  # Use simPoint consistently
                'wavelength': simDict.get('simPoint'),  # Use simPoint for wavelength
                'simType': simDict.get('simType', 'MOE'),
                'simRange': simDict.get('simRange'),
                'simResolution': simDict.get('simResolution'),
                'third_vars': simDict.get('third_vars', [{'polarization': 's', 'incidentAngle': 0.0}]),
                'device': self.device
            }

            # Get material parameters
            mat1_params = self.mat1(sim_params)
            mat2_params = self.mat2(sim_params)
            param0_params = self.param0(sim_params)
            paramN_params = self.paramN(sim_params)

            # Calculate transfer matrices
            tms = self.physics.transferMatrices(
                parameters=x,
                mat1=mat1_params,
                mat2=mat2_params,
                param0=param0_params,
                paramN=paramN_params,
                sim_params=sim_params
            )

            return tms

        except Exception as ex:
            print(f"Error in transferMatrices: {str(ex)}")
            print(f"Parameters shape: {x.shape if torch.is_tensor(x) else None}")
            print(f"Device: {x.device if torch.is_tensor(x) else None}")
            raise

    def _costFunc_wrapper(self, x):
        """Wrapper for the cost function to interface with scipy.optimize"""
        try:
            # Convert input to tensor and ensure it requires gradients
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True)

            # Create a function that takes a tensor and returns a scalar
            def tensor_cost_fn(params):
                # Update simulation dictionary with the current parameters
                self.simDict['parameters'] = params

                # Calculate cost using physics
                cost = self.physics.costFunc(
                    simPoints=[self.simDict['simPoint']],
                    callPoints=[self.simDict['simPoint']],
                    third_vars=self.simDict.get('third_vars', [{'polarization': 's', 'incidentAngle': 0.0}]),
                    cur_x=params,
                    cur_y=None,
                    all_mat1params=[self.mat1({'callPoint': cp, 'device': self.device}) for cp in [self.simDict['simPoint']]],
                    all_mat2params=[self.mat2({'callPoint': cp, 'device': self.device}) for cp in [self.simDict['simPoint']]],
                    all_param0s=[self.param0({'callPoint': cp, 'device': self.device}) for cp in [self.simDict['simPoint']]],
                    all_paramNs=[self.paramN({'callPoint': cp, 'device': self.device}) for cp in [self.simDict['simPoint']]],
                    all_fields=self.simDict.get('fields', []),
                    all_tms=self.simDict.get('tms', []),
                    all_global_bcs=self.simDict.get('global_bcs', []),
                    all_cf_factors=[1.0],
                    all_scheduler_factors=[1.0],
                    all_custom_input=None
                )

                # Handle both tensor and float returns
                if not torch.is_tensor(cost):
                    cost = torch.tensor(cost, dtype=torch.float32, device=self.device, requires_grad=True)

                # Ensure cost is a tensor scalar
                if cost.numel() > 1:
                    cost = cost.sum()
                return cost

            # Calculate cost value for scipy.optimize (needs float)
            cost_tensor = tensor_cost_fn(x_tensor)
            cost_value = float(cost_tensor.detach().cpu().item())

            # If gradient is needed (x is numpy array and jac is True)
            if isinstance(x, np.ndarray) and hasattr(self, '_compute_gradient') and self._compute_gradient:
                # Zero all previous gradients
                if x_tensor.grad is not None:
                    x_tensor.grad.zero_()

                # Recompute with gradient tracking
                cost_tensor = tensor_cost_fn(x_tensor)  # Reuse tensor_cost_fn for consistency

                # Compute gradient
                cost_tensor.backward()

                # Get gradient if it exists
                if x_tensor.grad is not None:
                    grad_np = x_tensor.grad.detach().cpu().numpy()
                    return cost_value, grad_np.astype(np.float64)
                else:
                    print("Warning: Gradient is None")
                    return cost_value, np.zeros_like(x)

            return cost_value

        except Exception as e:
            print(f"Error in _costFunc_wrapper: {str(e)}")
            print(f"Input shape: {x_tensor.shape if torch.is_tensor(x_tensor) else None}")
            print(f"Device: {x_tensor.device if torch.is_tensor(x_tensor) else None}")
            raise

    def _compute_gradient(self, x):
        """Computes gradient with PyTorch tensor support"""
        try:
            # Convert input to tensor and enable gradient computation
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True)

            # Update simulation dictionary
            self.simDict['parameters'] = x_tensor

            # Calculate cost with gradient tracking
            cost = self.physics.costFunc(
                simPoints=[self.simDict['simPoint']],
                callPoints=[self.simDict['simPoint']],
                third_vars=self.simDict.get('third_vars', [{'polarization': 's', 'incidentAngle': 0.0}]),
                cur_x=x_tensor,
                cur_y=None,
                all_mat1params=[self.mat1({'callPoint': cp, 'device': self.device}) for cp in [self.simDict['simPoint']]],
                all_mat2params=[self.mat2({'callPoint': cp, 'device': self.device}) for cp in [self.simDict['simPoint']]],
                all_param0s=[self.param0({'callPoint': cp, 'device': self.device}) for cp in [self.simDict['simPoint']]],
                all_paramNs=[self.paramN({'callPoint': cp, 'device': self.device}) for cp in [self.simDict['simPoint']]],
                all_fields=self.simDict.get('fields', []),
                all_tms=self.simDict.get('tms', []),
                all_global_bcs=self.simDict.get('global_bcs', []),
                all_cf_factors=[1.0],
                all_scheduler_factors=[1.0],
                all_custom_input=None
            )

            # Compute gradient
            cost.backward()
            grad = x_tensor.grad

            # Convert gradient to numpy array
            return grad.detach().cpu().numpy()

        except Exception as ex:
            print(f"Error in _compute_gradient: {str(ex)}")
            print(f"Input shape: {x_tensor.shape if torch.is_tensor(x_tensor) else None}")
            print(f"Device: {x_tensor.device if torch.is_tensor(x_tensor) else None}")
            raise

    def _default_callback(self, xk):
        """Default callback function for optimization tracking"""
        self.iterations += 1
        if self.debug_verbosity:
            print(f"Iteration {self.iterations}: Cost = {self.L.item()}")
        # Store evolution data
        self.res.evo.append({
            'iteration': self.iterations,
            'cost': self.L.item(),
            'parameters': xk.copy(),
            'timestamp': datetime.now()
        })
        # Check if scheduler wants to terminate
        if self.scheduler != '':
            if self.scheduler(self):
                raise termOpt

    def optimize(self, x0=None, minimize_kwargs=None):
        """
        Optimize the parameters using scipy.optimize with proper device handling
        Args:
            x0: Initial parameters (tensor or array-like)
            minimize_kwargs: Dictionary of arguments to pass to scipy.minimize
        """
        try:
            if x0 is None:
                x0 = self.simDict['parameters']

            # Convert x0 to tensor if it's not already
            if not torch.is_tensor(x0):
                x0 = torch.tensor(x0, dtype=torch.float32, device=self.device)
            elif x0.device != self.device:
                x0 = x0.to(self.device)

            # Set up default optimization options if not provided
            if minimize_kwargs is None:
                minimize_kwargs = {
                    'method': 'L-BFGS-B',
                    'jac': True,  # Use gradient information
                    'options': {
                        'maxiter': 1000,
                        'maxfun': 15000,
                        'ftol': 1e-12,
                        'gtol': 1e-12,
                        'disp': True
                    }
                }

            # Set callback if not provided
            if 'callback' not in minimize_kwargs:
                minimize_kwargs['callback'] = self._default_callback

            # Ensure jac is True for gradient computation
            minimize_kwargs['jac'] = True

            # Set flag for gradient computation
            self._compute_gradient = True

            # Run optimization
            result = minimize(self._costFunc_wrapper, x0.cpu().numpy(), **minimize_kwargs)

            # Clean up
            self._compute_gradient = False

            # Create optimization results object
            opt_results = optimizationResults()
            opt_results.x = torch.tensor(result.x, device=self.device)
            opt_results.success = result.success
            opt_results.message = result.message
            opt_results.fun = result.fun
            opt_results.nfev = result.nfev
            opt_results.nit = result.nit
            if hasattr(result, 'njev'):
                opt_results.njev = result.njev
            if hasattr(result, 'nhev'):
                opt_results.nhev = result.nhev

            return opt_results

        except Exception as e:
            print(f"Error in optimize: {str(e)}")
            print(f"x0 shape: {x0.shape if torch.is_tensor(x0) else None}")
            print(f"Device: {x0.device if torch.is_tensor(x0) else None}")
            raise

class optimizationResults:
    """Houses optimization results with PyTorch tensor support"""
    def __init__(self):
        self.evo = []
        self.success = False
        self.message = ''
        self.nfev = 0
        self.nit = 0

    def __str__(self):
        return self.prep_display()

    def __repr__(self):
        return self.__str__()

    def prep_display(self):
        """Prepares optimization results for display"""
        out = f"Optimization completed with {self.nit} iterations\n"
        out += f"Success: {self.success}\n"
        out += f"Message: {self.message}\n"
        out += f"Number of function evaluations: {self.nfev}"
        return out
