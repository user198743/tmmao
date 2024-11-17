import torch
import numpy as np
from copy import deepcopy
from agnostic_invDes_torch import agnostic_invDes
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

class agnostic_director:
    """Director class for inverse design optimization with PyTorch support"""

    def __init__(self, device=None):
        """Initialize with optional device specification"""
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.aid = agnostic_invDes('', '', '', '', '', '', device=self.device)
        self.physics_package = None
        self.mat1Call = None
        self.mat2Call = None
        self.param0Call = None
        self.paramNCall = None
        self.customInputCall = None
        self.simType = None
        self.num_layers = None
        self.initial_layer_thickness = None
        self.initial_ys = None
        self.y_bounds = None
        self.x_bounds = None
        self.num_intervals = None
        self.substrates = None
        self.superstrates = None

    def _ensure_tensor(self, x, dtype=torch.float32):
        """Ensures input is a PyTorch tensor on the correct device"""
        if torch.is_tensor(x):
            return x.to(self.device)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def set_physics(self, physics_package, mat1Call, mat2Call, param0Call, paramNCall, customInputCall):
        """Set physics functions with device support"""
        self.physics_package = physics_package
        self.mat1Call = mat1Call
        self.mat2Call = mat2Call
        self.param0Call = param0Call
        self.paramNCall = paramNCall
        self.customInputCall = customInputCall

        # Set material functions in physics package
        self.physics_package.set_materials(
            mat1=mat1Call,
            mat2=mat2Call,
            param0=param0Call,
            paramN=paramNCall
        )

        # Update physics functions in inverse design object with new signature
        self.aid.updatePhysicsFunctions(physics_package)

    def set_optimizerFunctions(self, cfFactorCall, schedulerCall, dynamic_scheduling=True, paramPrunerCall=None):
        """Set optimizer functions with device awareness"""
        self.physics_package.cfFactorCall = cfFactorCall
        if dynamic_scheduling:
            self.aid.scheduler = schedulerCall
        if paramPrunerCall is not None:
            self.aid.paramPruner = paramPrunerCall

    def set_structure(self, simType, num_layers, initial_layer_thickness, initial_ys, y_bounds, x_bounds, num_intervals, substrates, superstrates):
        """Set structure parameters with device support"""
        self.simType = simType
        self.num_layers = num_layers
        self.initial_layer_thickness = self._ensure_tensor(initial_layer_thickness)
        self.initial_ys = self._ensure_tensor(initial_ys)
        self.y_bounds = y_bounds
        self.x_bounds = x_bounds
        self.num_intervals = num_intervals
        self.substrates = substrates
        self.superstrates = superstrates

        # Create and store the structure object
        class Structure:
            def __init__(self, num_layers, initial_ys, y_bounds, x_bounds, num_intervals):
                self.num_layers = num_layers
                self.params = initial_ys
                self.y_bounds = y_bounds
                self.x_bounds = x_bounds
                self.num_intervals = num_intervals
                # Create bounds list for optimization
                self.bounds = [(y_bounds[0], y_bounds[1]) for _ in range(num_layers)]

            def get_num_params(self):
                return self.num_layers

            def get_params(self):
                return self.params

        # Initialize structure with proper parameters
        self.structure = Structure(
            num_layers=num_layers,
            initial_ys=self.initial_ys,
            y_bounds=y_bounds,
            x_bounds=x_bounds,
            num_intervals=num_intervals
        )

        # Update physics package parameters
        if hasattr(self, 'physics_package'):
            self.physics_package.set_structure(
                simType=self.simType,
                num_layers=num_layers,
                initial_layer_thickness=self.initial_layer_thickness,
                mat1Call=self.mat1Call,
                mat2Call=self.mat2Call,
                param0Call=self.param0Call,
                paramNCall=self.paramNCall,
                customInputCall=self.customInputCall,
                substrates=substrates,
                superstrates=superstrates,
                y_bounds=self.y_bounds,
                x_bounds=self.x_bounds,
                num_intervals=self.num_intervals
            )

    def run_optimization(self, simRange, simResolution=1, third_variables=None, ftol=1e-12, gtol=1e-12, simPoint=None, verbose=1):
        """Run optimization with device support"""
        try:
            # Ensure all inputs are on the correct device
            simRange = self._ensure_tensor(simRange)
            if simPoint is None:
                simPoint = simRange[0]
            simPoint = self._ensure_tensor(simPoint)

            # Initialize simulation dictionary with required keys
            self.simDict = {
                'physics': 'optics',  # Set physics type
                'simRange': simRange,
                'simResolution': simResolution,
                'simPoint': simPoint,
                'simType': self.simType,
                'third_vars': third_variables,
                'device': self.device  # Ensure device is included
            }

            # Initialize parameters
            x0 = self.structure.get_params()
            x0 = self._ensure_tensor(x0)

            # Set up optimization parameters
            minimize_kwargs = {
                'method': 'L-BFGS-B',
                'jac': True,
                'bounds': self.structure.bounds,
                'options': {
                    'ftol': ftol,
                    'gtol': gtol,
                    'maxiter': 1000,
                    'maxfun': 15000,
                    'disp': bool(verbose)
                }
            }

            # Initialize inverse design object if not already initialized
            if not hasattr(self, 'aid'):
                self.aid = agnostic_invDes(device=self.device)
                self.aid.updatePhysicsFunctions(self.physics_package)

            # Update simulation dictionary
            self.aid.simDict = self.simDict

            try:
                # Run optimization
                result = self.aid.optimize(
                    x0=x0,
                    minimize_kwargs=minimize_kwargs
                )

                # Compute and store final MOE output
                if hasattr(self.physics_package, 'set_simulation'):
                    # Update simulation with optimized parameters
                    self.physics_package.set_simulation(
                        simRange=self.simDict['simRange'],
                        simResolution=self.simDict['simResolution'],
                        simScale='linear',
                        third_variables=self.simDict.get('third_vars', None),
                        logSim=False
                    )

                    # Compute final MOE output for all wavelengths
                    parameters = result.x if hasattr(result, 'x') else result
                    parameters = self._ensure_tensor(parameters)
                    self.simDict['parameters'] = parameters

                    # Initialize output MOE tensor
                    outputMOE = torch.zeros_like(self.simDict['simRange'])

                    # Compute transmission for each wavelength
                    for i, wavelength in enumerate(self.simDict['simRange']):
                        # Update simPoint for current wavelength
                        self.simDict['simPoint'] = wavelength

                        # Compute transfer matrices and fields
                        tms = self.transferMatrices(self.simDict)
                        self.simDict['parameters'] = parameters  # Ensure parameters are in simDict
                        fields = self.physics_package.globalBoundaries(
                            parameters, self.simDict
                        )

                        # Compute transmission and store result
                        outputMOE[i] = self.physics_package.T(fields)

                    # Store the complete MOE output tensor
                    self.physics_package.outputMOE = outputMOE

                return result
            except Exception as e:
                print(f"Error in optimization: {str(e)}")
                print(f"Input shape: {x0.shape}")
                print(f"Device: {self.device}")
                raise

        except Exception as e:
            print(f"Error in run_optimization: {str(e)}")
            print(f"Device: {self.device}")
            raise

    def init_analysis(self, simRange, simResolution, simScale, simType, third_variables,
                     logSim=False, substrates=None, superstrates=None, num_layers=None):
        """Initialize analysis with device support"""
        if num_layers is None:
            num_layers = self.num_layers
        if substrates is None:
            substrates = self.substrates
        if superstrates is None:
            superstrates = self.superstrates

        self.physics_package.set_structure(
            simType=simType,
            num_layers=num_layers,
            mat1Call=self.mat1Call,
            mat2Call=self.mat2Call,
            param0Call=self.param0Call,
            paramNCall=self.paramNCall,
            customInputCall=self.customInputCall,
            substrates=substrates,
            superstrates=superstrates
        )

        self.physics_package.set_simulation(
            simRange=simRange,
            simResolution=simResolution,
            simScale=simScale,
            third_variables=third_variables,
            logSim=logSim
        )

    def transferMatrices(self, simDict):
        """Compute transfer matrices with device support"""
        parameters = simDict.get('parameters', None)
        if parameters is None:
            return []

        # Initialize lists
        tms = []
        tracked_info = {}

        # Ensure sim_params contains necessary information
        sim_params = {
            'simType': self.simType,
            'simRange': self.physics_package.simRange,
            'simResolution': self.physics_package.simResolution,
            'simScale': self.physics_package.simScale,
            'third_variables': simDict.get('third_vars', None),  # Pass through third_variables
            'physics': self.physics_package.get_name(),
            'simPoint': simDict.get('simPoint', self.physics_package.simPoint),  # Add simPoint
            'wavelength': simDict.get('simPoint', self.physics_package.simPoint),  # Add wavelength for compatibility
            'mat1_params': {'refractiveIndex': 1.0},  # Default material parameters
            'mat2_params': {'refractiveIndex': 1.0}
        }

        # Ensure parameters is a tensor with gradients
        if not torch.is_tensor(parameters):
            parameters = torch.tensor(parameters, dtype=torch.float32, device=self.device, requires_grad=True)
        elif parameters.device != self.device:
            parameters = parameters.to(self.device)

        # Process each layer
        for i in range(len(parameters)):
            # Combine parameters for current interface
            combined_params = torch.stack([
                parameters[i],
                parameters[i] if i < len(parameters) - 1 else parameters[-1],
                parameters[i + 1] if i < len(parameters) - 1 else parameters[-1]
            ])

            # Compute transfer matrix for current layer
            tm_result = self.physics_package.tm(combined_params, sim_params, tracked_info)
            if isinstance(tm_result, tuple):
                tm, tracked_info = tm_result
            else:
                tm = tm_result
                tracked_info = {}

            tms.append(tm)

        return tms
