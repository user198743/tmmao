import torch
import numpy as np
from math import pi
from cmath import exp, sqrt
import torch.nn.functional as F
from misc_utils import comp_utils

class optics_tmm(comp_utils):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.simRange = None
        self.simResolution = None
        self.simScale = None
        self.mat1 = None
        self.mat2 = None
        self.param0 = None
        self.paramN = None
        self.outputMOE = None
        self._ensure_tensor = lambda x: torch.tensor(x, device=self.device) if not torch.is_tensor(x) else x.to(self.device)
        return

    def set_materials(self, mat1=None, mat2=None, param0=None, paramN=None):
        self.mat1 = mat1
        self.mat2 = mat2
        self.param0 = param0
        self.paramN = paramN

    @property
    def costFunction(self):
        return self.costFunc

    @property
    def costFunction_gradPhi(self):
        return self.d_costFunc_phi

    @property
    def costFunction_gradFields(self):
        return self.d_costFunc_fields

    @property
    def costFunction_gradE(self):
        return self.d_costFunc_fields  # This is the same as gradFields for our implementation

    @property
    def globalBoundaries(self):
        return self.global_bcs

    @property
    def transferMatrices(self):
        """Property decorator for transfer matrices calculation with device support and gradient tracking"""
        def transfer_matrices_func(self, sim_dict):
            """Calculate transfer matrices with device support and gradient tracking"""
            try:
                # Extract parameters from sim_dict
                parameters = sim_dict.get('parameters')
                if parameters is None:
                    raise ValueError("No parameters found in sim_dict")

                # Get simulation parameters
                sim_point = self.get_sim_point(sim_dict)
                sim_params = {
                    'simPoint': sim_point,
                    'physics': 'optics',
                    'polarization': self.get_polarization(sim_dict.get('third_vars')),
                    'device': self.device
                }

                # Calculate material parameters
                mat1_params = self.mat1(sim_params) if self.mat1 else None
                mat2_params = self.mat2(sim_params) if self.mat2 else None
                param0_val = self.param0(sim_params) if self.param0 else None
                paramN_val = self.paramN(sim_params) if self.paramN else None

                # Ensure parameters is a tensor with gradients enabled
                if not torch.is_tensor(parameters):
                    parameters = torch.tensor(parameters, dtype=torch.float32, device=self.device, requires_grad=True)
                elif not parameters.requires_grad:
                    parameters.requires_grad_(True)

                # Calculate transfer matrices with gradient tracking
                tms = self.tm(parameters, sim_params)

                return tms

            except Exception as e:
                print(f"Error in transfer_matrices_func: {str(e)}")
                print(f"Device: {self.device}")
                print(f"Parameters shape: {parameters.shape if torch.is_tensor(parameters) else None}")
                raise

        return transfer_matrices_func.__get__(self, self.__class__)

    @property
    def transferMatrices_gradPhi(self):
        def transfer_matrices_grad_func(simDict):
            parameters = simDict.get('parameters', None)
            if parameters is None:
                return []

            # Initialize lists
            dtms = []
            tracked_info = {}

            # Get simulation parameters
            sim_params = {
                'physics': 'optics',
                'simPoint': simDict.get('simPoint', None),  # Changed from callPoint to simPoint
                'simType': simDict.get('simType', 'MOE'),
                'simRange': simDict.get('simRange', None),
                'simResolution': simDict.get('simResolution', None),
                'device': self.device
            }

            # Process each layer
            for i in range(len(parameters) - 1):
                x = parameters[i]
                y = parameters[i]
                yp1 = parameters[i + 1]

                # Compute transfer matrix for current layer
                tm = self.tm(x, y, yp1, sim_params, tracked_info)
                dtm_result = self.dtm(x, y, yp1, sim_params, tracked_info, tm)
                if isinstance(dtm_result, tuple):
                    dtm, tracked_info = dtm_result
                else:
                    dtm = dtm_result

                dtms.append(dtm)

            return dtms

        return transfer_matrices_grad_func

    def get_name(self):
        return 'optics'

    def set_structure(self, simType='MOE', num_layers=None, initial_layer_thickness=None,
                     mat1Call=None, mat2Call=None, param0Call=None, paramNCall=None,
                     customInputCall=None, substrates=None, superstrates=None,
                     y_bounds=None, x_bounds=None, num_intervals=None):
        """Set structure with proper tensor handling for material properties"""
        self.simType = simType
        self.num_layers = num_layers

        if initial_layer_thickness is not None:
            self.initial_layer_thickness = self._ensure_tensor(initial_layer_thickness)

        # Create wrapper functions for material properties with error handling and device tracking
        if mat1Call is not None:
            def mat1_wrapper(simRange):
                try:
                    result = mat1Call(simRange)
                    if isinstance(result, dict):
                        if 'refractiveIndex' not in result:
                            raise KeyError(f"Material property dictionary must contain 'refractiveIndex'. Got keys: {list(result.keys())}")
                        n = result['refractiveIndex']
                        return torch.tensor(n, device=self.device, dtype=torch.complex64)
                    return self._ensure_tensor(result)
                except Exception as e:
                    print(f"ERROR in mat1_wrapper: {str(e)}")
                    raise

            self.mat1Call = mat1_wrapper

        if mat2Call is not None:
            def mat2_wrapper(simRange):
                try:
                    result = mat2Call(simRange)
                    if isinstance(result, dict):
                        if 'refractiveIndex' not in result:
                            raise KeyError(f"Material property dictionary must contain 'refractiveIndex'. Got keys: {list(result.keys())}")
                        n = result['refractiveIndex']
                        return torch.tensor(n, device=self.device, dtype=torch.complex64)
                    return self._ensure_tensor(result)
                except Exception as e:
                    print(f"ERROR in mat2_wrapper: {str(e)}")
                    raise

            self.mat2Call = mat2_wrapper

        if param0Call is not None:
            self.param0Call = param0Call
        if paramNCall is not None:
            self.paramNCall = paramNCall
        if customInputCall is not None:
            self.customInputCall = customInputCall

        # Convert substrates and superstrates to tensors if they're not already
        if substrates is not None:
            if isinstance(substrates, dict):
                self.substrates = {k: torch.tensor(v, device=self.device) if not torch.is_tensor(v) else v.to(self.device)
                                 for k, v in substrates.items()}
            else:  # Handle list or other input types
                if isinstance(substrates, (list, tuple)):
                    # Convert numeric values to tensors, keep strings as is
                    self.substrates = [torch.tensor(v, device=self.device) if isinstance(v, (int, float, complex, np.ndarray))
                                     else v for v in substrates]
                else:
                    self.substrates = substrates
        else:
            self.substrates = None

        if superstrates is not None:
            if isinstance(superstrates, dict):
                self.superstrates = {k: torch.tensor(v, device=self.device) if not torch.is_tensor(v) else v.to(self.device)
                                   for k, v in superstrates.items()}
            else:  # Handle list or other input types
                if isinstance(superstrates, (list, tuple)):
                    # Convert numeric values to tensors, keep strings as is
                    self.superstrates = [torch.tensor(v, device=self.device) if isinstance(v, (int, float, complex, np.ndarray))
                                       else v for v in superstrates]
                else:
                    self.superstrates = superstrates
        else:
            self.superstrates = None

    def set_simulation(self, simRange, simResolution=1, simScale='linear', third_variables=None, logSim=False, simPoint=None):
        """Set simulation parameters with device support"""
        # Handle tensor parameters
        self.simRange = simRange.clone().detach().to(device=self.device) if torch.is_tensor(simRange) else torch.tensor(simRange, dtype=torch.float64, device=self.device)

        # Handle scalar parameters
        self.simResolution = simResolution  # Keep as scalar
        self.simScale = simScale  # Keep as string
        self.logSim = logSim  # Keep as boolean

        # Handle third variables with device support
        if third_variables is not None:
            if isinstance(third_variables, dict):
                self.third_variables = {k: torch.tensor(v, device=self.device) if not torch.is_tensor(v) else v.to(self.device)
                                      for k, v in third_variables.items()}
            else:
                self.third_variables = torch.tensor(third_variables, device=self.device) if not torch.is_tensor(third_variables) else third_variables.to(self.device)
        else:
            self.third_variables = None

        # Set simulation point
        if simPoint is not None:
            self.simPoint = simPoint.clone().detach().to(device=self.device) if torch.is_tensor(simPoint) else torch.tensor(simPoint, dtype=torch.float64, device=self.device)
        else:
            self.simPoint = self.simRange[0].clone().detach()

        # Initialize simulation points
        if len(self.simRange) > 1:
            start = self.simRange[0].item()
            end = self.simRange[-1].item()
            num_points = int(self.simResolution * (len(self.simRange) - 1) + 1)
            self.simPoints = torch.linspace(start, end, num_points, dtype=torch.float64, device=self.device)
            if self.logSim:
                self.simPoints = torch.exp(self.simPoints)
        else:
            self.simPoints = self.simRange.clone().detach()

    def safe_extract_value(self, params, key='n', default_value=1.0):
        """Safely extract value from parameters with proper device handling"""
        try:
            if params is None:
                return torch.tensor(default_value, dtype=torch.complex64, device=self.device)

            if isinstance(params, (int, float, complex)):
                return torch.tensor(params, dtype=torch.complex64, device=self.device)

            if isinstance(params, dict):
                if key in params:
                    value = params[key]
                    if torch.is_tensor(value):
                        return value.to(device=self.device, dtype=torch.complex64)
                    return torch.tensor(value, dtype=torch.complex64, device=self.device)
                return torch.tensor(default_value, dtype=torch.complex64, device=self.device)

            if torch.is_tensor(params):
                return params.to(device=self.device, dtype=torch.complex64)

            # Default fallback
            return torch.tensor(default_value, dtype=torch.complex64, device=self.device)

        except Exception as e:
            print(f"Error in safe_extract_value: {str(e)}")
            print(f"params: {params}")
            print(f"key: {key}")
            print(f"default_value: {default_value}")
            raise

    def get_sim_point(self, sim_params):
        if 'simPoint' in sim_params:
            return torch.tensor(float(sim_params['simPoint']), dtype=torch.float32, device=self.device)
        elif 'wavelength' in sim_params:
            return torch.tensor(float(sim_params['wavelength']), dtype=torch.float32, device=self.device)
        else:
            raise ValueError("No simulation point or wavelength found in sim_params")

    def get_polarization(self, sim_params=None):
        """Get polarization from sim_params or return default."""
        if sim_params is None:
            return 's'  # Default polarization
        third_vars = sim_params.get('third_vars', [{'polarization': 's'}])
        if not third_vars:
            return 's'
        return third_vars[0].get('polarization', 's')

    def global_bcs(self, x, sim_params):
        """Calculate global boundary conditions with device support"""
        try:
            # Ensure x is a tensor with proper device placement
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            elif x.device != self.device:
                x = x.to(self.device)

            # Calculate transfer matrices
            tms = self.transferMatrices(sim_params)

            # Initialize fields with proper device placement
            fields = torch.zeros(2, dtype=torch.complex64, device=self.device)
            fields[0] = 1.0  # Incident field amplitude

            # Calculate global fields
            result = torch.matmul(tms, fields)

            return result

        except Exception as e:
            print(f"Error in global_bcs: {str(e)}")
            print(f"x shape: {x.shape if torch.is_tensor(x) else None}")
            print(f"Device: {self.device}")
            raise

    def left_tm(self, parameters, sim_params):
        """Calculate left transfer matrix with device support"""
        try:
            # Get simulation parameters safely
            sim_point = sim_params.get('simPoint', sim_params.get('wavelength'))
            if sim_point is None:
                raise ValueError("No simulation point or wavelength found in sim_params")

            # Convert to tensor and calculate k0
            k0 = 2 * torch.pi * torch.tensor(float(sim_point), dtype=torch.float32, device=self.device)

            # Get material parameters with default value
            n0 = torch.tensor(1.0, dtype=torch.complex64, device=self.device)

            # Ensure parameters is a tensor with gradients
            if not torch.is_tensor(parameters):
                parameters = torch.tensor(parameters, dtype=torch.float32, device=self.device, requires_grad=True)
            elif parameters.device != self.device:
                parameters = parameters.to(self.device)

            # Calculate phase with proper broadcasting
            if parameters.dim() > 0:
                phi = k0 * n0 * parameters.unsqueeze(-1)
            else:
                phi = k0 * n0 * parameters

            # Build transfer matrix with proper broadcasting
            tm = torch.zeros((*phi.shape[:-1], 2, 2), dtype=torch.complex64, device=self.device)
            tm[..., 0, 0] = torch.exp(-1j * phi.squeeze(-1))
            tm[..., 0, 1] = 0
            tm[..., 1, 0] = 0
            tm[..., 1, 1] = torch.exp(1j * phi.squeeze(-1))

            return tm

        except Exception as e:
            print(f"Error in left_tm: {str(e)}")
            print(f"Parameters shape: {parameters.shape if torch.is_tensor(parameters) else None}")
            print(f"Device: {parameters.device if torch.is_tensor(parameters) else None}")
            raise

    def d_left_tm(self, yp1, sim_params, tracked_info, tm, matLeft={}, mat1Right={}, mat2Right={}):
        """Calculate derivative of left transfer matrix"""
        # Calculate derivatives using interface_tm derivatives
        dx, dy, _ = self.d_interface_tm(0, yp1, yp1, sim_params, tracked_info, tm, matLeft, matLeft, mat1Right, mat2Right)

        # Return derivatives directly as PyTorch tensors
        return dx, dy

    # Override comp_utils methods for PyTorch compatibility
    def mul(self, a, b):
        """Matrix multiplication with proper device handling"""
        try:
            # Ensure inputs are tensors on the correct device
            if not torch.is_tensor(a):
                a = torch.tensor(a, dtype=torch.complex64, device=self.device)
            elif a.device != self.device:
                a = a.to(self.device)

            if not torch.is_tensor(b):
                b = torch.tensor(b, dtype=torch.complex64, device=self.device)
            elif b.device != self.device:
                b = b.to(self.device)

            # Perform matrix multiplication
            return torch.matmul(a, b)

        except Exception as ex:
            print(f"Error in mul: {str(ex)}")
            print(f"a shape: {a.shape if torch.is_tensor(a) else None}")
            print(f"b shape: {b.shape if torch.is_tensor(b) else None}")
            print(f"a device: {a.device if torch.is_tensor(a) else None}")
            print(f"b device: {b.device if torch.is_tensor(b) else None}")
            raise

    def div(self, a, b):
        """Division with proper device handling"""
        try:
            # Ensure inputs are tensors on the correct device
            if not torch.is_tensor(a):
                a = torch.tensor(a, dtype=torch.complex64, device=self.device)
            elif a.device != self.device:
                a = a.to(self.device)

            if not torch.is_tensor(b):
                b = torch.tensor(b, dtype=torch.complex64, device=self.device)
            elif b.device != self.device:
                b = b.to(self.device)

            # Perform division
            return a / b

        except Exception as ex:
            print(f"Error in div: {str(ex)}")
            print(f"a shape: {a.shape if torch.is_tensor(a) else None}")
            print(f"b shape: {b.shape if torch.is_tensor(b) else None}")
            print(f"a device: {a.device if torch.is_tensor(a) else None}")
            print(f"b device: {b.device if torch.is_tensor(b) else None}")
            raise

    def interface_tm(self, parameters, sim_params, tracked_info=None, k0=None, kx=None, polarization=None):
        """Calculate interface transfer matrix with device support"""
        try:
            # Ensure parameters is a tensor with gradients
            if not torch.is_tensor(parameters):
                parameters = torch.tensor(parameters, dtype=torch.float32, device=self.device, requires_grad=True)
            elif not parameters.requires_grad:
                parameters.requires_grad_(True)

            # Get simulation parameters if not provided
            if k0 is None or kx is None or polarization is None:
                sim_point = sim_params.get('simPoint', sim_params.get('wavelength'))
                if sim_point is None:
                    raise ValueError("No simulation point or wavelength found in sim_params")

                k0 = 2 * torch.pi * torch.tensor(float(sim_point), dtype=torch.float32, device=self.device)
                theta = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                kx = k0 * torch.sin(theta)
                third_vars = sim_params.get('third_vars', [{'polarization': 's'}])
                polarization = third_vars[0].get('polarization', 's') if third_vars else 's'

            # Get material parameters from sim_params
            mat1_params = sim_params.get('mat1_params', {'refractiveIndex': 1.0})
            mat2_params = sim_params.get('mat2_params', {'refractiveIndex': 1.0})

            # Get refractive indices and ensure they're tensors
            n1 = torch.tensor(mat1_params['refractiveIndex'], dtype=torch.complex64, device=self.device)
            n2 = torch.tensor(mat2_params['refractiveIndex'], dtype=torch.complex64, device=self.device)

            # Calculate wave vectors
            kz1 = torch.sqrt(k0**2 * n1**2 - kx**2 + 0j)
            kz2 = torch.sqrt(k0**2 * n2**2 - kx**2 + 0j)

            # Calculate transfer matrix elements based on polarization
            if polarization == 's':
                r = (kz1 - kz2) / (kz1 + kz2)
                t = 2 * kz1 / (kz1 + kz2)
            else:  # p-polarization
                r = (n2**2 * kz1 - n1**2 * kz2) / (n2**2 * kz1 + n1**2 * kz2)
                t = 2 * n1 * n2 * kz1 / (n2**2 * kz1 + n1**2 * kz2)

            # Build transfer matrix
            tm = torch.zeros((2, 2), dtype=torch.complex64, device=self.device)
            tm[0, 0] = 1 / t
            tm[0, 1] = r / t
            tm[1, 0] = r / t
            tm[1, 1] = 1 / t

            return tm

        except Exception as e:
            print(f"Error in interface_tm: {str(e)}")
            print(f"Parameters shape: {parameters.shape if torch.is_tensor(parameters) else None}")
            print(f"Device: {parameters.device if torch.is_tensor(parameters) else None}")
            print(f"k0: {k0}")
            print(f"kx: {kx}")
            print(f"polarization: {polarization}")
            raise

    def get_tr(self, kzL, kzR, nL, nR, pol):
        """Calculate transmission and reflection coefficients."""
        # Ensure all inputs are complex tensors on the correct device
        kzL = torch.as_tensor(kzL, dtype=torch.complex64, device=self.device)
        kzR = torch.as_tensor(kzR, dtype=torch.complex64, device=self.device)
        nL = torch.as_tensor(nL, dtype=torch.complex64, device=self.device)
        nR = torch.as_tensor(nR, dtype=torch.complex64, device=self.device)

        if pol == 'TE':
            # TE mode calculations
            num = 2 * kzL
            den = kzL + kzR
            t = num / den
            r = (kzL - kzR) / den
        else:
            # TM mode calculations
            num = 2 * nR * kzL
            den = nL * kzR + nR * kzL
            t = num / den
            r = (nL * kzR - nR * kzL) / den

        return t, r

    def right_tm(self, parameters, sim_params):
        """Calculate right transfer matrix with device support"""
        try:
            # Get simulation parameters safely
            sim_point = sim_params.get('simPoint', sim_params.get('wavelength'))
            if sim_point is None:
                raise ValueError("No simulation point or wavelength found in sim_params")

            # Convert to tensor and calculate k0
            k0 = 2 * torch.pi * torch.tensor(float(sim_point), dtype=torch.float32, device=self.device)

            # Get material parameters with default value
            nN = torch.tensor(1.0, dtype=torch.complex64, device=self.device)

            # Ensure parameters is a tensor with gradients
            if not torch.is_tensor(parameters):
                parameters = torch.tensor(parameters, dtype=torch.float32, device=self.device, requires_grad=True)
            elif parameters.device != self.device:
                parameters = parameters.to(self.device)

            # Calculate phase with proper broadcasting
            if parameters.dim() > 0:
                phi = k0 * nN * parameters.unsqueeze(-1)
            else:
                phi = k0 * nN * parameters

            # Build transfer matrix with proper broadcasting
            tm = torch.zeros((*phi.shape[:-1], 2, 2), dtype=torch.complex64, device=self.device)
            tm[..., 0, 0] = torch.exp(-1j * phi.squeeze(-1))
            tm[..., 0, 1] = 0
            tm[..., 1, 0] = 0
            tm[..., 1, 1] = torch.exp(1j * phi.squeeze(-1))

            return tm

        except Exception as e:
            print(f"Error in right_tm: {str(e)}")
            print(f"Parameters shape: {parameters.shape if torch.is_tensor(parameters) else None}")
            print(f"Device: {parameters.device if torch.is_tensor(parameters) else None}")
            raise

    def d_right_tm(self, y, sim_params, tracked_info, tm, matRight={}, mat1Left={}, mat2Left={}):
        """Calculate derivative of right transfer matrix."""
        # Get simulation point and calculate k0
        simPoint = self.get_sim_point(sim_params)
        k0 = 2 * np.pi * simPoint
        k0_tensor = torch.as_tensor(k0, dtype=torch.complex64, device=self.device)

        # Convert y to tensor if needed
        y_val = self.safe_extract_value(y)
        y = y_val.clone().detach() if torch.is_tensor(y_val) else torch.as_tensor(y_val, dtype=torch.complex64, device=self.device)

        # Extract material parameters with defaults
        n1_left = self.safe_extract_value(mat1Left, key='n', default_value=1.0)
        k1_left = self.safe_extract_value(mat1Left, key='k', default_value=0.0)
        n2_left = self.safe_extract_value(mat2Left, key='n', default_value=1.0)
        k2_left = self.safe_extract_value(mat2Left, key='k', default_value=0.0)
        nR = self.safe_extract_value(matRight, key='n', default_value=1.0)
        kR = self.safe_extract_value(matRight, key='k', default_value=0.0)

        # Convert to complex refractive indices
        n1_left_complex = torch.as_tensor(n1_left - 1j * k1_left, dtype=torch.complex64, device=self.device)
        n2_left_complex = torch.as_tensor(n2_left - 1j * k2_left, dtype=torch.complex64, device=self.device)
        nR_complex = torch.as_tensor(nR - 1j * kR, dtype=torch.complex64, device=self.device)

        # Calculate effective index for left medium
        nLeft = self.mul(n1_left_complex, y) + self.mul(n2_left_complex, (1-y))

        # Calculate wave vectors
        kz_left = torch.sqrt(k0_tensor**2 * nLeft**2 - self.kx**2 + 0j)
        kz_right = torch.sqrt(k0_tensor**2 * nR_complex**2 - self.kx**2 + 0j)

        # Calculate polarization-dependent coefficients
        if self.get_polarization(sim_params) == 'p':
            alpha_left = kz_left / (k0_tensor * nLeft**2)
            alpha_right = kz_right / (k0_tensor * nR_complex**2)
            # Calculate derivatives
            dalpha_left = (k0_tensor * (n1_left_complex - n2_left_complex) *
                         (kz_left / (k0_tensor * nLeft**3)))
        else:  # 's' polarization
            alpha_left = kz_left / k0_tensor
            alpha_right = kz_right / k0_tensor
            # Calculate derivatives
            dalpha_left = (k0_tensor * (n1_left_complex - n2_left_complex) *
                         (1 / (2 * k0_tensor * kz_left)))

        # Calculate derivative of transfer matrix
        dtm = torch.zeros((2, 2), dtype=torch.complex64, device=self.device)
        dtm[0, 0] = 0.5 * dalpha_left / alpha_right
        dtm[0, 1] = -0.5 * dalpha_left / alpha_right
        dtm[1, 0] = -0.5 * dalpha_left / alpha_right
        dtm[1, 1] = 0.5 * dalpha_left / alpha_right

        return dtm, tracked_info

    def tm(self, parameters, sim_params, tracked_info=None):
        """Calculate transfer matrix with proper device handling and gradient tracking"""
        try:
            # Ensure parameters is a tensor with gradients
            if not torch.is_tensor(parameters):
                parameters = torch.tensor(parameters, dtype=torch.float32, device=self.device, requires_grad=True)
            elif parameters.device != self.device:
                parameters = parameters.to(self.device)

            # Extract x, y, yp1 from combined parameters
            x, y, yp1 = parameters[0], parameters[1], parameters[2]

            # Get simulation parameters safely
            sim_point = sim_params.get('simPoint', sim_params.get('wavelength'))
            if sim_point is None:
                raise ValueError("No simulation point or wavelength found in sim_params")

            # Convert to tensor and calculate k0
            k0 = 2 * torch.pi * torch.tensor(float(sim_point), dtype=torch.float32, device=self.device)

            # Calculate transfer matrices for each interface
            left_tm = self.left_tm(x, sim_params)
            interface_tm = self.interface_tm(y, sim_params, None, k0, None, None)
            right_tm = self.right_tm(yp1, sim_params)

            # Combine matrices while maintaining gradient connection
            result = self.mul(self.mul(left_tm, interface_tm), right_tm)

            return result

        except Exception as e:
            print(f"Error in tm calculation: {str(e)}")
            print(f"Parameters shape: {parameters.shape if torch.is_tensor(parameters) else None}")
            print(f"Device: {parameters.device if torch.is_tensor(parameters) else None}")
            raise

    def dtm(self, x, y, yp1, sim_params, tracked_info, tm, mat1params={}, mat2params={}):
        return self.d_interface_tm(x, y, yp1, sim_params, tracked_info, tm, mat1params, mat2params, mat1params, mat2params)

    def d_interface_tm(self, x, y, yp1, sim_params, tracked_info, tm, mat1Left={}, mat2Left={}, mat1Right={}, mat2Right={}):
        """Calculate derivative of interface transfer matrix"""
        # Set default refractive indices if not provided
        default_mat = {'refractiveIndex': 1.0}  # Default to air
        mat1Left = mat1Left or default_mat
        mat2Left = mat2Left or default_mat
        mat1Right = mat1Right or default_mat
        mat2Right = mat2Right or default_mat

        # Get kx from tracked_info
        kx = tracked_info.get('kx', 0.0)
        kx_t = torch.tensor(kx, dtype=torch.complex64, device=self.device) if not torch.is_tensor(kx) else kx.to(self.device)

        # Handle log_ratio and n_is_optParam cases
        if not self.n_is_optParam:
            if self.log_ratio:
                y = torch.tensor((100**float(y) - 1)/99, dtype=torch.float32, device=self.device)
                yp1 = torch.tensor((100**float(yp1) - 1)/99, dtype=torch.float32, device=self.device)

            # Get refractive indices and interpolate
            n1L = torch.tensor(mat1Left['refractiveIndex'], dtype=torch.complex64, device=self.device)
            n2L = torch.tensor(mat2Left['refractiveIndex'], dtype=torch.complex64, device=self.device)
            n1R = torch.tensor(mat1Right['refractiveIndex'], dtype=torch.complex64, device=self.device)
            n2R = torch.tensor(mat2Right['refractiveIndex'], dtype=torch.complex64, device=self.device)

            # Calculate derivatives of refractive indices
            dnL_dy = n2L - n1L
            dnR_dyp1 = n2R - n1R

            # Calculate wave vectors and their derivatives
            k0 = 2 * torch.pi * self.get_sim_point(sim_params)
            kL = k0 * (y * n2L + (1-y) * n1L)
            kR = k0 * (yp1 * n2R + (1-yp1) * n1R)

            dkL_dy = k0 * dnL_dy
            dkR_dyp1 = k0 * dnR_dyp1

            # Calculate z components and their derivatives
            kzL = torch.sqrt(kL**2 - kx_t**2 + 0j)
            kzR = torch.sqrt(kR**2 - kx_t**2 + 0j)

            dkzL_dy = kL * dkL_dy / kzL
            dkzR_dyp1 = kR * dkR_dyp1 / kzR

            # Calculate epsilon values for p-polarization
            epL = kL**2
            epR = kR**2

        else:
            # Direct use of y and yp1 as refractive indices
            k0 = 2 * torch.pi * self.get_sim_point(sim_params)
            kL = k0 * torch.tensor(float(y), dtype=torch.complex64, device=self.device)
            kR = k0 * torch.tensor(float(yp1), dtype=torch.complex64, device=self.device)

            dkL_dy = k0
            dkR_dyp1 = k0

            kzL = torch.sqrt(kL**2 - kx_t**2 + 0j)
            kzR = torch.sqrt(kR**2 - kx_t**2 + 0j)

            dkzL_dy = kL * dkL_dy / kzL
            dkzR_dyp1 = kR * dkR_dyp1 / kzR

            epL = kL**2
            epR = kR**2

        # Get polarization
        polarization = 'p'
        if 'third_vars' in sim_params:
            if isinstance(sim_params['third_vars'], list):
                if len(sim_params['third_vars']) > 0:
                    polarization = sim_params['third_vars'][0].get('polarization', 'p')
            else:
                polarization = sim_params['third_vars'].get('polarization', 'p')

        # Calculate transmission and reflection coefficient derivatives
        if polarization == 'p':
            d = epL * kzR + kzL * epR
            dt_dy = 2 * torch.sqrt(epL * epR) * kzL * dkL_dy / d
            dr_dy = (epR * kzL * dkL_dy - epL * kzR * dkL_dy) / d
            dt_dyp1 = 2 * torch.sqrt(epL * epR) * kzL * dkR_dyp1 / d
            dr_dyp1 = (epR * kzL * dkR_dyp1 - epL * kzR * dkR_dyp1) / d
        else:  # s-polarization
            d = kzR + kzL
            dt_dy = 2 * kzL * dkL_dy / d
            dr_dy = (kzL * dkL_dy - kzR * dkL_dy) / d
            dt_dyp1 = 2 * kzL * dkR_dyp1 / d
            dr_dyp1 = (kzL * dkR_dyp1 - kzR * dkR_dyp1) / d

        # Calculate phase derivatives
        dphase_dx = 1j * kzL
        dphase_dy = 1j * x * dkzL_dy

        # Calculate transfer matrix derivatives
        dt_dx = torch.zeros_like(tm, dtype=torch.complex64, device=self.device)
        dt_dy = torch.zeros_like(tm, dtype=torch.complex64, device=self.device)
        dt_dyp1 = torch.zeros_like(tm, dtype=torch.complex64, device=self.device)

        # Phase contribution
        dt_dx[0, 0] = -dphase_dx * tm[0, 0]
        dt_dx[1, 1] = dphase_dx * tm[1, 1]

        # Material parameter contributions
        dt_dy[0, 0] = -dphase_dy * tm[0, 0] + (1/dt_dy) * torch.tensor([[1, dr_dy], [dr_dy, 1]], dtype=torch.complex64, device=self.device)[0, 0]
        dt_dy[0, 1] = (1/dt_dy) * torch.tensor([[1, dr_dy], [dr_dy, 1]], dtype=torch.complex64, device=self.device)[0, 1]
        dt_dy[1, 0] = (1/dt_dy) * torch.tensor([[1, dr_dy], [dr_dy, 1]], dtype=torch.complex64, device=self.device)[1, 0]
        dt_dy[1, 1] = dphase_dy * tm[1, 1] + (1/dt_dy) * torch.tensor([[1, dr_dy], [dr_dy, 1]], dtype=torch.complex64, device=self.device)[1, 1]

        dt_dyp1[0, 0] = (1/dt_dyp1) * torch.tensor([[1, dr_dyp1], [dr_dyp1, 1]], dtype=torch.complex64, device=self.device)[0, 0]
        dt_dyp1[0, 1] = (1/dt_dyp1) * torch.tensor([[1, dr_dyp1], [dr_dyp1, 1]], dtype=torch.complex64, device=self.device)[0, 1]
        dt_dyp1[1, 0] = (1/dt_dyp1) * torch.tensor([[1, dr_dyp1], [dr_dyp1, 1]], dtype=torch.complex64, device=self.device)[1, 0]
        dt_dyp1[1, 1] = (1/dt_dyp1) * torch.tensor([[1, dr_dyp1], [dr_dyp1, 1]], dtype=torch.complex64, device=self.device)[1, 1]

        # Convert to numpy arrays before returning
        return (dt_dx.detach().numpy(),
                dt_dy.detach().numpy(),
                dt_dyp1.detach().numpy())

    def Tdb(self, e):
        """Transmission in dB"""
        return 20 * torch.log10(torch.abs(e[0]))

    def Rdb(self, e):
        """Reflection in dB"""
        return 20 * torch.log10(torch.abs(e[1]))

    def A(self, e):
        """Absorption with proper tensor handling"""
        try:
            # Ensure input is a tensor on the correct device
            if not torch.is_tensor(e):
                e = torch.tensor(e, dtype=torch.complex64, device=self.device)
            elif e.device != self.device:
                e = e.to(device=self.device)

            # Handle both single field and batched fields
            if e.dim() == 1:
                return 1 - (torch.abs(e[0])**2 + torch.abs(e[1])**2)
            else:
                return 1 - (torch.abs(e[:, 0])**2 + torch.abs(e[:, 1])**2)
        except Exception as ex:
            print(f"Error in A calculation: {str(ex)}")
            print(f"Input shape: {e.shape if torch.is_tensor(e) else None}")
            raise

    def T(self, fields):
        """Calculate transmission with proper gradient tracking"""
        try:
            # Get field components while preserving gradients
            if isinstance(fields, (list, tuple)):
                E_inc = fields[0]
                E_trans = fields[-1]
            else:
                E_inc = fields[0:1]
                E_trans = fields[-1:]

            # Calculate transmission while maintaining gradient connection
            T = torch.abs(E_trans)**2 / torch.abs(E_inc)**2

            # Ensure result maintains gradient connection and is a scalar
            T_real = T.real
            if T_real.numel() > 1:
                T_real = T_real.mean()  # Use mean for multiple values

            return T_real

        except Exception as e:
            print(f"Error in T calculation: {str(e)}")
            print(f"Fields shape: {fields.shape if torch.is_tensor(fields) else [f.shape if torch.is_tensor(f) else None for f in fields]}")
            raise

    def R(self, e):
        """Reflection with proper tensor handling"""
        try:
            # Ensure input is a tensor on the correct device
            if not torch.is_tensor(e):
                e = torch.tensor(e, dtype=torch.complex64, device=self.device)
            elif e.device != self.device:
                e = e.to(device=self.device)

            # Handle both single field and batched fields
            if e.dim() == 1:
                return torch.abs(e[0])**2
            else:
                return torch.abs(e[:, 0])**2
        except Exception as ex:
            print(f"Error in R calculation: {str(ex)}")
            print(f"Input shape: {e.shape if torch.is_tensor(e) else None}")
            raise

    def dTde(self, e):
        """Derivative of transmission with respect to field with proper tensor handling"""
        try:
            # Ensure input is a tensor on the correct device
            if not torch.is_tensor(e):
                e = torch.tensor(e, dtype=torch.complex64, device=self.device)
            elif e.device != self.device:
                e = e.to(device=self.device)

            # Handle both single field and batched fields
            if e.dim() == 1:
                return torch.tensor([2 * torch.conj(e[1]) / torch.abs(e[1])**2,
                                  -2 * torch.conj(e[0]) / torch.abs(e[0])**2],
                                 dtype=torch.complex64, device=self.device)
            else:
                return torch.stack([2 * torch.conj(e[:, 1]) / torch.abs(e[:, 1])**2,
                                 -2 * torch.conj(e[:, 0]) / torch.abs(e[:, 0])**2],
                                dim=1)
        except Exception as ex:
            print(f"Error in dTde calculation: {str(ex)}")
            print(f"Input shape: {e.shape if torch.is_tensor(e) else None}")
            raise

    def dRde(self, e):
        """Derivative of reflection with respect to field with proper tensor handling"""
        try:
            # Ensure input is a tensor on the correct device
            if not torch.is_tensor(e):
                e = torch.tensor(e, dtype=torch.complex64, device=self.device)
            elif e.device != self.device:
                e = e.to(device=self.device)

            # Handle both single field and batched fields
            if e.dim() == 1:
                return torch.tensor([2 * torch.conj(e[0]), 0],
                                 dtype=torch.complex64, device=self.device)
            else:
                return torch.stack([2 * torch.conj(e[:, 0]),
                                 torch.zeros_like(e[:, 1])],
                                dim=1)
        except Exception as ex:
            print(f"Error in dRde calculation: {str(ex)}")
            print(f"Input shape: {e.shape if torch.is_tensor(e) else None}")
            raise

    def dAde(self, e):
        """Derivative of absorption with respect to field with proper tensor handling"""
        try:
            # Ensure input is a tensor on the correct device
            if not torch.is_tensor(e):
                e = torch.tensor(e, dtype=torch.complex64, device=self.device)
            elif e.device != self.device:
                e = e.to(device=self.device)

            # Handle both single field and batched fields
            if e.dim() == 1:
                return -2 * torch.tensor([torch.conj(e[0]), torch.conj(e[1])],
                                      dtype=torch.complex64, device=self.device)
            else:
                return -2 * torch.stack([torch.conj(e[:, 0]), torch.conj(e[:, 1])],
                                     dim=1)
        except Exception as ex:
            print(f"Error in dAde calculation: {str(ex)}")
            print(f"Input shape: {e.shape if torch.is_tensor(e) else None}")
            raise

    def dRdbde(self, e):
        """Derivative of reflection in dB with respect to field with proper tensor handling"""
        try:
            # Ensure input is a tensor on the correct device
            if not torch.is_tensor(e):
                e = torch.tensor(e, dtype=torch.complex64, device=self.device)
            elif e.device != self.device:
                e = e.to(device=self.device)

            # Calculate log10 constant on device
            log10_const = torch.log(torch.tensor(10.0, device=self.device))

            # Handle both single field and batched fields
            if e.dim() == 1:
                return 20 / (log10_const * torch.abs(e[1])) * torch.tensor(
                    [0, torch.conj(e[1])/torch.abs(e[1])],
                    dtype=torch.complex64, device=self.device)
            else:
                zeros = torch.zeros_like(e[:, 0])
                conj_norm = torch.conj(e[:, 1])/torch.abs(e[:, 1])
                return 20 / (log10_const * torch.abs(e[:, 1])) * torch.stack([zeros, conj_norm], dim=1)
        except Exception as ex:
            print(f"Error in dRdbde calculation: {str(ex)}")
            print(f"Input shape: {e.shape if torch.is_tensor(e) else None}")
            raise

    def dTdbde(self, e):
        """Derivative of transmission in dB with respect to field with proper tensor handling"""
        try:
            # Ensure input is a tensor on the correct device
            if not torch.is_tensor(e):
                e = torch.tensor(e, dtype=torch.complex64, device=self.device)
            elif e.device != self.device:
                e = e.to(device=self.device)

            # Calculate log10 constant on device
            log10_const = torch.log(torch.tensor(10.0, device=self.device))

            # Handle both single field and batched fields
            if e.dim() == 1:
                return 20 / (log10_const * torch.abs(e[0])) * torch.tensor(
                    [torch.conj(e[0])/torch.abs(e[0]), 0],
                    dtype=torch.complex64, device=self.device)
            else:
                conj_norm = torch.conj(e[:, 0])/torch.abs(e[:, 0])
                zeros = torch.zeros_like(e[:, 1])
                return 20 / (log10_const * torch.abs(e[:, 0])) * torch.stack([conj_norm, zeros], dim=1)
        except Exception as ex:
            print(f"Error in dTdbde calculation: {str(ex)}")
            print(f"Input shape: {e.shape if torch.is_tensor(e) else None}")
            raise

    def costFunc(self, simPoints, callPoints, third_vars, cur_x, cur_y, all_mat1params,
                 all_mat2params, all_param0s, all_paramNs, all_fields, all_tms,
                 all_global_bcs, all_cf_factors, all_scheduler_factors, all_custom_input):
        """Cost function with proper gradient tracking"""
        try:
            # Initialize cost using input tensor operations to maintain gradient
            L = torch.zeros(1, dtype=torch.float32, device=self.device)
            L = L + 0 * torch.sum(cur_x[0:1])  # Connect to input tensor graph

            # Process each field
            for n, (fields, cf_factors) in enumerate(zip(all_fields, all_cf_factors)):
                # Use existing tensors or create new ones connected to input
                if isinstance(fields, (list, tuple)):
                    fields_tensor = torch.stack([
                        f if torch.is_tensor(f) else torch.tensor(f, dtype=torch.complex64, device=self.device)
                        for f in fields
                    ])
                else:
                    fields_tensor = fields if torch.is_tensor(fields) else torch.tensor(fields, dtype=torch.complex64, device=self.device)

                # Calculate transmission (T function maintains gradients)
                T_vals = self.T(fields_tensor)

                # Process cf_factors while maintaining gradient connection
                if isinstance(cf_factors, dict):
                    sorted_keys = sorted([int(k) if isinstance(k, str) else k for k in cf_factors.keys()])
                    weights = torch.tensor([cf_factors[str(k) if isinstance(k, str) else k]
                                         for k in sorted_keys],
                                        dtype=torch.float32, device=self.device)
                else:
                    weights = torch.ones(1, dtype=torch.float32, device=self.device)

                # Update cost while maintaining gradient connection
                L = L + torch.sum(weights * T_vals)

            # Return tensor for gradient computation
            final_cost = -L
            if torch.is_tensor(final_cost):
                if final_cost.numel() > 1:
                    final_cost = final_cost.mean()

            return final_cost  # Return tensor for gradient computation

        except Exception as e:
            print(f"Error in costFunc: {str(e)}")
            print(f"Device: {self.device}")
            print(f"Fields shape: {[f.shape if torch.is_tensor(f) else None for f in all_fields]}")
            print(f"Current x shape: {cur_x.shape if torch.is_tensor(cur_x) else None}")
            raise

    def costFunc_navgtv(self, simPoints, callPoints, third_vars, cur_x, cur_y, all_mat1params,
                       all_mat2params, all_param0s, all_paramNs, all_fields, all_tms,
                       all_global_bcs, all_cf_factors, all_scheduler_factors, all_custom_input):
        """Cost function for non-averaged transmission/reflection
        cf_factors keys: [0: T linear, 1: R linear, 2: T log, 3: R log, 4: A linear, 5: [T-target,cost], 6: [R-Target, cost]]
        """
        # Convert inputs to PyTorch tensors with consistent dtype
        L = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # Vectorized operations for cost function calculation
        for n, (fields, cf_factors) in enumerate(zip(all_fields, all_cf_factors)):
            # Calculate transmission for each field
            fields_tensor = torch.tensor(fields, dtype=torch.complex64, device=self.device)
            T_vals = self.T(fields_tensor)

            # Convert dictionary keys to integers and sort them
            sorted_keys = sorted([int(k) if isinstance(k, str) else k for k in cf_factors.keys()])
            # Get values in sorted key order
            weights = torch.tensor([cf_factors[str(k) if isinstance(k, str) else k] for k in sorted_keys],
                                dtype=torch.float32, device=self.device)

            # Apply weights and sum
            L += torch.sum(weights * T_vals)

        # Return negative cost as tensor (don't convert to float yet)
        return -L

    def d_costFunc_fields(self, simPoints, callPoints, third_vars, cur_x, cur_y, all_mat1params,
                         all_mat2params, all_param0s, all_paramNs, all_fields, all_tms,
                         all_global_bcs, all_cf_factors, all_scheduler_factors, all_custom_input,
                         costFunc_outputs, chosen_simPoints):
        """Derivative of cost function with respect to fields"""
        dLde = []

        # Vectorized operations for field derivatives
        for n, (fields, cf_factors) in enumerate(zip(all_fields, all_cf_factors)):
            # Calculate field derivatives
            fields_tensor = torch.tensor(fields, dtype=torch.complex64, device=self.device)
            dT = torch.stack([self.dTde(e) for e in fields_tensor])

            # Convert dictionary keys to integers and sort them
            sorted_keys = sorted([int(k) if isinstance(k, str) else k for k in cf_factors.keys()])
            # Get values in sorted key order with consistent dtype
            weights = torch.tensor([cf_factors[str(k) if isinstance(k, str) else k] for k in sorted_keys],
                               dtype=torch.float32, device=self.device)

            # Apply weights with proper broadcasting
            weighted_dT = -weights.unsqueeze(1) * dT

            dLde.append(weighted_dT.cpu().tolist())

        return dLde

    def d_costFunc_phi(self, simPoints, callPoints, third_vars, cur_x, cur_y, all_mat1params,
                      all_mat2params, all_param0s, all_paramNs, all_fields, all_tms,
                      all_global_bcs, all_cf_factors, all_scheduler_factors, all_custom_input,
                      costFunc_outputs, chosen_simPoints):
        """Derivative of cost function with respect to optimization parameters"""
        # Convert inputs to tensors
        cur_x = torch.tensor(cur_x, dtype=torch.float64, device=self.device, requires_grad=True)
        cur_y = torch.tensor(cur_y, dtype=torch.float64, device=self.device, requires_grad=True)

        # Initialize gradient tensor
        dLdphi = torch.zeros(len(cur_x) + len(cur_y), dtype=torch.float64, device=self.device)

        # Calculate gradients using automatic differentiation
        for n, (fields, cf_factors) in enumerate(zip(all_fields, all_cf_factors)):
            weights = torch.tensor(list(cf_factors.values()), dtype=torch.float64, device=self.device)
            T_vals = torch.stack([self.T(torch.tensor(e, dtype=torch.complex64, device=self.device)) for e in fields])
            weighted_T = weights * T_vals

            # Accumulate gradients
            if torch.is_grad_enabled():
                grads = torch.autograd.grad(weighted_T.sum(), [cur_x, cur_y], allow_unused=True)
                if grads[0] is not None:
                    dLdphi[:len(cur_x)] += grads[0]
                if grads[1] is not None:
                    dLdphi[len(cur_x):] += grads[1]

        return -dLdphi.cpu().tolist()

    def Twrap(self, e):
        """Wrapped transmission"""
        return torch.clamp(self.T(e), min=0.0, max=1.0)

    def Rwrap(self, e):
        """Wrapped reflection"""
        return torch.clamp(self.R(e), min=0.0, max=1.0)

    def indicators(self, simPoints, callPoints, third_vars, cur_x, cur_y, all_mat1params,
                  all_mat2params, all_param0s, all_paramNs, all_fields, all_tms,
                  all_global_bcs, all_cf_factors, all_scheduler_factors, all_custom_input):
        """Calculate performance indicators"""
        # Initialize indicators dictionary
        indicators = {}

        # Convert fields to tensors for vectorized operations
        all_fields_t = [torch.tensor(fields, dtype=torch.complex64, device=self.device) for fields in all_fields]

        # Calculate average transmission and reflection
        T_avg = torch.mean(torch.stack([
            torch.mean(torch.stack([self.Twrap(e) for e in fields]))
            for fields in all_fields_t
        ]))

        R_avg = torch.mean(torch.stack([
            torch.mean(torch.stack([self.Rwrap(e) for e in fields]))
            for fields in all_fields_t
        ]))

        indicators['transmission'] = T_avg.cpu().item()
        indicators['reflection'] = R_avg.cpu().item()

        return indicators

    def interpolate(self, simPoints, callPoints, third_vars, cur_x, cur_y, all_mat1params,
                   all_mat2params, all_param0s, all_paramNs, all_fields, all_tms,
                   all_global_bcs, all_cf_factors, all_scheduler_factors, all_custom_input):
        """Interpolate fields between simulation points"""
        # Convert inputs to PyTorch tensors
        simPoints_t = torch.tensor(simPoints, dtype=torch.float64, device=self.device)
        callPoints_t = torch.tensor(callPoints, dtype=torch.float64, device=self.device)

        # Calculate transmission and reflection for all points
        all_fields_t = [torch.tensor(fields, dtype=torch.complex64, device=self.device) for fields in all_fields]
        T_vals = torch.stack([self.Twrap(fields[0]) for fields in all_fields_t])
        R_vals = torch.stack([self.Rwrap(fields[0]) for fields in all_fields_t])

        # Use PyTorch's interpolation
        T_interp = torch.nn.functional.interpolate(
            T_vals.unsqueeze(0).unsqueeze(0),
            size=len(callPoints),
            mode='linear',
            align_corners=True
        ).squeeze()

        R_interp = torch.nn.functional.interpolate(
            R_vals.unsqueeze(0).unsqueeze(0),
            size=len(callPoints),
            mode='linear',
            align_corners=True
        ).squeeze()

        return T_interp.cpu().tolist(), R_interp.cpu().tolist()
