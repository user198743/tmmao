import torch
import numpy as np
from math import pi
from cmath import exp, sqrt
import torch.nn.functional as F
from misc_utils import comp_utils

class optics_tmm(comp_utils):
    def __init__(self):
        super().__init__()
        self.n_is_optParam = False
        self.log_ratio = False
        self.kx = 0.0
        self.log_cap = -400
        self.zm = torch.zeros((2, 2), dtype=torch.complex64)
        self.a0 = 1
        self.bN = torch.complex(torch.tensor(0.0), torch.tensor(0.0))
        return

    def get_name(self):
        return 'optics'

    def global_bcs(self, param0, paramN, sim_params, mat1params={}, mat2params={}):
        """Return global boundary conditions for optimization"""
        # Initialize kx from sim_params if available
        kx = 0.0
        if 'third_vars' in sim_params:
            if isinstance(sim_params['third_vars'], list):
                if len(sim_params['third_vars']) > 0:
                    kx = torch.sin(torch.tensor(sim_params['third_vars'][0].get('incidentAngle', 0.0)))
            else:
                kx = torch.sin(torch.tensor(sim_params['third_vars'].get('incidentAngle', 0.0)))
        self.kx = kx
        return {'kx': self.kx}

    def left_tm(self, yp1, sim_params, matLeft={}, mat1Right={}, mat2Right={}):
        """Calculate transfer matrix for left boundary"""
        # Initialize transfer matrices list if not already done
        if not hasattr(self, 'all_tms'):
            self.all_tms = [[]]

        if self.n_is_optParam:
            y = matLeft.get('refractiveIndex', 1.0)
        else:
            y = 1.0

        # Get the transfer matrix
        tm, info = self.interface_tm(0, y, yp1, sim_params, {'kx': self.kx}, matLeft, matLeft, mat1Right, mat2Right)

        # Convert to numpy for compatibility
        tm_np = tm.detach().numpy()

        # Store in the transfer matrices list
        if len(self.all_tms) == 0 or not isinstance(self.all_tms[-1], list):
            self.all_tms.append([])
        self.all_tms[-1].append(tm_np)

        return tm_np, info

    def d_left_tm(self, yp1, sim_params, tracked_info, tm, matLeft={}, mat1Right={}, mat2Right={}):
        """Calculate derivative of left transfer matrix"""
        if self.n_is_optParam:
            y = matLeft.get('refractiveIndex', 1.0)
        else:
            y = 1.0
        dx, dy, dyp1 = self.d_interface_tm(0, y, yp1, sim_params, tracked_info, tm, matLeft, matLeft, mat1Right, mat2Right)
        return dyp1

    # Override comp_utils methods for PyTorch compatibility
    def mul(self, a, b):
        """Multiply two matrices, handling both numpy arrays and PyTorch tensors"""
        if torch.is_tensor(a) and torch.is_tensor(b):
            return torch.matmul(a, b)
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return super().mul(a, b)
        else:
            # Convert to PyTorch tensors if mixed types
            a_tensor = torch.tensor(a, dtype=torch.complex64) if not torch.is_tensor(a) else a
            b_tensor = torch.tensor(b, dtype=torch.complex64) if not torch.is_tensor(b) else b
            return torch.matmul(a_tensor, b_tensor)

    def div(self, a, b):
        """Divide two values, handling both numpy arrays and PyTorch tensors"""
        if torch.is_tensor(a) and torch.is_tensor(b):
            return a / b
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return super().div(a, b)
        else:
            # Convert to PyTorch tensors if mixed types
            a_tensor = torch.tensor(a, dtype=torch.complex64) if not torch.is_tensor(a) else a
            b_tensor = torch.tensor(b, dtype=torch.complex64) if not torch.is_tensor(b) else b
            return a_tensor / b_tensor

    def interface_tm(self, x, y, yp1, sim_params, tracked_info, mat1Left={}, mat2Left={}, mat1Right={}, mat2Right={}):
        """Calculate transfer matrix for interface"""
        # Debug logging
        print("=== Debug Info ===")
        print(f"x: {x}")
        print(f"y: {y}")
        print(f"yp1: {yp1}")
        print(f"sim_params: {sim_params}")
        print(f"mat1Left: {mat1Left}")
        print(f"mat2Left: {mat2Left}")
        print("================")

        # Set default refractive indices if not provided
        default_mat = {'refractiveIndex': 1.0}  # Default to air
        mat1Left = mat1Left or default_mat
        mat2Left = mat2Left or default_mat
        mat1Right = mat1Right or default_mat
        mat2Right = mat2Right or default_mat

        # Get kx from tracked_info and convert to tensor
        kx = tracked_info.get('kx', 0.0)
        kx_t = torch.tensor(kx, dtype=torch.complex64) if not torch.is_tensor(kx) else kx

        # Handle log_ratio and n_is_optParam cases
        if not self.n_is_optParam:
            if self.log_ratio:
                y_val = self.safe_extract_value(y)
                yp1_val = self.safe_extract_value(yp1)
                y = torch.tensor((100**y_val - 1)/99, dtype=torch.float32)
                yp1 = torch.tensor((100**yp1_val - 1)/99, dtype=torch.float32)
            else:
                y = torch.tensor(self.safe_extract_value(y), dtype=torch.float32)
                yp1 = torch.tensor(self.safe_extract_value(yp1), dtype=torch.float32)

            # Get refractive indices and interpolate
            n1L = torch.tensor(self.safe_extract_value(mat1Left), dtype=torch.complex64)
            n2L = torch.tensor(self.safe_extract_value(mat2Left), dtype=torch.complex64)
            n1R = torch.tensor(self.safe_extract_value(mat1Right), dtype=torch.complex64)
            n2R = torch.tensor(self.safe_extract_value(mat2Right), dtype=torch.complex64)

            # Interpolate refractive indices
            nL = y * n2L + (1-y) * n1L
            nR = yp1 * n2R + (1-yp1) * n1R
        else:
            nL = torch.tensor(self.safe_extract_value(y), dtype=torch.complex64)
            nR = torch.tensor(self.safe_extract_value(yp1), dtype=torch.complex64)

        # Calculate wave vectors
        sim_point = self.get_sim_point(sim_params)
        k0 = 2 * pi * sim_point

        # Calculate kz values using PyTorch operations
        kzL = torch.sqrt(nL**2 * k0**2 - kx_t**2 + 0j)
        kzR = torch.sqrt(nR**2 * k0**2 - kx_t**2 + 0j)

        # Get transmission and reflection coefficients
        t, r = self.get_tr(kzL, kzR, nL, nR, self.get_polarization(sim_params))

        # Build transfer matrix using PyTorch operations
        tm = torch.zeros((2, 2), dtype=torch.complex64)
        tm[0, 0] = 1/t
        tm[0, 1] = r/t
        tm[1, 0] = r/t
        tm[1, 1] = 1/t

        # Update tracked info
        tracked_info.update({
            'kzL': kzL,
            'kzR': kzR,
            'nL': nL,
            'nR': nR,
            't': t,
            'r': r,
            'k0': k0
        })

        return tm, tracked_info

    def get_tr(self, kzL, kzR, nL, nR, pol):
        """Calculate transmission and reflection coefficients"""
        # Convert inputs to PyTorch tensors if they aren't already
        kzL = torch.as_tensor(kzL, dtype=torch.complex64)
        kzR = torch.as_tensor(kzR, dtype=torch.complex64)
        nL = torch.as_tensor(nL, dtype=torch.complex64)
        nR = torch.as_tensor(nR, dtype=torch.complex64)

        # Calculate permittivities
        epL = nL**2
        epR = nR**2

        if pol == 'p':  # TM polarization
            d = epL * kzR + kzL * epR
            t = 2 * torch.sqrt(epL * epR) * kzL / d
            r = (epR * kzL - epL * kzR) / d
        else:  # TE polarization (s)
            d = kzR + kzL
            t = 2 * kzL / d
            r = (kzL - kzR) / d

        # Convert PyTorch tensors to numpy arrays
        return t.detach().numpy(), r.detach().numpy()

    def right_tm(self, x, y, sim_params, tracked_info, mat1Left={}, mat2Left={}, matRight={}):
        # Initialize transfer matrices list if not already done
        if not hasattr(self, 'all_tms'):
            self.all_tms = [[]]

        # Get the transfer matrix
        tm, info = self.interface_tm(x, y, 1, sim_params, tracked_info, mat1Left, mat2Left, matRight, matRight)

        # Convert to numpy for compatibility with the rest of the system
        tm_np = tm.detach().numpy()

        # Store in the transfer matrices list
        if len(self.all_tms) == 0 or not isinstance(self.all_tms[-1], list):
            self.all_tms.append([])
        self.all_tms[-1].append(tm_np)

        return tm_np

    def d_right_tm(self, x, y, sim_params, tracked_info, tm, mat1Left={}, mat2Left={}, matRight={}):
        dx, dy, _ = self.d_interface_tm(x, y, 1, sim_params, tracked_info, tm, mat1Left, mat2Left, matRight, matRight)
        return dx, dy

    def tm(self, x, xp1, y, yp1, sim_params, tracked_info, mat1params={}, mat2params={}):
        if tracked_info is None or 'kx' not in tracked_info:
            tracked_info = tracked_info or {}
            # Get incidentAngle from sim_params['third_vars'] list of dicts
            incident_angle = sim_params.get('third_vars', [{'incidentAngle': 0}])[0].get('incidentAngle', 0)
            tracked_info['kx'] = torch.sin(torch.tensor(incident_angle, dtype=torch.float64))
        return self.interface_tm(x, y, yp1, sim_params, tracked_info, mat1params, mat2params, mat1params, mat2params)

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
        kx_t = torch.tensor(kx, dtype=torch.complex64) if not torch.is_tensor(kx) else kx

        # Handle log_ratio and n_is_optParam cases
        if not self.n_is_optParam:
            if self.log_ratio:
                y = torch.tensor((100**float(y) - 1)/99, dtype=torch.float32)
                yp1 = torch.tensor((100**float(yp1) - 1)/99, dtype=torch.float32)

            # Get refractive indices and interpolate
            n1L = torch.tensor(mat1Left['refractiveIndex'], dtype=torch.complex64)
            n2L = torch.tensor(mat2Left['refractiveIndex'], dtype=torch.complex64)
            n1R = torch.tensor(mat1Right['refractiveIndex'], dtype=torch.complex64)
            n2R = torch.tensor(mat2Right['refractiveIndex'], dtype=torch.complex64)

            # Calculate derivatives of refractive indices
            dnL_dy = n2L - n1L
            dnR_dyp1 = n2R - n1R

            # Calculate wave vectors and their derivatives
            k0 = 2 * torch.pi / torch.tensor(float(sim_params['simPoint']), dtype=torch.complex64)
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
            k0 = 2 * torch.pi / torch.tensor(float(sim_params['simPoint']), dtype=torch.complex64)
            kL = k0 * torch.tensor(float(y), dtype=torch.complex64)
            kR = k0 * torch.tensor(float(yp1), dtype=torch.complex64)

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
        dt_dx = torch.zeros_like(tm, dtype=torch.complex64)
        dt_dy = torch.zeros_like(tm, dtype=torch.complex64)
        dt_dyp1 = torch.zeros_like(tm, dtype=torch.complex64)

        # Phase contribution
        dt_dx[0, 0] = -dphase_dx * tm[0, 0]
        dt_dx[1, 1] = dphase_dx * tm[1, 1]

        # Material parameter contributions
        dt_dy[0, 0] = -dphase_dy * tm[0, 0] + (1/dt_dy) * torch.tensor([[1, dr_dy], [dr_dy, 1]], dtype=torch.complex64)[0, 0]
        dt_dy[0, 1] = (1/dt_dy) * torch.tensor([[1, dr_dy], [dr_dy, 1]], dtype=torch.complex64)[0, 1]
        dt_dy[1, 0] = (1/dt_dy) * torch.tensor([[1, dr_dy], [dr_dy, 1]], dtype=torch.complex64)[1, 0]
        dt_dy[1, 1] = dphase_dy * tm[1, 1] + (1/dt_dy) * torch.tensor([[1, dr_dy], [dr_dy, 1]], dtype=torch.complex64)[1, 1]

        dt_dyp1[0, 0] = (1/dt_dyp1) * torch.tensor([[1, dr_dyp1], [dr_dyp1, 1]], dtype=torch.complex64)[0, 0]
        dt_dyp1[0, 1] = (1/dt_dyp1) * torch.tensor([[1, dr_dyp1], [dr_dyp1, 1]], dtype=torch.complex64)[0, 1]
        dt_dyp1[1, 0] = (1/dt_dyp1) * torch.tensor([[1, dr_dyp1], [dr_dyp1, 1]], dtype=torch.complex64)[1, 0]
        dt_dyp1[1, 1] = (1/dt_dyp1) * torch.tensor([[1, dr_dyp1], [dr_dyp1, 1]], dtype=torch.complex64)[1, 1]

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
        """Absorption"""
        return 1 - (torch.abs(e[0])**2 + torch.abs(e[1])**2)

    def T(self, e):
        """Transmission"""
        return torch.abs(e[0])**2

    def R(self, e):
        """Reflection"""
        return torch.abs(e[1])**2

    def dTde(self, e):
        """Derivative of transmission with respect to field"""
        return torch.tensor([2 * torch.conj(e[0]), 0], dtype=torch.complex64)

    def dRde(self, e):
        """Derivative of reflection with respect to field"""
        return torch.tensor([0, 2 * torch.conj(e[1])], dtype=torch.complex64)

    def dAde(self, e):
        """Derivative of absorption with respect to field"""
        return -1 * (self.dTde(e) + self.dRde(e))

    def dRdbde(self, e):
        """Derivative of reflection in dB with respect to field"""
        return 20 / (torch.log(torch.tensor(10.0)) * torch.abs(e[1])) * torch.tensor(
            [0, torch.conj(e[1])/torch.abs(e[1])], dtype=torch.complex64)

    def dTdbde(self, e):
        """Derivative of transmission in dB with respect to field"""
        return 20 / (torch.log(torch.tensor(10.0)) * torch.abs(e[0])) * torch.tensor(
            [torch.conj(e[0])/torch.abs(e[0]), 0], dtype=torch.complex64)

    def costFunc(self, simPoints, callPoints, third_vars, cur_x, cur_y, all_mat1params, all_mat2params,
                all_param0s, all_paramNs, all_fields, all_tms, all_global_bcs, all_cf_factors,
                all_scheduler_factors, all_custom_input):
        """Main cost function"""
        if self.avg_pol:
            return self.costFunc_avgpol(simPoints, callPoints, third_vars, cur_x, cur_y,
                                      all_mat1params, all_mat2params, all_param0s, all_paramNs,
                                      all_fields, all_tms, all_global_bcs, all_cf_factors,
                                      all_scheduler_factors, all_custom_input)
        return self.costFunc_navgtv(simPoints, callPoints, third_vars, cur_x, cur_y,
                                  all_mat1params, all_mat2params, all_param0s, all_paramNs,
                                  all_fields, all_tms, all_global_bcs, all_cf_factors,
                                  all_scheduler_factors, all_custom_input)

    def costFunc_navgtv(self, simPoints, callPoints, third_vars, cur_x, cur_y, all_mat1params,
                       all_mat2params, all_param0s, all_paramNs, all_fields, all_tms,
                       all_global_bcs, all_cf_factors, all_scheduler_factors, all_custom_input):
        """Cost function for non-averaged transmission/reflection
        cf_factors keys: [0: T linear, 1: R linear, 2: T log, 3: R log, 4: A linear, 5: [T-target,cost], 6: [R-Target, cost]]
        """
        # Convert inputs to PyTorch tensors
        L = torch.tensor(0.0, dtype=torch.float64)

        # Vectorized operations for cost function calculation
        for n, (fields, cf_factors) in enumerate(zip(all_fields, all_cf_factors)):
            # Calculate transmission for each field
            T_vals = torch.stack([self.T(torch.tensor(e, dtype=torch.complex64)) for e in fields])

            # Convert dictionary keys to integers and sort them
            sorted_keys = sorted([int(k) if isinstance(k, str) else k for k in cf_factors.keys()])
            # Get values in sorted key order
            weights = torch.tensor([cf_factors[str(k) if isinstance(k, str) else k] for k in sorted_keys], dtype=torch.float64)

            # Apply weights and sum
            L += torch.sum(weights * T_vals)

        return -L.item()

    def d_costFunc_fields(self, simPoints, callPoints, third_vars, cur_x, cur_y, all_mat1params,
                         all_mat2params, all_param0s, all_paramNs, all_fields, all_tms,
                         all_global_bcs, all_cf_factors, all_scheduler_factors, all_custom_input,
                         costFunc_outputs, chosen_simPoints):
        """Derivative of cost function with respect to fields"""
        dLde = []

        # Vectorized operations for field derivatives
        for n, (fields, cf_factors) in enumerate(zip(all_fields, all_cf_factors)):
            # Calculate field derivatives
            dT = torch.stack([self.dTde(torch.tensor(e, dtype=torch.complex64)) for e in fields])

            # Apply weights
            weights = torch.tensor(list(cf_factors.values()), dtype=torch.float64)
            weighted_dT = -weights.unsqueeze(1) * dT

            dLde.append(weighted_dT.tolist())

        return dLde

    def d_costFunc_phi(self, simPoints, callPoints, third_vars, cur_x, cur_y, all_mat1params,
                      all_mat2params, all_param0s, all_paramNs, all_fields, all_tms,
                      all_global_bcs, all_cf_factors, all_scheduler_factors, all_custom_input,
                      costFunc_outputs, chosen_simPoints):
        """Derivative of cost function with respect to optimization parameters"""
        # Convert inputs to tensors
        cur_x = torch.tensor(cur_x, dtype=torch.float64, requires_grad=True)
        cur_y = torch.tensor(cur_y, dtype=torch.float64, requires_grad=True)

        # Initialize gradient tensor
        dLdphi = torch.zeros(len(cur_x) + len(cur_y), dtype=torch.float64)

        # Calculate gradients using automatic differentiation
        for n, (fields, cf_factors) in enumerate(zip(all_fields, all_cf_factors)):
            weights = torch.tensor(list(cf_factors.values()), dtype=torch.float64)
            T_vals = torch.stack([self.T(torch.tensor(e, dtype=torch.complex64)) for e in fields])
            weighted_T = weights * T_vals

            # Accumulate gradients
            if torch.is_grad_enabled():
                grads = torch.autograd.grad(weighted_T.sum(), [cur_x, cur_y], allow_unused=True)
                if grads[0] is not None:
                    dLdphi[:len(cur_x)] += grads[0]
                if grads[1] is not None:
                    dLdphi[len(cur_x):] += grads[1]

        return -dLdphi.tolist()

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
        all_fields_t = [torch.tensor(fields, dtype=torch.complex64) for fields in all_fields]

        # Calculate average transmission and reflection
        T_avg = torch.mean(torch.stack([
            torch.mean(torch.stack([self.Twrap(e) for e in fields]))
            for fields in all_fields_t
        ]))

        R_avg = torch.mean(torch.stack([
            torch.mean(torch.stack([self.Rwrap(e) for e in fields]))
            for fields in all_fields_t
        ]))

        indicators['transmission'] = T_avg.item()
        indicators['reflection'] = R_avg.item()

        return indicators

    def interpolate(self, simPoints, callPoints, third_vars, cur_x, cur_y, all_mat1params,
                   all_mat2params, all_param0s, all_paramNs, all_fields, all_tms,
                   all_global_bcs, all_cf_factors, all_scheduler_factors, all_custom_input):
        """Interpolate fields between simulation points"""
        # Convert inputs to PyTorch tensors
        simPoints_t = torch.tensor(simPoints, dtype=torch.float64)
        callPoints_t = torch.tensor(callPoints, dtype=torch.float64)

        # Calculate transmission and reflection for all points
        all_fields_t = [torch.tensor(fields, dtype=torch.complex64) for fields in all_fields]
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

        return T_interp.tolist(), R_interp.tolist()
