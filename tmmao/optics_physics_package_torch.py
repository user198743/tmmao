import torch
from misc_utils import comp_utils
from copy import deepcopy
from math import copysign
import cmath as cm

class optics_tmm(comp_utils):
    def __init__(self):
        self.log_cap = -400
        self.n_is_optParam = False
        self.log_ratio = False
        # Initialize zero matrix as torch tensor
        self.zm = torch.zeros((2, 2), dtype=torch.complex64)
        # Complex initial conditions
        self.a0, self.bN = 1, torch.complex(torch.tensor(0.0), torch.tensor(0.0))
        self.kx = 1
        self.avg_pol = False
        self.subLen = 0.5
        return

    def get_name(self):
        return 'optics'

    def global_bcs(self, param0, paramN, sim_params, mat1params={}, mat2params={}):
        # Convert numpy arrays to torch tensors with complex support
        A = torch.tensor([[1.0 + 0j, 0], [0, 0]], dtype=torch.complex64)
        C = torch.tensor([self.a0, self.bN], dtype=torch.complex64)
        B = torch.tensor([[0.0 + 0j, 0], [0, 1]], dtype=torch.complex64)
        return (A, B, C)

    def left_tm(self, yp1, sim_params, matLeft={}, mat1Right={}, mat2Right={}):
        self.kx = torch.sin(torch.tensor(sim_params['third_vars']['incidentAngle']))
        if self.n_is_optParam:
            y = matLeft['refractiveIndex']
        else:
            y = 1
        tm, info = self.interface_tm(0, y, yp1, sim_params, {'kx': self.kx}, matLeft, matLeft, mat1Right, mat2Right)
        return tm, info

    def d_left_tm(self, yp1, sim_params, tracked_info, tm, matLeft={}, mat1Right={}, mat2Right={}):
        if self.n_is_optParam:
            y = matLeft['refractiveIndex']
        else:
            y = 1
        dx, dy, dyp1 = self.d_interface_tm(0, y, yp1, sim_params, tracked_info, tm, matLeft, matLeft, mat1Right, mat2Right)
        return dyp1

    def interface_tm(self, x, y, yp1, sim_params, tracked_info, mat1Left={}, mat2Left={}, mat1Right={}, mat2Right={}):
        kx = tracked_info['kx']
        if not self.n_is_optParam:
            if self.log_ratio:
                y = (100**y - 1)/99
                yp1 = (100**yp1 - 1)/99
            n1L, n2L = mat1Left['refractiveIndex'], mat2Left['refractiveIndex']
            n1R, n2R = mat1Right['refractiveIndex'], mat2Right['refractiveIndex']
            nL = y * n2L + (1-y) * n1L
            nR = yp1 * n2R + (1-yp1) * n1R
        else:
            nL, nR = y, yp1

        # Convert to torch tensors and compute
        epL = torch.tensor(nL**2, dtype=torch.complex64)
        epR = torch.tensor(nR**2, dtype=torch.complex64)
        kx_t = torch.tensor(kx, dtype=torch.complex64)

        # Complex square root using torch
        kzL = torch.sqrt(epL - kx_t**2)
        kzR = torch.sqrt(epR - kx_t**2)

        t, r = self.get_tr(kzL, kzR, torch.tensor(nL), torch.tensor(nR), sim_params['third_vars']['polarization'])

        # Phase calculations
        phase = 1j * kzL * (2 * torch.pi / sim_params['simPoint']) * x
        p = torch.tensor([[torch.exp(-phase), 0], [0, torch.exp(phase)]], dtype=torch.complex64)
        m = (1/t) * torch.tensor([[1, r], [r, 1]], dtype=torch.complex64)

        return torch.matmul(p, m), {'kx': kx, 'p': p, 'm': m}

    def get_tr(self, kzL, kzR, nL, nR, pol):
        epL = nL**2
        epR = nR**2
        if pol == 'p':
            d = epL * kzR + kzL * epR
            t = 2 * torch.sqrt(epL * epR) * kzL / d
            r = (epR * kzL - epL * kzR) / d
        else:
            d = kzR + kzL
            t = 2 * kzL / d
            r = (kzL - kzR) / d
        return t, r

    def right_tm(self, x, y, sim_params, tracked_info, mat1Left={}, mat2Left={}, matRight={}):
        tm, info = self.interface_tm(x, y, 1, sim_params, tracked_info, mat1Left, mat2Left, matRight, matRight)
        return tm

    def d_right_tm(self, x, y, sim_params, tracked_info, tm, mat1Left={}, mat2Left={}, matRight={}):
        dx, dy, _ = self.d_interface_tm(x, y, 1, sim_params, tracked_info, tm, mat1Left, mat2Left, matRight, matRight)
        return dx, dy

    def tm(self, x, xp1, y, yp1, sim_params, tracked_info, mat1params={}, mat2params={}):
        if tracked_info is None or 'kx' not in tracked_info:
            tracked_info = tracked_info or {}
            tracked_info['kx'] = torch.sin(torch.tensor(sim_params['third_vars']['incidentAngle'], dtype=torch.float64))
        return self.interface_tm(x, y, yp1, sim_params, tracked_info, mat1params, mat2params, mat1params, mat2params)

    def dtm(self, x, y, yp1, sim_params, tracked_info, tm, mat1params={}, mat2params={}):
        return self.d_interface_tm(x, y, yp1, sim_params, tracked_info, tm, mat1params, mat2params, mat1params, mat2params)

    def d_interface_tm(self, x, y, yp1, sim_params, tracked_info, tm, mat1Left={}, mat2Left={}, mat1Right={}, mat2Right={}):
        # Initialize gradients
        dx = torch.zeros(2, 2, dtype=torch.complex64)
        dy = torch.zeros(2, 2, dtype=torch.complex64)
        dyp1 = torch.zeros(2, 2, dtype=torch.complex64)

        # Extract tracked info
        kx = tracked_info['kx']
        p = tracked_info['p']
        m = tracked_info['m']

        if not self.n_is_optParam:
            if self.log_ratio:
                y = (100**y - 1)/99
                yp1 = (100**yp1 - 1)/99
            n1L, n2L = mat1Left['refractiveIndex'], mat2Left['refractiveIndex']
            n1R, n2R = mat1Right['refractiveIndex'], mat2Right['refractiveIndex']
            nL = y * n2L + (1-y) * n1L
            nR = yp1 * n2R + (1-yp1) * n1R

            # Compute derivatives with respect to material parameters
            dnLdy = n2L - n1L
            dnRdyp1 = n2R - n1R
        else:
            nL, nR = y, yp1
            dnLdy = 1
            dnRdyp1 = 1

        # Convert to tensors
        nL_t = torch.tensor(nL, dtype=torch.complex64)
        nR_t = torch.tensor(nR, dtype=torch.complex64)
        kx_t = torch.tensor(kx, dtype=torch.complex64)

        # Compute wave vectors
        epL = nL_t**2
        epR = nR_t**2
        kzL = torch.sqrt(epL - kx_t**2)
        kzR = torch.sqrt(epR - kx_t**2)

        # Phase derivative
        dphase = 1j * kzL * (2 * torch.pi / sim_params['simPoint'])
        dp = torch.tensor([[-torch.exp(-dphase), 0], [0, torch.exp(dphase)]], dtype=torch.complex64)
        dx = torch.matmul(dp, m)

        # Material derivatives
        if sim_params['third_vars']['polarization'] == 'p':
            d = epL * kzR + kzL * epR
            dt_dy = 2 * torch.sqrt(epL * epR) * kzL * dnLdy / d
            dr_dy = (epR * kzL * dnLdy - epL * kzR * dnLdy) / d
            dt_dyp1 = 2 * torch.sqrt(epL * epR) * kzL * dnRdyp1 / d
            dr_dyp1 = (epR * kzL * dnRdyp1 - epL * kzR * dnRdyp1) / d
        else:
            d = kzR + kzL
            dt_dy = 2 * kzL * dnLdy / d
            dr_dy = (kzL * dnLdy - kzR * dnLdy) / d
            dt_dyp1 = 2 * kzL * dnRdyp1 / d
            dr_dyp1 = (kzL * dnRdyp1 - kzR * dnRdyp1) / d

        # Construct derivative matrices
        dm_dy = (1/dt_dy) * torch.tensor([[1, dr_dy], [dr_dy, 1]], dtype=torch.complex64)
        dm_dyp1 = (1/dt_dyp1) * torch.tensor([[1, dr_dyp1], [dr_dyp1, 1]], dtype=torch.complex64)

        dy = torch.matmul(p, dm_dy)
        dyp1 = torch.matmul(p, dm_dyp1)

        return dx, dy, dyp1

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
