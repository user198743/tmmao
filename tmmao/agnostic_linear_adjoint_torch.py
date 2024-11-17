import torch

class agnostic_linear_tmm:
    """Solves an arbitrary linear tmm problem, given the list of transfer matrices."""

    def __init__(self, device='cpu'):
        """Initializes instance of agnostic_linear_tmm"""
        self.device = device
        return

    def solve_e(self, tms, A, B, C):
        """Solves for the field distribution using PyTorch operations."""
        e = []
        t_full = torch.eye(len(C), dtype=torch.complex64, device=self.device)
        for t in tms:
            t_full = torch.matmul(t_full, t)
        try:
            # Convert numpy arrays to PyTorch tensors if needed
            A_tensor = A if torch.is_tensor(A) else torch.as_tensor(A, dtype=torch.complex64, device=self.device)
            B_tensor = B if torch.is_tensor(B) else torch.as_tensor(B, dtype=torch.complex64, device=self.device)
            C_tensor = C if torch.is_tensor(C) else torch.as_tensor(C, dtype=torch.complex64, device=self.device)

            # Solve the system using PyTorch's solve
            e.append(torch.linalg.solve(torch.matmul(A_tensor, t_full) + B_tensor, C_tensor))
        except RuntimeError as err:
            print('The boundary conditions you entered do not have a solution. Error:', err)
            raise

        tms.reverse()
        for t in tms:
            e.append(torch.matmul(t, e[-1]))
        e.reverse()
        tms.reverse()
        return e


class agnostic_linear_adjoint(agnostic_linear_tmm):
    """Solves an arbitrary linear adjoint tmm problem using PyTorch operations."""

    def __init__(self, device='cpu'):
        """Initializes instance of agnostic_linear_adjoint"""
        super().__init__(device)
        self.ala_eadj = []
        self.ala_dLdphi = []
        self.ala_dLdx = []
        self.ala_nonzero_dTdphis = []
        self.ala_nonzero_dLdes = []
        self.ala_nonzero_dLdphis = []
        return

    def solve_adjoint(self, all_tms, all_fields, dLdphi, all_dLde, all_dTdphi, all_global_bc, nonzero_dTdphi=[]):
        """Solves for the adjoint fields and gradient using PyTorch operations."""
        del self.ala_eadj[:]
        del self.ala_dLdx[:]
        self.ala_all_tms = all_tms
        self.ala_all_fields = all_fields
        self.ala_dLdphi = dLdphi
        self.ala_all_dLde = all_dLde
        self.ala_all_dTdphi = all_dTdphi
        self.ala_all_global_bc = all_global_bc
        self._process_nonzeros(all_dTdphi, all_dLde, dLdphi, nonzero_dTdphi)
        self.ala_dim = len(self.ala_all_fields[0][0])
        self._solve_eadj()
        self._solve_dLdx()
        return

    def _process_nonzeros(self, dTdphi, dLde, dLdphi, nonzero_dTdphi):
        """Filters out default values of solve_adjoint arguments."""
        if len(nonzero_dTdphi) == 0:
            for k in range(len(dTdphi)):
                self.ala_nonzero_dTdphis.append([])
                use_dtset = dTdphi[k]
                num_dTs = len(use_dtset[0])
                for phi_set in use_dtset:
                    self.ala_nonzero_dTdphis[-1].append(set(range(num_dTs)))
        elif not hasattr(nonzero_dTdphi[0][0], '__iter__'):
            self.ala_nonzero_dTdphis = nonzero_dTdphi * len(dTdphi)
        else:
            self.ala_nonzero_dTdphis = nonzero_dTdphi
        return

    def _solve_eadj(self):
        """Solves for adjoint fields using PyTorch operations."""
        for n in range(len(self.ala_all_tms)):
            self.ala_eadj.append([])
            tmsr = reversed(list(self.ala_all_tms[n]))
            dLdesr = list(reversed(list(self.ala_all_dLde[n])))
            A, B, C = self.ala_all_global_bc[n]

            # Convert to PyTorch tensors if needed
            A = A if torch.is_tensor(A) else torch.as_tensor(A, dtype=torch.complex64, device=self.device)
            B = B if torch.is_tensor(B) else torch.as_tensor(B, dtype=torch.complex64, device=self.device)

            t_full = torch.eye(self.ala_dim, dtype=torch.complex64, device=self.device)
            Lsum = torch.zeros(self.ala_dim, dtype=torch.complex64, device=self.device)

            j = 1
            for t in tmsr:
                t_full = torch.matmul(t, t_full)
                Lsum += torch.matmul(t_full.t(), -1 * dLdesr[j])
                j += 1

            Lsum += -1 * dLdesr[0]
            a = torch.matmul(A, t_full).t() + B.t()

            try:
                self.ala_eadj[-1].append(torch.linalg.solve(a, Lsum))
            except RuntimeError as err:
                print('The boundary conditions you entered do not have a solution. Error:', err)
                raise

            tms = list(self.ala_all_tms[n])
            dLdes = list(self.ala_all_dLde[n])
            self.ala_eadj[-1].append(torch.matmul(A.t(), self.ala_eadj[-1][-1]) + dLdes[0])

            i = 1
            for t in tms[:-1]:
                self.ala_eadj[-1].append(dLdes[i] + torch.matmul(t.t(), self.ala_eadj[-1][-1]))
                i += 1
        return

    def _solve_dLdx(self):
        """Solves for gradient using PyTorch operations."""
        for m in range(len(self.ala_dLdphi)):
            self.ala_dLdx.append(0)

        for n in range(len(self.ala_eadj)):
            eadj = self.ala_eadj[n]
            e = self.ala_all_fields[n]

            for m in range(len(self.ala_dLdphi)):
                d_sum = torch.tensor(0.0, dtype=torch.complex64, device=self.device)
                dTdphi = self.ala_all_dTdphi[n][m]
                nonzero_ds = self.ala_nonzero_dTdphis[n][m]

                for j in nonzero_ds:
                    d_sum += torch.matmul(eadj[j+1], torch.matmul(dTdphi[j], e[j+1]))

                self.ala_dLdx[m] += d_sum

        for m in range(len(self.ala_dLdphi)):
            self.ala_dLdx[m] = 2 * torch.real(self.ala_dLdx[m]) + self.ala_dLdphi[m]

        return
