#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import local modules
from agnostic_director_torch import agnostic_director
from optics_physics_package_torch import optics_tmm
from agnostic_linear_adjoint_torch import agnostic_linear_adjoint, agnostic_linear_tmm
from agnostic_invDes_torch import agnostic_invDes
import materials_library

# Set up matplotlib for non-interactive backend
plt.switch_backend('agg')

# Convert initial MOE data to tensors on the specified device
xMOE = torch.tensor([1.3,1.325,1.35,1.375,1.4,1.425,1.45,1.475,1.5,1.525,1.55,1.575,1.6,1.625,1.65,1.675,1.7,1.725,1.75,1.775,1.8,1.825,1.85,1.875,1.9,1.925,1.95,1.975,2,2.025,2.05,2.075,2.1,2.125,2.15,2.175,2.2,2.225,2.25,2.275,2.3,2.325,2.35,2.375,2.4,2.425,2.45,2.475,2.5], dtype=torch.float32)

yMOE = torch.tensor([0.218319151,0.433592405,0.429471486,0.973253696,2.827936251,6.212286227,10.47045032,14.72879358,19.205456,22.69901034,25.09944598,24.00570018,21.0555697,17.88765759,15.37515086,13.08096327,10.1308328,9.145933024,10.89051536,13.83733118,16.45716099,19.73248577,22.68028703,27.26615381,31.74290581,36.11045344,41.67924912,48.12159016,54.12711373,56.31048441,57.5109262,56.30860312,53.79573805,51.71915293,51.39010647,52.37115408,51.82378846,49.3108338,45.26937632,41.99252859,37.84177717,32.162075,25.49872725,21.12849206,19.70650606,21.34224237,27.34624299,34.00645525,42.85084447], dtype=torch.float32) * 1E-2

class MOEInterpolation:
    def __init__(self, device):
        self.device = device

    def __call__(self, x):
        """Interpolate MOE data with device support"""
        # Ensure input is a tensor on the correct device
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)

        # Find indices for interpolation
        idx = torch.searchsorted(xMOE.to(self.device), x)
        idx = torch.clamp(idx, 1, len(xMOE)-1)

        # Get surrounding points
        x0 = xMOE[idx-1].to(self.device)
        x1 = xMOE[idx].to(self.device)
        y0 = yMOE[idx-1].to(self.device)
        y1 = yMOE[idx].to(self.device)

        # Linear interpolation
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (x - x0)

def condition_checker_moe(simDict, device=None):
    """Check optimization conditions with device support"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameters = simDict['parameters']
    if not torch.is_tensor(parameters):
        parameters = torch.tensor(parameters, device=device)
    return torch.all(parameters >= 0).item()

class dyn_sched:
    def __init__(self):
        self.iteration = 0

    def dynamic_scheduler(self, simDict):
        """Dynamic scheduler with device support"""
        self.iteration += 1
        if self.iteration % 50 == 0:
            return True
        return False

def main(verbose=1, device=None):
    """Main function with PyTorch device support and verbosity control"""
    # Initialize device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize director with device
    director = agnostic_director(device=device)
    print(f"Director initialized with device: {director.device}")

    # Set up physics package
    physics = optics_tmm(device=device)
    print(f"Physics package initialized with device: {physics.device}")

    # Initialize interpolation and target MOE data
    moe = MOEInterpolation(device=device)
    xMOE = torch.linspace(1.3, 2.5, 49, device=device)  # Match simRange
    yMOE = moe(xMOE)  # Get target MOE response

    # Set material functions
    def mat1(x):
        """Material 1 function with device support"""
        if isinstance(x, dict):
            sim_params = x.copy()
            sim_params['device'] = device
        else:
            sim_params = {
                'physics': 'optics',
                'simPoint': x,
                'device': device
            }
        return materials_library.silicon(sim_params)

    def mat2(x):
        """Material 2 function with device support"""
        if isinstance(x, dict):
            sim_params = x.copy()
            sim_params['device'] = device
        else:
            sim_params = {
                'physics': 'optics',
                'simPoint': x,
                'device': device
            }
        return materials_library.siliconDioxide(sim_params)

    def param0(x):
        """Parameter 0 function with device support"""
        if isinstance(x, dict):
            sim_params = x.copy()
            sim_params['device'] = device
        else:
            sim_params = {
                'physics': 'optics',
                'simPoint': x,
                'device': device
            }
        return materials_library.air(sim_params)

    def paramN(x):
        """Parameter N function with device support"""
        if isinstance(x, dict):
            sim_params = x.copy()
            sim_params['device'] = device
        else:
            sim_params = {
                'physics': 'optics',
                'simPoint': x,
                'device': device
            }
        return materials_library.silicon(sim_params)

    def customInput(x):
        return moe(x)

    # Set physics functions
    director.set_physics(physics, mat1, mat2, param0, paramN, customInput)

    # Set optimizer functions
    def cfFactor(iteration):
        return 1

    scheduler = dyn_sched()
    director.set_optimizerFunctions(cfFactor, scheduler.dynamic_scheduler)

    # Set structure parameters
    num_layers = 500
    initial_layer_thickness = torch.ones(num_layers, device=device) * 100E-9
    initial_ys = torch.ones(num_layers, device=device) * 0.5

    director.set_structure(
        simType='MOE',
        num_layers=num_layers,
        initial_layer_thickness=initial_layer_thickness,
        initial_ys=initial_ys,
        y_bounds=(0, 1),
        x_bounds=(50E-9, 150E-9),
        num_intervals=1000,
        substrates=['Si'],
        superstrates=['Air']
    )

    # Set simulation parameters
    simRange = torch.linspace(1.3, 2.5, 49, device=device)
    simResolution = 1
    third_variables = None

    # Initialize simulation in physics package
    physics.set_simulation(
        simRange=simRange,
        simResolution=simResolution,
        simScale='linear',
        third_variables=third_variables,
        logSim=False
    )

    # Run optimization
    result = director.run_optimization(
        simRange=simRange,
        simResolution=simResolution,
        third_variables=third_variables,
        ftol=1e-12,
        gtol=1e-12,
        simPoint=simRange[0],
        verbose=verbose
    )

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(xMOE.detach().cpu().numpy(), yMOE.detach().cpu().numpy(), 'b-', label='Target')
    plt.plot(simRange.detach().cpu().numpy(), physics.outputMOE.detach().cpu().numpy(), 'r--', label='Optimized')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Transmission')
    plt.legend()
    plt.grid(True)

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Save plot
    plt.savefig('output/optimization_results.png')
    plt.close()

    # Save MOE data
    np.savetxt('outputMoe.txt', physics.outputMOE.detach().cpu().numpy())

    return result

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = main(device=device)
    print(result)
