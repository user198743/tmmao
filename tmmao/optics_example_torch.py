import torch
import os
from agnostic_director import agnostic_director
from optics_physics_package_torch import optics_tmm  # Use PyTorch version
import materials_library
import matplotlib.pyplot as plt

def main(device=None):
    # Set default device if none specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize scheduler
    ds = dyn_sched()

    # Initialize director
    ad = agnostic_director()

    # Initial conditions - convert to torch tensor
    x0 = torch.tensor([0.32441199332467563, 0.2421437838701929, 0.47340025698204713,
                      0.4825611995119943, 0.19989972134583547, 0.14736970395447987,
                      0.10213421303000284, 0.16361662441880945, 0.2796481376976949,
                      0.10995312837538893, 0.5725525242494137, 0.19900440425370333],
                     dtype=torch.float64, device=device)

    # Load physics functions with device
    physics = optics_tmm(device=device)  # Pass device to physics package
    ad.set_physics(physics_package=physics,
                  mat1Call=materials_library.siliconDioxide,
                  mat2Call=materials_library.silicon,
                  param0Call=materials_library.air,
                  paramNCall=materials_library.bk7,
                  customInputCall=materials_library.air)

    # Load optimizer functions
    ad.set_optimizerFunctions(cfFactorCall=condition_checker_moe,
                             schedulerCall=ds.dynamic_scheduler,
                             dynamic_scheduling=True,
                             paramPrunerCall=None)

    # Initialize structure
    ad.set_structure(simType='independent',
                    num_layers=12,
                    initial_layer_thickness=x0.cpu().tolist(),  # Convert tensor to CPU before converting to list
                    initial_ys=[1,0]*6,
                    y_bounds=[0,1],
                    x_bounds=[0.005,1],
                    num_intervals=1,
                    substrates=[],superstrates=[])

    # Define discretization map
    discretization_map = [
        {'var_x':True,'var_y':False,'scramble_x':[False,0.02],'scramble_y':[False,0.01],
         'merge_neighbors':[False,0.05],'fully_discretize':False,'purge_ablated':[False,0.01]},
        {'var_x':True,'var_y':False,'scramble_x':[False,0.02],'scramble_y':[False,0.01],
         'merge_neighbors':[False,0.05],'fully_discretize':False,'purge_ablated':[True,0.01]}
    ]

    # Run optimization
    ad.run_optimization(simRange=[1.3,2.5],
                       simResolution=60,
                       third_variables=[{'incidentAngle':0,'polarization':'s'}],
                       discretization_map=discretization_map,
                       simScale=1E-6,
                       ftol=1E-9,
                       gtol=1E-5,
                       save_name='Moe',
                       save_directory=os.path.join(os.getcwd(), 'output'),
                       save=True,
                       logSim=False,
                       verbose=1,
                       evo_preamble_tag='',
                       presearch_threshold=float('inf'))  # Replace np.inf with float('inf')

    # Analysis and plotting
    ad.print_evo_info(sparse=True)

    # Initialize analysis
    ad.init_analysis(simRange=[1.3,2.5],
                    simResolution=200,
                    simScale=1E-6,
                    simType='independent',
                    third_variables=[{'incidentAngle':0,'polarization':'s'}],
                    logSim=False,
                    substrates=[],superstrates=[],
                    num_layers=None)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize plot
    ad.init_plot(save_name='MoeTableau',
                 save_dir=output_dir,
                 save=True,
                 show=True,
                 size=(6,10),
                 dpi=300,
                 dark=False,
                 savetype='.png',
                 bbox_inches=None,
                 tight_layout=True,
                 subplots_adjust={'left':None,'right':None,'top':None,'bottom':None,
                                'hspace':None,'wspace':None},
                 sharex=False,
                 height_ratios=[3,1,1,3,3])

    # Add subplots
    ad.add_subplotCostFunction(plot_grand_iters=False,
                              lineColor='C2',
                              linestyle='-',
                              linewidth=1,
                              axiswidth=0.5,
                              tickwidth=0.5,
                              ticklength=1.5,
                              xlabel='Iterations',
                              ylabel='Cost function',
                              labelfontsize=7,
                              tickfontsize=7,
                              logy=True)

    # Save plots
    plt.savefig(os.path.join(output_dir, 'optimization_results.png'))
    plt.close()

class dyn_sched:
    def __init__(self):
        pass

    def dynamic_scheduler(self, L_prev, L_cur, its, x, y, scheduler_factors):
        return scheduler_factors

def condition_checker_moe(sim_params):
    return {'transmission': 1.0}

if __name__ == "__main__":
    main()  # Uses default device (CUDA if available, else CPU)
