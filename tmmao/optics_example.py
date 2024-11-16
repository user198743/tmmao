#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Add the current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import local modules
from agnostic_director import agnostic_director
from optics_physics_package import optics_tmm
import materials_library

# Set up matplotlib for non-interactive backend
plt.switch_backend('agg')

# Define interpolation for MOE
xMOE=np.array([1300.984413,1335.438884,1380.721903,1405.33224,1425.020509,1435.849057,1444.708778,1451.599672,1459.474979,1470.303527,1492.945037,1511.648893,1529.368335,1540.196883,1549.056604,1558.900738,1576.62018,1599.26169,1628.794094,1647.497949,1659.310911,1668.170632,1676.045939,
            1683.921247,1690.812141,1697.703035,1704.593929,1712.469237,1720.344545,1728.219852,1737.079573,1748.892535,1761.68991,1780.393765,1796.144381,1815.83265,1830.598852,1844.38064,1854.224774,1862.100082,1872.92863,1884.741591,1905.414274,1928.055783,1953.650533,1979.245283,
            2003.855619,2017.637408,2030.434783,2038.31009,2048.154225,2054.060705,2060.9516,2067.842494,2074.733388,2081.624282,2087.530763,2094.421657,2105.250205,2118.04758,2136.751436,2165.299426,2212.551272,2284.413454,2364.150943,2424.200164,2500.015587])*1E-3
yMOE=np.array([0.218319151,0.433592405,0.429471486,0.973253696,2.827936251,6.212286227,10.47045032,14.72879358,19.205456,22.69901034,25.09944598,24.00570018,21.0555697,17.88765759,15.37515086,13.08096327,10.1308328,9.145933024,10.89051536,13.83733118,16.45716099,19.73248577,22.68028703,
            27.26615381,31.74290581,36.11045344,41.67924912,48.12159016,54.12711373,56.31048441,57.5109262,56.30860312,53.79573805,51.71915293,51.39010647,52.37115408,51.82378846,49.3108338,45.26937632,41.99252859,37.84177717,32.162075,25.49872725,21.12849206,19.70650606,21.34224237,
            27.34624299,34.00645525,42.85084447,49.62079861,57.26420853,61.30423265,62.72326233,61.19377408,54.75008927,48.30640445,41.09837864,33.6718545,24.71611088,18.27188855,11.39031123,6.145903588,3.193085558,1.985297788,1.759632649,0.552919902,0.109293953])*1E-2
f_nirMOE=interp1d(xMOE,yMOE,kind='quadratic',fill_value='extrapolate')

def condition_checker_moe(sim_params):
    """cfFactorCall for physics terms in cost function."""
    l=sim_params['callPoint']*1E6
    if l<1.3 or l>2.5:
        return [0,0,0,0,0,[0,1],[0,0]]
    else:
        return [0,0,0,0,0,[f_nirMOE(l),1],[0,0]]

fp=10
class dyn_sched:
    def __init__(self):
        self.gi_its=0
        return

    def dynamic_scheduler(self,prevL,curL,it,cur_x,cur_y,prev_factors,no_cb=False):
        matCost,atanMag,atanArg,footprint,expArg=0,70,400,fp,0.2
        if it==-1:
            return [matCost,atanMag,atanArg,footprint,expArg,0,0]
        self.gi_its+=1
        rvs=[matCost,atanMag,atanArg,footprint,expArg,prev_factors[5],prev_factors[6]]
        if no_cb:
            return True, rvs
        return rvs

def main():
    # Initialize scheduler
    ds=dyn_sched()

    # Initialize director
    ad=agnostic_director()

    # Initial conditions
    x0=[0.32441199332467563, 0.2421437838701929, 0.47340025698204713, 0.4825611995119943, 0.19989972134583547, 0.14736970395447987, 0.10213421303000284, 0.16361662441880945, 0.2796481376976949, 0.10995312837538893, 0.5725525242494137, 0.19900440425370333]

    # Load physics functions
    ad.set_physics(physics_package=optics_tmm,
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
                    initial_layer_thickness=x0,
                    initial_ys=[1,0]*6,
                    y_bounds=[0,1],
                    x_bounds=[0.005,1],
                    num_intervals=1,
                    substrates=[],superstrates=[])

    # Define discretization map
    discretization_map=[
        {'var_x':True,'var_y':False,'scramble_x':[False,0.02],'scramble_y':[False,0.01],'merge_neighbors':[False,0.05],'fully_discretize':False,'purge_ablated':[False,0.01]},
        {'var_x':True,'var_y':False,'scramble_x':[False,0.02],'scramble_y':[False,0.01],'merge_neighbors':[False,0.05],'fully_discretize':False,'purge_ablated':[True,0.01]}
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
                       presearch_threshold=np.inf)

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
                 subplots_adjust={'left':None,'right':None,'top':None,'bottom':None,'hspace':None,'wspace':None},
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

if __name__ == "__main__":
    main()
