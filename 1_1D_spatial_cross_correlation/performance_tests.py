import numpy as np
from script_to_get_offset import *
import matplotlib.pyplot as plt

# I load the functions python -i path_to_this_file/performance_tests.py
# then run in the command line.

def TestFunctionPerformance(runs=1000):
    """returns array of times the function runs for

    Args:
        runs (int, optional): Number of times to run function. Defaults to 1000.

    Returns:
        (np array): 1D array of times it took to run the main function in seconds
    """
    main_folder = './'

    times = []
    for r in range(runs):
        t, d = main(main_folder, print_results=False)
        times.append(t)

    return np.array(times)


def PlotRunTimes(times,
                 xt=np.linspace(1, 5, 5),
                 xy=np.linspace(0, 700, 8),
                 file_save_path="1_1D_spatial_cross_correlation/data/results/run_time_1D_script_2.pdf"
                 ):
    """plots run time and saves result

    Args:
        times ([type]): [description]
        xt (list, optional): x-axis ticks. Defaults to [1.5, 1.7, 1.9, 2.1, 2.3].
        xy (list, optional): y-axis ticks. Defaults to [0, 250, 500, 750, 1000].
        file_save_path (str, optional): location where image is saved.
    """

    plt.hist(times)
    plt.xticks(xt, fontsize=20)
    plt.yticks(xy, fontsize=20)

    plt.xlabel('Run time [seconds]', fontsize=20)
    plt.ylabel('Count', fontsize=20)

    plt.title('Histograms of run times', fontsize=20)
    try:
        plt.savefig(file_save_path, bbox_inches='tight', dpi=100)
    except:
        pass