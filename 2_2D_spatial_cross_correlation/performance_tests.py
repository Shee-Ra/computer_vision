from get_wally import *
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
    times = []
    for r in range(runs):
        start = time.time()
        main(small_image='2_2D_spatial_cross_correlation/data/wallypuzzle_rocket_man.png',
                    big_image='2_2D_spatial_cross_correlation/data/wallypuzzle.png')
        end = time.time()
        times.append(end-start)

    return np.array(times)


def PlotRunTimes(times,
                 xt=[0, 1.0, 2.0, 3.0, 4.0],
                 xy=[0, 250, 500, 750, 1000],
                 file_save_path="2_2D_spatial_cross_correlation/data/output/run_time_2D_script.pdf" #
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
        main_folder = './'
        file_save_path = os.path.join(main_folder,file_save_path)
        plt.savefig(file_save_path, bbox_inches='tight', dpi=100)
    except:
        pass