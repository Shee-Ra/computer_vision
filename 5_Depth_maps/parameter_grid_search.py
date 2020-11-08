
from get_depth import *
import numpy as np

# I load the functions python -i 5_Depth_maps/parameter_grid_search.py
# then run in GridSearchDepthHyperParameters in the command line.

def GridSearchDepthHyperParameters(win_search_space=np.unique(np.concatenate((np.linspace(10, 25, 16), np.linspace(5, 50, 10)), axis=0)),
                                   overlap_search_space=np.unique(np.concatenate((np.linspace(0, 0.9, 10), np.linspace(0.7, 0.99, 30)), axis=0)),
                                   padding_search_space=np.linspace(0, 2, 3),
                                   window_search_strategy_search_space=[True, False],
                                   mylagfunc_search_space=[True, False]):
    """ Performed parameter search for depth maps.

    Args:
        win_search_space (1D np array): window search size in pixels. Defaults to np.linspace(5,50,10).
        overlap_search_space (1D np array): overlap of windows. Defaults to np.linspace(0.7,0.99,30).
        padding_search_space (1D np array): pad window to increase search space. Defaults to np.linspace(0, 2, 3)
        window_search_strategy_search_space (list): search in a widow or along a row. Defaults to [True, False].
        mylagfunc_search_space (list): uses my 2Dlag function or scipy's signal.correlate. Defaults to [True, False].
    """
    filenames=os.listdir('5_Depth_maps/data/')
    jpgs = [x for x in filenames if '.jpg' in x]
    file_pairs = []

    for x in jpgs:
        name = x.split('_')[0]
        if x.split('_')[1] == 'left.jpg':
            file_pairs.append(('5_Depth_maps/data/'+x, '5_Depth_maps/data/'+name+'_right.jpg'))            

    ims = ['giraffe', 'block', 'rubiks']
    for k, p in enumerate(file_pairs):
        left, right = load_image_pairs(p)
        log.debug('image pair loaded')

    for wss in window_search_strategy_search_space:
        for b in mylagfunc_search_space:
                for win in win_search_space:
                    for o in overlap_search_space:  
                        for pad in padding_search_space:   
                            try:
                                lags_mapped_to_2D = GetDepthMap(left, 
                                                                right, 
                                                                windowsize=int(win), 
                                                                overlap=o, 
                                                                padding=int(pad), 
                                                                windowsearch=wss, 
                                                                use_my_lag_function=b)
                                filename=f'5_Depth_maps/data/research/{ims[k]}_mylagfnc_{b}_winsearch_{wss}_window_{int(win)}px_overlap_{int(o*100)}pc_padding_{int(pad)}sq.pdf'
                                cmap = plt.cm.jet
                                image = cmap(lags_mapped_to_2D)
                                plt.imshow(lags_mapped_to_2D)
                                plt.colorbar()
                                # plt.show()
                                plt.savefig(filename)
                                plt.clf()
                                print(f'saved window {int(win)}, overlap {int(o*100)} and padding {int(pad)} with my lag function {b} and window search {wss}')
                            except:
                                pass 