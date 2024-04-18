"""
Work with DENCLUE clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


def denclue(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the DENCLUE algorithm only using the `numpy` module

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'xi'. There could be others.
       params_dict['sigma'] should be in the range [.1, 10], while
       params_dict['xi'] should be in the range .1 to 10. The
       dictionary values scalar float values.

    Return value:
    """

    computed_labels: NDArray[np.int32] | None = None
    SSE: float | None = None
    ARI: float | None = None

    return computed_labels, SSE, ARI


def denclue_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.

    """

    answers = {}

    # Return your `denclue` function
    answers["denclue_function"] = denclue

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using DENCLUE
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.  For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # Variable `groups` is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = None

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = None
    plot_SSE = None
    answers["scatterplot_cluster_with_largest_ARI"] = plot_ARI
    answers["scatterplot_cluster_with_smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A list of floats
    answers["mean_std_ARI"] = None

    # A single float
    answers["mean_ARI"] = None

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = denclue_clustering()
    with open("denclue_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
