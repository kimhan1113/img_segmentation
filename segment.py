import numpy as np
from skimage import io
from skimage.segmentation import slic
from skimage.future import graph
import time # DEBUG
import os

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def seg_merge_label(path, n_segments=10, compactness=1, measure_time=False):
    if measure_time:   
        start = time.time()
    
    if os.path.isfile(path):
        img = io.imread(path)
        label = slic(img, n_segments, compactness, start_label=1, enforce_connectivity=False)
        
        
        rag = graph.rag_mean_color(img, label)
        label_mer = graph.merge_hierarchical(label, rag, thresh = 35, rag_copy = False,\
            in_place_merge = True, merge_func = merge_mean_color, weight_func = _weight_mean_color)
        
        if measure_time:
            end = time.time()
            print('seg_merge_label({}): {}s'.format(path, end-start))\
        
        return np.asarray(label_mer)
