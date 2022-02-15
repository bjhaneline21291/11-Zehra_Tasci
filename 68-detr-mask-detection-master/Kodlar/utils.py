"""
Various utility functions
"""
#%% Setup

#%% Box functions
def min_max_to_cx_cy(min_max_box):
    """Convert min-max box to center-height-width box
    """
    # Calculate the center and dimensions
    cx = (min_max_box[0] + min_max_box[2]) / 2
    w = min_max_box[2] - min_max_box[0]
    cy = (min_max_box[1] + min_max_box[3]) / 2
    h = min_max_box[3] - min_max_box[1]

    return [cx, cy, h, w]

def cx_cy_to_min_max(cx_cy_box):
    """Convert a center-width-height box to a min-max box
    """
    # Calculate the center and dimensions
    (xmin, xmax), (ymin, ymax) = [(cx_cy_box[i] - cx_cy_box[j] / 2, cx_cy_box[i] + cx_cy_box[j] / 2)
                                  for [i,j] in [(0,3), (1,2)]]

    return [xmin, ymin, xmax, ymax]