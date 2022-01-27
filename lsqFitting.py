
#----------------------------------------------------------------------------
# lsqFitting.py: Contains two functions that we use in modifyThickness.py and
#                predict.py. These functions use the skimage least-square
#                ellipse model fitting to find parameters of the best fitting
#                ellipse, and use ellipse parameters to produce a list of
#                coordinates contained in the ellipse.
#----------------------------------------------------------------------------

import numpy as np
from skimage.measure import EllipseModel
import random
#from matplotlib.patches import Ellipse
#import matplotlib.pyplot as plt

# For algorithm theories please see EllipseModel handbook
def lsqEllipse(coordits):
    coordits = random.sample(coordits, min(200, len(coordits)))
    #Too much data will lead to overflow
    ell = EllipseModel()
    ell.estimate(np.array(coordits).astype(np.float32))
    return ell.params
    #xc, yc, a, b, theta = ell.params
    #return xc, yc, a, b, theta

# Polar coordinate function to generate the points that belong to an ellipse
def ellipseCoordits(xc, yc, a, b, theta, dots):
    coordits = []
    cos, sin = np.cos(theta), np.sin(theta)
    for i in range(dots):
        angle = 2*i*np.pi / dots
        x, y = a*np.cos(angle), b*np.sin(angle)
        x, y = xc + x*cos+y*sin, yc + x*sin-y*cos
        coordits.append((x, y))
    return coordits
