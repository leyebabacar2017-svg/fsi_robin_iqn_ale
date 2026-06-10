import numpy as np

def find_tip_dof(
    V,
    x_tip=0.60,
    y_tip=0.20
):

    coords = V.tabulate_dof_coordinates()

    dist = (
        (coords[:,0]-x_tip)**2
        +
        (coords[:,1]-y_tip)**2
    )

    return np.argmin(dist)