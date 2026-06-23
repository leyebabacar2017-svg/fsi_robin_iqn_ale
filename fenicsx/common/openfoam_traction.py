from dolfinx import fem
import numpy as np


def create_openfoam_traction(
    mesh,
    traction_values
):
    """
    Traction constante issue d'OpenFOAM.

    traction_values =
        [tx, ty]
    """

    traction = fem.Constant(
        mesh,
        np.array(
            traction_values,
            dtype=np.float64
        )
    )

    return traction
