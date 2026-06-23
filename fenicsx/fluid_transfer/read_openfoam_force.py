import numpy as np


def read_openfoam_force(
    filename,
    interface_length=0.02
):
    """
    Lecture de la dernière ligne du fichier
    OpenFOAM force.dat.

    Retourne :
        traction moyenne [tx, ty]

    avec

        tx = Fx / interface_length
        ty = Fy / interface_length
    """

    data = np.loadtxt(
        filename,
        comments="#"
    )

    if data.ndim == 1:
        data = data.reshape(
            1,
            -1
        )

    last = data[-1]

    time = last[0]

    Fx = last[1]
    Fy = last[2]

    tx = Fx / interface_length
    ty = Fy / interface_length

    print()
    print("OpenFOAM force reader")
    print("---------------------")
    print("time =", time)
    print("Fx   =", Fx)
    print("Fy   =", Fy)
    print("tx   =", tx)
    print("ty   =", ty)

    return np.array(
        [
            tx,
            ty
        ]
    )


if __name__ == "__main__":

    traction = read_openfoam_force(
        "openfoam/turek_cylinderA_validated/"
        "postProcessing/forcesBeam/0/force.dat"
    )

    print()
    print(
        "Average traction =",
        traction
    )
