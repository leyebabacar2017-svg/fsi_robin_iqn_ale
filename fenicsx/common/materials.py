def lame_parameters(E, nu):
    """
    Paramètres de Lamé
    """

    mu = E/(2.0*(1.0+nu))

    lmbda = (
        E*nu
        /
        ((1.0+nu)*(1.0-2.0*nu))
    )

    return mu, lmbda
