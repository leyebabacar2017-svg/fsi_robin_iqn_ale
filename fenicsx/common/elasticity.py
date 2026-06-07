import ufl

def eps(u):
    """
    Tenseur des petites déformations
    """

    G = ufl.grad(u)

    G2 = ufl.as_tensor(
        [
            [G[0,0], G[0,1]],
            [G[1,0], G[1,1]]
        ]
    )

    return ufl.sym(G2)

def sigma(u, mu, lmbda):
    """
    Loi de Hooke isotrope
    """

    e = eps(u)

    return (
        lmbda * ufl.tr(e) * ufl.Identity(2)
        +
        2.0 * mu * e
    )
