from dolfinx import fem

def assemble_load_vector(
    load_form,
    bcs=None
):

    F = fem.petsc.assemble_vector(
        fem.form(load_form)
    )

    if bcs is not None:

        fem.petsc.apply_lifting(
            F,
            [fem.form(load_form)],
            [bcs]
        )

        fem.petsc.set_bc(
            F,
            bcs
        )

    F.assemble()

    return F