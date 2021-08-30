from dolfin import *
from dolfin_adjoint import *
    
def solve_adjoint(bdries, alpha, data, dirichlet_tags):
    '''One shot formulation with [P1]**3 of "mother" problem'''
    mesh = bdries.mesh()
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)

    D = FunctionSpace(mesh, 'DG', 2)
    keys = ('dir_data', 'neumann_data', 'volume_data')
    for key in keys:
        val = data[key]
        if isinstance(val, dict):
            data[key] = {k: interpolate(val[k], D) for k in val}
        else:
            data[key] = interpolate(val, D)
            
    neumann_tags = set((1, 2, 3, 4)) - set(dirichlet_tags)    

    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 0)

    f = interpolate(Expression("0.0", degree=1, name='Control'), W)
    u = Function(V, name='State')
    v = TestFunction(V)

    # Define and solve the Poisson equation to generate the dolfin-adjoint annotation
    F = (inner(grad(u), grad(v))*dx
         - inner(f, v)*dx
         - sum(inner(data['neumann_data'][tag], v)*ds(tag) for tag in neumann_tags))
    bc = [DirichletBC(V, data['dir_data'][tag], bdries, tag)  for tag in dirichlet_tags]
    solve(F == 0, u, bc)

    alpha = Constant(alpha)
    d = data['volume_data']
    J = assemble((0.5 * inner(u - d, u - d)) * dx + alpha / 2 * f ** 2 * dx)
    
    control = Control(f)
    rf = ReducedFunctional(J, control)
    f_opt = minimize(rf, bounds=(-2.0, 2.0), tol=1e-12,
                     options={"gtol": 1e-12, "factr": 0.0})

    f.assign(f_opt)
    solve(F == 0, u, bc)

    return u, f_opt

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from gmshnics import gUnitSquare
    from xii import ii_Function, apply_bc, ii_convert
    import dolfin as df
    from oneshot_fruitfly import (simon_test_case, my_test_case, nz_test_case,
                                  get_system)
    from dolfin import *
    from dolfin_adjoint import *
    import numpy as np
    
    alpha_value = 1E0
    data = nz_test_case(alpha_value)
    u_true, f_true = data['u_state'], data['f_control']

    bdry_domains = {1: CompiledSubDomain('near(x[0], 0)'),
                    2: CompiledSubDomain('near(x[0], 1)'),
                    3: CompiledSubDomain('near(x[1], 0)'),
                    4: CompiledSubDomain('near(x[1], 1)')}
    
    dirichlet_tags = (1, 2)

    hs, eus, eps = [], [], []
    for k in (4, 8, 16, 32, 64):
        scale = 1./(k/4)
        # mesh, entity_fs = gUnitSquare(scale)
        # bdries = entity_fs[1]

        mesh = UnitSquareMesh(k, k, 'crossed')
        bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        [bdry_domains[tag].mark(bdries, tag) for tag in (1, 2, 3, 4)]

        uh, fh = solve_adjoint(bdries, alpha_value, data, dirichlet_tags)

        A, b, bcs, W = get_system(bdries, alpha_value, data, dirichlet_tags)

        wh = ii_Function(W)
        A, b = apply_bc(A, b, bcs)
        A, b = map(ii_convert, (A, b))

        df.solve(A, wh.vector(), b)

        uh_o, fh_o, _ = wh

        print(errornorm(uh_o, uh, 'L2', degree_rise=0),
              errornorm(fh_o, fh, 'L2', degree_rise=0))

        eus.append(errornorm(u_true, uh, 'L2'))
        eps.append(errornorm(f_true, fh, 'L2'))
        hs.append(mesh.hmin())
    
        if len(hs) > 1:
            rate_u = np.log(eus[-1]/eus[-2])/np.log(hs[-1]/hs[-2])
            rate_p = np.log(eps[-1]/eps[-2])/np.log(hs[-1]/hs[-2])            
        else:
            rate_u, rate_p = np.nan, np.nan

        File('uh.pvd') << uh
        File('u.pvd') << interpolate(u_true, uh.function_space())
        File('eu.pvd') << (uh.vector().axpy(-1, interpolate(u_true, uh.function_space()).vector()), uh)[1]

        File('fh.pvd') << fh
        File('f.pvd') << interpolate(f_true, fh.function_space())
        File('ef.pvd') << (fh.vector().axpy(-1, interpolate(f_true, fh.function_space()).vector()), fh)[1]
        
        
        print(' '.join([f'h={hs[-1]:.2E}',
                        f'eu={eus[-1]:.3E}[{rate_u:.2f}]',
                        f'ep={eps[-1]:.3E}[{rate_p:.2f}]']))
