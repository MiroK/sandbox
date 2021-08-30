from dolfin import *
from xii import *
import sympy as sp
import ulfy


def simon_test_case(alpha_value):
    '''For unit square with 1, 2, 3, 4 boundaries'''
    # The one from dolfin adjoint
    mesh = UnitSquareMesh(2, 2)
    x, y = SpatialCoordinate(mesh)

    w = sin(pi*x)*sin(pi*y)
    d = 1/(2*pi **2)
    d = d*w

    alpha = Constant(1)
    f_control = 1/(1+alpha*4*pi**4)*w
    u_state = 1/(2*pi**2)*f_control

    normals = [Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1))]
    fluxes = [dot(grad(u_state), n) for n in normals]

    subs = {alpha: sp.Symbol('alpha')}

    expr = lambda f: ulfy.Expression(f, subs=subs, degree=5, alpha=alpha_value)

    return {'u_state': expr(u_state),
            'f_control': expr(f_control),
            'dir_data': dict(enumerate([expr(u_state)]*4, 1)),
            'neumann_data': dict(enumerate(map(expr, fluxes), 1)),
            'volume_data': expr(d)}


def my_test_case(alpha_value):
    '''For unit square with 1, 2, 3, 4 boundaries'''
    # The one from dolfin adjoint
    mesh = UnitSquareMesh(2, 2)
    x, y = SpatialCoordinate(mesh)

    w = cos(pi*x)*cos(pi*y)
    d = 1/(2*pi **2)
    d = d*w

    alpha = Constant(1)
    f_control = 1/(1+alpha*4*pi**4)*w
    u_state = 1/(2*pi**2)*f_control

    normals = [Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1))]
    fluxes = [dot(grad(u_state), n) for n in normals]

    subs = {alpha: sp.Symbol('alpha')}

    expr = lambda f: ulfy.Expression(f, subs=subs, degree=5, alpha=alpha_value)

    return {'u_state': expr(u_state),
            'f_control': expr(f_control),
            'dir_data': dict(enumerate([expr(u_state)]*4, 1)),
            'neumann_data': dict(enumerate(map(expr, fluxes), 1)),
            'volume_data': expr(d)}


def nz_test_case(alpha_value):
    '''For unit square with 1, 2, 3, 4 boundaries'''
    # The one from dolfin adjoint
    mesh = UnitSquareMesh(2, 2)
    x, y = SpatialCoordinate(mesh)

    w = cos(pi*x)*cos(pi*y) + x*y
    d = 1/(2*pi **2)
    d = d*w

    alpha = Constant(1)
    f_control = 1/(1+alpha*4*pi**4)*w
    u_state = 1/(2*pi**2)*f_control

    normals = [Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1))]
    fluxes = [dot(grad(u_state), n) for n in normals]

    subs = {alpha: sp.Symbol('alpha')}

    expr = lambda f: ulfy.Expression(f, subs=subs, degree=5, alpha=alpha_value)

    return {'u_state': expr(u_state),
            'f_control': expr(f_control),
            'dir_data': dict(enumerate([expr(u_state)]*4, 1)),
            'neumann_data': dict(enumerate(map(expr, fluxes), 1)),
            'volume_data': expr(d)}


def get_system(bdries, alpha, data, dirichlet_tags):
    '''One shot formulation with [P1]**3 of "mother" problem'''
    mesh = bdries.mesh()
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)

    neumann_tags = set((1, 2, 3, 4)) - set(dirichlet_tags)    

    V = FunctionSpace(mesh, 'CG', 1)  # State space
    G = FunctionSpace(mesh, 'DG', 0)  # Control
    M = FunctionSpace(mesh, 'CG', 1)  # Multiplier
    W = (V, G, M)

    u, f, lm = map(TrialFunction, W)
    v, g, mm = map(TestFunction, W)

    alpha = Constant(alpha)

    a = block_form(W, 2)
    a[0][0] = inner(u, v)*dx
    a[0][2] = inner(grad(v), grad(lm))*dx
    #if dirichlet_tags:
    #    a[0][2] += sum(inner(v, lm)*ds(tag) for tag in dirichlet_tags)
    
    a[1][1] = alpha*inner(f, g)*dx
    a[1][2] = -inner(g, lm)*dx

    a[2][0] = inner(grad(u), grad(mm))*dx
    #if dirichlet_tags:
    #    a[2][0] += sum(inner(u, mm)*ds(tag) for tag in dirichlet_tags)
    
    a[2][1] = -inner(f, mm)*dx

    L = block_form(W, 1)
    L[0] = inner(data['volume_data'], v)*dx
    # Neumann contributions
    #if dirichlet_tags:
    #    L[2] = -sum(inner(data['dir_data'][tag], mm)*ds(tag) for tag in dirichlet_tags)
    if neumann_tags:
        # L[0] += sum(inner(data['neumann_data'][tag], v)*ds(tag) for tag in neumann_tags)        
        L[2] += sum(inner(data['neumann_data'][tag], mm)*ds(tag) for tag in neumann_tags)

    V_bcs = [DirichletBC(V, data['dir_data'][tag], bdries, tag) for tag in dirichlet_tags]
    L_bcs = []
    M_bcs = [DirichletBC(M, Constant(0), bdries, tag) for tag in dirichlet_tags]

    A, b = map(ii_assemble, (a, L))
    bcs = [V_bcs, L_bcs, M_bcs]

    return A, b, bcs, W

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from adjoint_fruitfly import solve_adjoint
    from gmshnics import gUnitSquare
    import numpy as np
    
    alpha_value = 1E2
    data = my_test_case(alpha_value)
    u_true, f_true = data['u_state'], data['f_control']

    bdry_domains = {1: CompiledSubDomain('near(x[0], 0)'),
                    2: CompiledSubDomain('near(x[0], 1)'),
                    3: CompiledSubDomain('near(x[1], 0)'),
                    4: CompiledSubDomain('near(x[1], 1)')}
    
    dirichlet_tags = (1, 2, )

    hs, eus, eps = [], [], []
    for k in (4, 8, 16, 32, 64):
        scale = 1./(k/4)
        # mesh, entity_fs = gUnitSquare(scale)
        # bdries = entity_fs[1]

        mesh = UnitSquareMesh(k, k, 'crossed')
        bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        [bdry_domains[tag].mark(bdries, tag) for tag in (1, 2, 3, 4)]

        A, b, bcs, W = get_system(bdries, alpha_value, data, dirichlet_tags)

        wh = ii_Function(W)
        A, b = apply_bc(A, b, bcs)
        A, b = map(ii_convert, (A, b))

        solve(A, wh.vector(), b)

        uh, fh, _ = wh

        uh_a, fh_a = solve_adjoint(bdries, alpha_value, data, dirichlet_tags)
        print(errornorm(uh_a, uh, 'L2', degree_rise=0),
              errornorm(fh_a, fh, 'L2', degree_rise=0))
        
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
