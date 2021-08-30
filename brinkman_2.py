from dolfin import *
import ulfy
import sympy as sp


def brinkman_mms(mesh, params=None):
    '''Manufactured solution for [] case with 2 networks'''
    # Expect all
    if params is not None:
        # Scalars 
        mu, lmbda, dt = (Constant(params[key]) for key in ('mu', 'lmbda', 'dt'))
        # Vectors ...
        (alpha_0, alpha_1,
         K_0, K_1,
         nu_0, nu_1,
         c_0, c_1) = (Constant(params[key]) for key in ('alpha_0', 'alpha_1',
                                                        'K_0', 'K_1',
                                                        'nu_0', 'nu_1',
                                                        'c_0', 'c_1'))
        # The exchange matrix for 2x2 reduces to single parameter
        beta = Constant(params['beta'])
    # Defaults are all 1
    else:
        params = dict((k, 1) for k in ('mu', 'lmbda', 'dt',
                                       'alpha_0', 'alpha_1', 'K_0', 'K_1', 'nu_0', 'nu_1', 'c_0', 'c_1',
                                       'beta'))
        return brinkman_mms(mesh, params)

    # Transformation of parameters
    gamma_0, gamma_1 = nu_0*(1/K_0)*alpha_0**2*(1/dt), nu_1*(1/K_1)*alpha_1**2*(1/dt)
    R_0, R_1 = K_0*dt*(1/alpha_0**2), K_1*dt*(1/alpha_1**2) 

    # We have 
    # -div(sigma(u) + p*I) = f0
    # -gamma*div(eps(v)) + (1/R)*v - grad(p) = f1
    # div(u) + div(v) - c*(1/alpha**2)*p0 = f2

    x, y = SpatialCoordinate(mesh)
    # Displacement
    phi = sin(pi*(x+y))
    
    u = as_vector((phi.dx(1), -phi.dx(0)))
    # Fluxes
    v0, v1 = as_vector((phi.dx(0), -phi.dx(1))), as_vector((phi.dx(0), -phi.dx(1))) 
    # Pressurees
    p0, p1 = cos(2*pi*(x-y)), cos(2*pi*(x-y))
    
    sigma = 2*mu*sym(grad(u)) + lmbda*div(u)*Identity(2) + p0*Identity(2) + p1*Identity(2)
    f0 = -div(sigma)
    
    f10 = -gamma_0*div(sym(grad(v0))) + (1/R_0)*v0 - grad(p0)
    f11 = -gamma_1*div(sym(grad(v1))) + (1/R_1)*v1 - grad(p1)
    
    f20 = div(u) + div(v0) - c_0*(1/alpha_0**2)*p0 + dt*(1/alpha_0)*beta*((1/alpha_1)*p1 - (1/alpha_0)*p0)
    f21 = div(u) + div(v1) - c_1*(1/alpha_1**2)*p1 + dt*(1/alpha_1)*beta*((1/alpha_0)*p0 - (1/alpha_1)*p1)    

    mu_, lmbda_, dt_ = sp.symbols('mu, lmbda, dt')
    alpha_0_, alpha_1_, K_0_, K_1_, nu_0_, nu_1_, c_0_, c_1_ = sp.symbols(
        'alpha_0, alpha_1, K_0, K_1, nu_0, nu_1, c_0, c_1'
    )
    beta_ = sp.Symbol('beta')

    symbols = (mu_, lmbda_, dt_,
               alpha_0_, alpha_1_, K_0_, K_1_, nu_0_, nu_1_, c_0_, c_1_,
               beta_)

    d = locals()
    subs = {d[s.name]: s for s in symbols}
    assert all(subs[k].name in params for k in subs)
    
    as_expr= lambda f: ulfy.Expression(f, subs=subs, degree=4, **params)
    
    def get_errors(wh):
        uh, v0h, v1h, p0h, p1h = wh

        return {'|eu|_1': errornorm(as_expr(u), uh, 'H1', degree_rise=2),
                '|ev0|_div': errornorm(as_expr(v0), v0h, 'Hdiv', degree_rise=2),
                '|ev1|_div': errornorm(as_expr(v1), v1h, 'Hdiv', degree_rise=2),
                '|ep0|_0': errornorm(as_expr(p0), p0h, 'L2', degree_rise=2),
                '|ep1|_0': errornorm(as_expr(p1), p1h, 'L2', degree_rise=2)}

    return {'parameters': params,
            'get_errors': get_errors,
            'forces': {'f0': as_expr(f0),
                       'f10': as_expr(f10), 'f11': as_expr(f11),
                       'f20': as_expr(f20), 'f21': as_expr(f21),
                       'biot_stress': as_expr(sigma),
                       'flux0_sym_grad': as_expr(sym(grad(v0))),
                       'flux1_sym_grad': as_expr(sym(grad(v1)))},
            'solution': {'w': as_expr(u), 
                         'v0': as_expr(v0), 'v1': as_expr(v1),
                         'p0': as_expr(p0), 'p1': as_expr(p1)}}


def system_brinkman(facet_f, mms):
    '''Auxiliary'''
    mesh = facet_f.mesh()
    # W = VectorFunctionSpace(mesh, 'Lagrange', 2)
    cell = mesh.ufl_cell()
    Welm = FiniteElement('Brezzi-Douglas-Marini', cell, 1)
    V0elm = FiniteElement('Brezzi-Douglas-Marini', cell, 1)
    V1elm = FiniteElement('Brezzi-Douglas-Marini', cell, 1)    
    Q0elm = FiniteElement('Discontinuous Lagrange', cell, 0)
    Q1elm = FiniteElement('Discontinuous Lagrange', cell, 0)

    Melm = MixedElement([Welm, V0elm, V1elm, Q0elm, Q1elm])
    M = FunctionSpace(mesh, Melm)

    ds = Measure('ds', domain=mesh, subdomain_data=facet_f)


    mu, lmbda, dt = (Constant(mms['parameters'][key]) for key in ('mu', 'lmbda', 'dt'))
    # Vectors ...
    (alpha_0, alpha_1,
     K_0, K_1,
     nu_0, nu_1,
     c_0, c_1) = (Constant(mms['parameters'][key]) for key in ('alpha_0', 'alpha_1',
                                                               'K_0', 'K_1',
                                                               'nu_0', 'nu_1',
                                                               'c_0', 'c_1'))
    # The exchange matrix for 2x2 reduces to single parameter
    beta = Constant(mms['parameters']['beta'])

    # Transformation of parameters
    gamma_0, gamma_1 = nu_0*(1/K_0)*alpha_0**2*(1/dt), nu_1*(1/K_1)*alpha_1**2*(1/dt)
    R_0, R_1 = K_0*dt*(1/alpha_0**2), K_1*dt*(1/alpha_1**2) 
    
    
    w, u0, u1, p0, p1 = TrialFunctions(M)
    phi, v0, v1, q0, q1 = TestFunctions(M)

    n = FacetNormal(mesh)

    eps = lambda x: sym(grad(x))
    a = (2*mu*inner(eps(w), eps(phi))*dx + lmbda*inner(div(w), div(phi))*dx + inner(p0, div(phi))*dx + inner(p1, div(phi))*dx
         + gamma_0*inner(eps(u0), eps(v0))*dx + (1/R_0)*inner(u0, v0)*dx      + inner(p0, div(v0))*dx
         + gamma_1*inner(eps(u1), eps(v1))*dx + (1/R_1)*inner(u1, v1)*dx                             + inner(p1, div(v1))*dx
         + inner(div(w), q0)*dx + inner(div(u0), q0)*dx - c_0*(1/alpha_0**2)*inner(p0, q0)*dx - dt*(1/alpha_0**2)*beta*inner(p0, q0)*dx + dt*(1/alpha_0)*(1/alpha_1)*beta*inner(p1, q0)*dx
         + inner(div(w), q1)*dx + inner(div(u1), q1)*dx + dt*(1/alpha_0)*(1/alpha_1)*beta*inner(p0, q1)*dx - c_1*(1/alpha_1**2)*inner(p1, q1)*dx - dt*(1/alpha_1**2)*beta*inner(p1, q1)*dx 
    )

    tangent = lambda v, n: v - n*dot(v, n)    
    # Stabilization
    n = FacetNormal(mesh)
    stab = Constant(20)
    hA = avg(FacetArea(mesh))  # The usual average

    # NOTE: these are the jump operators from Krauss, Zikatonov paper.
    # Jump is just a difference and it preserves the rank 
    Jump = lambda arg: arg('+') - arg('-')
    # Average is use dot with normal and AGAIN MINUS; it reduces the rank
    Avg = lambda arg, n: Constant(0.5)*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))
    
    # Interior penalty for displacement
    a += (- 2*mu*inner(Avg(eps(w), n), Jump(tangent(phi, n)))*dS
          - 2*mu*inner(Avg(eps(phi), n), Jump(tangent(w, n)))*dS
          + 2*mu*(stab/hA)*inner(Jump(tangent(w, n)), Jump(tangent(phi, n)))*dS)
    # Interior penalty for flux0
    a += (- gamma_0*inner(Avg(eps(u0), n), Jump(tangent(v0, n)))*dS
          - gamma_0*inner(Avg(eps(v0), n), Jump(tangent(u0, n)))*dS
          + gamma_0*(stab/hA)*inner(Jump(tangent(u0, n)), Jump(tangent(v0, n)))*dS)
    # Interior penalty for flux1
    a += (- gamma_1*inner(Avg(eps(u1), n), Jump(tangent(v1, n)))*dS
          - gamma_1*inner(Avg(eps(v1), n), Jump(tangent(u1, n)))*dS
          + gamma_1*(stab/hA)*inner(Jump(tangent(u1, n)), Jump(tangent(v1, n)))*dS)

    f_w, f_u0, f_u1, f_p0, f_p1 = (mms['forces'][key] for key in ('f0', 'f10', 'f11', 'f20', 'f21'))
    
    L = inner(f_w, phi)*dx + inner(f_u0, v0)*dx + inner(f_u1, v1)*dx + inner(f_p0, q0)*dx + inner(f_p1, q1)*dx

    traction = mms['forces']['biot_stress']
    # Add tangential stress on Dirichlet boundaries
    L += sum(inner(tangent(dot(traction, n), n), tangent(phi, n))*ds(tag)
             for tag in (1, 2, 3, 4))

    sg0, sg1 = mms['forces']['flux0_sym_grad'], mms['forces']['flux1_sym_grad']
    for tag in (1, 2, 3, 4):
        L += inner(gamma_0*dot(n, sg0) + mms['solution']['p0']*n, v0)*ds(tag)
        L += inner(gamma_1*dot(n, sg1) + mms['solution']['p1']*n, v1)*ds(tag)        

    u = mms['solution']['w']
    bcs = [DirichletBC(M.sub(0), u, facet_f, tag) for tag in (1, 2, 3, 4)]
    
    wh = Function(M)
    A, b = assemble_system(a, L, bcs)

    solver = LUSolver(A, 'mumps')
    solver.solve(wh.vector(), b)

    return wh.split(deepcopy=True)

# --------------------------------------------------------------------

if __name__ == '__main__':
    params = {
        'mu': 1.1, 'lmbda': 2, 'dt': 0.1,
        'alpha_0': 0.9, 'alpha_1': 1.2, 'K_0': 1.3, 'K_1': 1.5, 'nu_0': 1.24, 'nu_1': 1.9, 'c_0': 0.2, 'c_1': 0.3,
        'beta': 0.5
    }
    
    for n in (2, 4, 8, 16, 32, 64):
        mesh = UnitSquareMesh(n, n)
        mms = brinkman_mms(mesh, params=params)

        facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
        CompiledSubDomain('near(x[0], 1)').mark(facet_f, 2)
        CompiledSubDomain('near(x[1], 0)').mark(facet_f, 3)
        CompiledSubDomain('near(x[1], 1)').mark(facet_f, 4)

        wh = system_brinkman(facet_f, mms)
        ndofs = sum(xh.function_space().dim() for xh in wh)
        print(mms['get_errors'](wh), ndofs)
