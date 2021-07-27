from dolfin import *
import ulfy
import sympy as sp

# from block_utils import block_form, block_assemble, block_bc_apply
#import numpy as np
#from scipy.sparse import csr_matrix
    

def brinkman_mms(mesh, params=None):
    '''Manufactured solution for [] case'''
    # Expect all
    if params is not None:
        mu, lmbda, alpha, K, nu, c, dt = (
            Constant(params[key]) for key in ('mu', 'lmbda', 'alpha', 'K', 'nu', 'c', 'dt')
        )
    # All is one
    else:
        params = {'mu': 1, 'lmbda': 1,
                  'alpha': 1, 'K': 1, 'nu': 1, 'c': 1, 'dt': 1}
        return brinkman_mms(mesh, params)

    # Transformation of parameters
    gamma = nu*(1/K)*alpha**2*dt
    R = K*dt*(1/alpha**2)

    # We have 
    # -div(sigma(u) + p*I) = f0
    # -gamma*div(eps(v)) + (1/R)*v - grad(p) = f1
    # div(u) + div(v) - c*(1/alpha**2)*p0 = f2

    x, y = SpatialCoordinate(mesh)
    # Displacement
    phi = sin(pi*(x+y))
    
    u0 = as_vector((phi.dx(1), -phi.dx(0)))
    # Flux
    v0 = as_vector((phi.dx(0), -phi.dx(1)))
    # Pressure
    p0 = cos(2*pi*(x-y))
    
    sigma = 2*mu*sym(grad(u0)) + lmbda*div(u0)*Identity(2) + p0*Identity(2)
    f0 = -div(sigma)
    f1 = -gamma*div(sym(grad(v0))) + (1/R)*v0 - grad(p0)
    f2 = div(u0) + div(v0) - c*(1/alpha**2)*p0

    mu_, lmbda_, alpha_, K_, nu_, c_, dt_ = sp.symbols(
        'mu, lmbda, alpha, K, nu, c, dt'
    )

    subs = {mu: mu_, lmbda: lmbda_, alpha: alpha_, K: K_, nu: nu_, c: c_, dt: dt_}
    
    as_expr= lambda f: ulfy.Expression(f, subs=subs, degree=4,
                                       mu=1,
                                       lmbda=1,
                                       alpha=1,
                                       K=1,
                                       nu=1,
                                       c=1,
                                       dt=1)

    
    def get_errors(wh):
        uh, vh, ph = wh

        return {'|eu|_1': errornorm(as_expr(u0), uh, 'H1', degree_rise=2),
                '|eu|_div': errornorm(as_expr(v0), vh, 'Hdiv', degree_rise=2),
                '|ep|_0': errornorm(as_expr(p0), ph, 'L2', degree_rise=2)}

    return {'parameters': params,
            'get_errors': get_errors,
            'forces': {'f0': as_expr(f0), 'f1': as_expr(f1), 'f2': as_expr(f2),
                       'biot_stress': as_expr(sigma),
                       'flux_sym_grad': as_expr(sym(grad(v0)))},
            'solution': {'w': as_expr(u0), 'w0': as_expr(u0[0]), 'w1': as_expr(u0[1]),
                         'v': as_expr(v0), 'p': as_expr(p0)}}


def system_brinkman(mesh, mms):
    '''Auxiliary'''
    # W = VectorFunctionSpace(mesh, 'Lagrange', 2)
    cell = mesh.ufl_cell()
    Welm = FiniteElement('Brezzi-Douglas-Marini', cell, 1)
    Velm = FiniteElement('Brezzi-Douglas-Marini', cell, 1)
    #Welm = VectorElement('Lagrange', cell, 2)
    #Velm = VectorElement('Lagrange', cell, 2)
    Qelm = FiniteElement('Discontinuous Lagrange', cell, 0)

    Melm = MixedElement([Welm, Velm, Qelm])
    M = FunctionSpace(mesh, Melm)

    facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 1)').mark(facet_f, 2)
    CompiledSubDomain('near(x[1], 0)').mark(facet_f, 3)
    CompiledSubDomain('near(x[1], 1)').mark(facet_f, 4)

    ds = Measure('ds', domain=mesh, subdomain_data=facet_f)

    mu, lmbda, alpha, K, nu, c, dt = (
        Constant(mms['parameters'][key])
        for key in ('mu', 'lmbda', 'alpha', 'K', 'nu', 'c', 'dt')
    )
    # Transformation of parameters
    gamma = nu*(1/K)*alpha**2*dt
    R = K*dt*(1/alpha**2)
    
    w, u, p = TrialFunctions(M)
    phi, v, q = TestFunctions(M)

    n = FacetNormal(mesh)

    eps = lambda x: sym(grad(x))
    a = (2*mu*inner(eps(w), eps(phi))*dx + lmbda*inner(div(w), div(phi))*dx + inner(p, div(phi))*dx
         +gamma*inner(eps(u), eps(v))*dx + (1/R)*inner(u, v)*dx               + inner(p, div(v))*dx
         +inner(div(w), q)*dx + inner(div(u), q)*dx - c*(1/alpha**2)*inner(p, q)*dx)

    tangent = lambda v, n: v - n*dot(v, n)    
    # Stabilization
    n = FacetNormal(mesh)
    stab = Constant(10)
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
    # Interior penalty for flux
    a += (- gamma*inner(Avg(eps(u), n), Jump(tangent(v, n)))*dS
          - gamma*inner(Avg(eps(v), n), Jump(tangent(u, n)))*dS
          + gamma*(stab/hA)*inner(Jump(tangent(u, n)), Jump(tangent(v, n)))*dS)

    n = FacetNormal(mesh)
    
    f_w, f_u, f_p = (mms['forces'][key] for key in ('f0', 'f1', 'f2'))
    
    L = inner(f_w, phi)*dx + inner(f_u, v)*dx + inner(f_p, q)*dx

    traction = mms['forces']['biot_stress']
    # Add tangential stress on Dirichlet boundaries
    L += sum(inner(tangent(dot(traction, n), n), tangent(phi, n))*ds(tag)
             for tag in (1, 2, 3, 4))

    sg = mms['forces']['flux_sym_grad']
    for tag in (1, 2, 3, 4):
        L += inner(gamma*dot(n, sg) + mms['solution']['p']*n, v)*ds(tag)

    # Only want bcs on normal displacement
    #   4        1
    # 1  2     0  0
    #   3       1
    u00, u01 = mms['solution']['w0'], mms['solution']['w1']
    #bcs = [DirichletBC(M.sub(0).sub(0), u00, facet_f, 1),
    #       DirichletBC(M.sub(0).sub(0), u00, facet_f, 2),
    #       DirichletBC(M.sub(0).sub(1), u01, facet_f, 3),
    #       DirichletBC(M.sub(0).sub(1), u01, facet_f, 4)]

    u0, v0 = mms['solution']['w'], mms['solution']['v']        
    bcs = [DirichletBC(M.sub(0), u0, facet_f, tag) for tag in (1, 2, 3, 4)]
    # bcs.extend([DirichletBC(M.sub(1), v0, facet_f, tag) for tag in (1, 2, 3, 4)])
    
    wh = Function(M)
    #problem = LinearVariationalProblem(a, L, wh, bcs=bcs)
    #solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    # ksp = solver.snes.ksp
    solve(a == L, wh, bcs)
    #solver.solve()

    return wh.split(deepcopy=True)

# --------------------------------------------------------------------

if __name__ == '__main__':
    params = {
        'mu': 1, 'lmbda': 1, 'alpha': 1, 'K': 1, 'nu': 1, 'c': 1, 'dt': 1
    }
    
    for n in (2, 4, 8, 16, 32, 64):
        mesh = UnitSquareMesh(n, n)
        mms = brinkman_mms(mesh, params=None)

        wh, uh, ph = system_brinkman(mesh, mms)
        ndofs = (sum(xh.function_space().dim() for xh in (wh, uh, ph)))
        print(mms['get_errors']([wh, uh, ph]), ndofs)
