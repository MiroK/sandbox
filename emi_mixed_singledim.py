# Mixed singledimensional formulation of EMI model. We solve for flux and potential
# The transmembrane potential is computed as u_i - u_e when needed.

# NOTE: the form of the system to be solved at each time level is Ax = b
# where
#
#     [(1/sigma)*I + T'*T, grad;             [flux;
# A =  div,                   0]   and x =    potential]

from dolfin import *
import numpy.random
import numpy as np

# Set some default optimization parameters
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
#parameters["form_compiler"]["quadrature_degree"] = 4
parameters["ghost_mode"] = "shared_facet"

timer = Timer("X: EMI init and solve")

# Create outer domain
try:
    import sys
    N = int(sys.argv[1])
except:
    N = 1

mesh = RectangleMesh(Point(0, 0), Point(70.0e-4, 30.0e-4), N*35, N*15)

# Specify bottom left corner (a) and top right corner (b) of inner
# domain:
a = (10.e-4, 12.e-4) # Bottom left corner
b = (60.e-4, 18.e-4) # Top right corner

# Use the CompiledSubDomain rather than subclassing SubDomains in
# Python, faster run time.
in_membrane = """(near(x[0], %g) && x[1] >= %g && x[1] <= %g)
              || (near(x[0], %g) && x[1] >= %g && x[1] <= %g)
              || (near(x[1], %g) && x[0] >= %g && x[0] <= %g) 
              || (near(x[1], %g) && x[0] >= %g && x[0] <= %g)""" \
                  % (a[0], a[1], b[1], b[0], a[1], b[1],
                     a[1], a[0], b[0], b[1], a[0], b[0])
membrane = CompiledSubDomain(in_membrane)

# Define the membrane and intracellular spaces.
in_ics = "x[0] >= %g && x[0] <= %g && x[1] >= %g && x[1] <= %g" \
         % (a[0], b[0], a[1], b[1])
ics = CompiledSubDomain(in_ics)

# Define mesh function marking the interface facets by 1 (and all
# others by 0)
gamma = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
membrane.mark(gamma, 1)

nfacets_loc = sum(1 for _ in SubsetIterator(gamma, 1))
nfacets = mesh.mpi_comm().allgather(nfacets_loc)
assert sum(nfacets) > 0, nfacets

# Define mesh function marking the intracellular (0) and extracellular
# space (1) (just needed for the right assembly over interior facets).
cs = MeshFunction("size_t", mesh, mesh.topology().dim(), 1)
ics.mark(cs, 0)

# Define finite element spaces and mixed function space
S = FiniteElement("RT", mesh.ufl_cell(), 1)
U = FiniteElement("DG", mesh.ufl_cell(), 0)
M = FunctionSpace(mesh, MixedElement(S, U))

U = FunctionSpace(mesh, U)
# This is the space where we will have the transmembrane potential
V = FunctionSpace(mesh, "Discontinuous Lagrange Trace", 0)

# Unknowns and test functions
(s, u) = TrialFunctions(M)
(phi, w) = TestFunctions(M)

# Define facet normal. NB: n('+') is the outward normal when viewed
# from Cell '+' and n('-') is the outward normal when viewed from
# '-'. Note that FFC convention is that '+' is '0' and '-' is 1,
# marking the ICS as 0 and the rest as 1 ensures that '+' corresponds
# to the ICS outward normal.
dx = Measure("dx", domain=mesh, subdomain_data=cs)
n = FacetNormal(mesh)('-')

# Redefine dS based on information (in gamma) about interface
dS = Measure("dS", domain=mesh, subdomain_data=gamma)

# Material parameters
C_m = Constant(2.0)

# Conductivity as piecewise constant function
sigma = Function(U)
dsigma = TestFunction(U)
hK = CellVolume(mesh)
assemble((1/hK)*Constant(7)*dsigma*dx(0) + (1/hK)*Constant(3)*dsigma*dx(1),
         tensor=sigma.vector())

# One piece that is needed to get the orientation for u_i - u_e
color_f = Function(U)
assemble((1/hK)*Constant(0)*dsigma*dx(0) + (1/hK)*Constant(1)*dsigma*dx(1),
         tensor=color_f.vector())

GS = lambda u, K=color_f: conditional(gt(K('+'), K('-')), u('+'), u('-'))
# If the + is smalle pick + the smaller
SS = lambda u, K=color_f: conditional(le(K('+'), K('-')), u('+'), u('-'))

Jump = lambda u: GS(u) - SS(u)

# Parameters for I_ion
V_rest_value = -90.0
V_rest = Constant(V_rest_value)
V_eq = Constant(0.0)
g_leak = Constant(6.0e-2)
alpha = 2.0
t0 = 0.0
g_syn_stim = 2.5*50

# Set up synaptic input
x = SpatialCoordinate(mesh)
g_syn = conditional(x[0] <= 20.e-4, g_syn_stim, 0.0)

# Define initial condition (and function for previous solution) for v
v0 = Function(V)
v0.vector()[:] = V_rest_value

# Define timestep
dt_value = 0.02
dt = Constant(dt_value)

# NB: Use a Constant to represent time 
t = Constant(dt_value) 

# Define time dependent term for the stimulus in terms of time t
I_s = exp(-(t-t0)/alpha)
# Define I_ion:
# NOTE: unlike in multdim I_ion is explicit (in terms of v0 not v)
# though this coule probably be lifted
I_ion = g_leak*(v0('+') - V_rest) + g_syn*I_s*(V_rest-V_eq)
# dv/dt = ... can be expanded written in terms of v as 
v = (dt/C_m)*(dot(-s('-'), n) - I_ion) + v0('+')

F = ((inner(div(s), w) + 1.0/sigma*inner(s, phi) + inner(div(phi), u))*dx()
     - inner(v, dot(phi('-'), n))*dS(1))

# Split into left and right-hand sides automatically (error prone to
# do by hand)
(a, L) = system(F)

# Define some points to e.g. evaluate solution in
p = (35*10**-4, 15*10**-4)
p2 = (5*10**-4, 15*10**-4)
p3 = (10*10**-4, 15*10**-4)
    
# Assemble the left-hand side matrix 
A, b = map(assemble, (a, L))

# Solve by time-stepping
m = Function(M)
# NOTE: in the time stepping loop we need to update v0 as a function in
# V. To this end we will project the difference between potentials on
# the boundary onto V. 
flux, potential = split(m)
# The projection problem boils down to assembling into v0 the following
# linear form. The factor 1/FacetArea is like a mass matrix inverse
beta = TestFunction(V)
hA = FacetArea(mesh)

solver = LUSolver(A, 'umfpack')
for k in range(10):
    print("Solving at t = %g, %d unknowns" % (float(t), M.dim()))

    # Assemble right-hand side (changes with time, so need to reassemble)
    assemble(L, tensor=b)
    
    # Solve linear system of equations for m:
    solver.solve(m.vector(), b)

    # V0 update, note that beta is restricted here just to appeace FFC;
    # it is single valued on the facets
    assemble((1/avg(hA))*inner(Jump(potential), beta('+'))*dS(1, metadata={'quadrature_degree': 0}),
             tensor=v0.vector())

    flux_h, potential_h = m.split(deepcopy=True)                          
    # Print norm of v0 in lack of better options
    print("Max flux:", flux_h.vector().max())
    print("Min flux:", flux_h.vector().min())

    print("Max pot:", potential_h.vector().max())
    print("Min pot:", potential_h.vector().min())
    
    print("Max v:", v0.vector().max())
    print("Min v:", v0.vector().min())

    # Update time  
    t.assign(float(t + dt))
    
print("Size %d, time to solve (s): %g" % (M.dim(), timer.stop()))
