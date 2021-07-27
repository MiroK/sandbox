# Mixed multidimensional formulation of EMI model. We solve for flux, potential and
# transmembrane potential. As the name suggests transmembrane potential should only
# be defined on the membrane (marked by gamma below). However, a caveat of the setup
# here is that it is defined on ALL the facets of the mesh; only a subset of them
# form the mesh of the membrane. The rest is therefore elliminated by artificial
# boundary condition to obtain an invertible system. Still, the size of the system is
# considerably larger than what it needs to be.

# NOTE: the form of the system to be solved at each time level is Ax = b
# where
#
#     [(1/sigma)*I, grad, T';            [flux;
# A =  div,         0,    0;   and x =    potential;
#      T,           0,    0]              transmembrane potential]

# Dependencies: FEniCS 2019.+  

__author__ = "Marie E. Rognes (meg@simula.no) and Karoline Horgmo Jaeger (karolihj@simula.no)"

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
V = FiniteElement("Discontinuous Lagrange Trace", mesh.ufl_cell(), 0)

M = FunctionSpace(mesh, MixedElement(S, U, V))
U = FunctionSpace(mesh, U)
V = FunctionSpace(mesh, V)

# Unknowns and test functions
(s, u, v) = TrialFunctions(M)
(phi, w, beta) = TestFunctions(M)

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

# v, v_ and beta are all continuous but FEniCS doesn't know that, so
# we need to take a restriction ('+' or '-') when integrating over
# interior facets:
v = v('+')
v_ = v0('+')
beta = beta('+')

# HACK: This is a key hack, the space B is defined over the whole
# domain (but should just be defined over gamma). Set all degrees of
# freedom outside gamma to zero, and hope that it just works[TM].
bc = DirichletBC(M.sub(2), 0.0, gamma, 0)

# NB: Use a Constant to represent time 
t = Constant(dt_value) 

# Define time dependent term for the stimulus in terms of time t
I_s = exp(-(t-t0)/alpha)

# Define I_ion:  
#I_ion = g_leak*(v - V_rest) + g_syn*I_s*(v-V_eq)
I_ion = g_leak*(v - V_rest) + g_syn*I_s*(V_rest-V_eq)

F = ((inner(div(s), w) + 1.0/sigma*inner(s, phi) + inner(div(phi), u))*dx()
     - dot(phi('+'), n)*v*dS(1)
     + (C_m*(v - v_) + dt*dot(s('+'), n) + dt*I_ion)*beta*dS(1))

# Split into left and right-hand sides automatically (error prone to
# do by hand)
(a, L) = system(F)

# Define some points to e.g. evaluate solution in
p = (35*10**-4, 15*10**-4)
p2 = (5*10**-4, 15*10**-4)
p3 = (10*10**-4, 15*10**-4)
    
# Assemble the left-hand side matrix 
A, b = map(assemble, (a, L))
bc.apply(A)

# Solve by time-stepping
m = Function(M)

solver = LUSolver(A, 'mumps')
for k in range(10):
    print("Solving at t = %g, %d unknowns" % (float(t), M.dim()))

    # Assemble right-hand side (changes with time, so need to reassemble)
    assemble(L, tensor=b)
    
    # Apply boundary conditions to right-hand side
    bc.apply(b)
    
    # Solve linear system of equations for m:
    solver.solve(m.vector(), b)

    # Assign v (2'nd component of m) to v0
    flux_h, potential_h, transm_potential_h = m.split(deepcopy=True)
    v0.assign(transm_potential_h)

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
print(f'Redundant dofs {len(bc.get_boundary_values())}')

# Quick and dirty regression test
print("|m|_l2 = ", m.vector().norm("l2"))
