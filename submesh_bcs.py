# We illustrate how to use data on boundary meshes (and similar)
# to set boundary conditions. In general we assume u in V=V(Omega)
# is a function which we want to partially reconstruct using
# ub = Vb(Gamma). Here Gamma is some subset of Omega.
from dolfin import *
import numpy as np
import os

fscalar = Expression('x[0]+x[1]', degree=1)
fvector = Expression(('x[0]+x[1]', 'x[0]-2*x[1]'), degree=1)

f = fvector
# Pick space based on data scalar/vector
get_space = {0: lambda mesh: FunctionSpace(mesh, 'CG', 1),
             1: lambda mesh: VectorFunctionSpace(mesh, 'CG', 1)}[len(f.ufl_shape)] 

zero = {0: Constant(0), 1: Constant((0, 0))}[len(f.ufl_shape)]
one = {0: Constant(1), 1: Constant((1, 1))}[len(f.ufl_shape)]

# We will have data only one piece of boundary such that with MPI
# it will be likely that on some process the will be none
chi = 'near(x[0], 0) && (x[1] < 0.5 + DOLFIN_EPS)'

force = False
# Generate data for only part of the mesh where they should be
# used. Here we choose to have data on the entire boundary and
# the idea is then to use these to construct boundary conditions
# for functions on the entire(enclosed by the boundary) domain.
if force or not os.path.exists('bdry_mesh.xdmf'):
    assert MPI.comm_world.size == 1

    mesh = UnitSquareMesh(32, 32)  # Omega
    bmesh = BoundaryMesh(mesh, 'exterior')  # Gamma

    cell_f = MeshFunction('size_t', bmesh, 1, 0)
    CompiledSubDomain(chi).mark(cell_f, 1)
    bmesh = SubMesh(bmesh, cell_f, 1)
    
    # Now suppose there is data on Gamma ...
    Vb = get_space(bmesh)
    ub = interpolate(f, Vb)
    # ... and store it
    with XDMFFile(bmesh.mpi_comm(), 'bdry_mesh.xdmf') as file:
        file.write(bmesh)
    
    with XDMFFile(bmesh.mpi_comm(), 'bdry_data.xdmf') as file:
        file.write_checkpoint(ub, "ub", 0, XDMFFile.Encoding.HDF5, append=False)

# The real situation is that we want to reconstruct. So let
# us first load the mesh ...
bmesh = Mesh(MPI.comm_self)
with XDMFFile(bmesh.mpi_comm(), 'bdry_mesh.xdmf') as file:
    file.read(bmesh)
# ... and after creating the corresponding function space ...
Vb = get_space(bmesh)
ub = Function(Vb)
# ... also load the data
with XDMFFile(bmesh.mpi_comm(), 'bdry_data.xdmf') as file:
    file.read_checkpoint(ub, "ub", 0)
# NOTE: we load the mesh of the boundary with MPI.comm_self, i.e. every
# process sees the entire boundary mesh and the mesh is not shared. This
# allows us to evaluate dofs below without further communication between
# processes because every requiest can be satisfied on every CPU.
# It is assumed that mesh is small. 

# The full mesh, however, is shared. So each CPU will only make requests
# for dofs of V that belong to it
mesh = UnitSquareMesh(32, 32)    
# Now we want to create a function on Omega which matched ub
# on the boundary (or in general some subset of Omega). We assume
# that the subset is marked in Omega, i.e. that we have
facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
CompiledSubDomain(chi).mark(facet_f, 1)
ds = Measure('ds', domain=mesh, subdomain_data=facet_f)

V = get_space(mesh)
# We can use DirichletBC to extract degree of freedom of V that
# reside on gamma, that is, where facet_f == 1
bc = DirichletBC(V, zero, facet_f, 1)
# Due to parallelism, the process does not own (has write acccess) to
# all the dofs it can read so we need to get only the owned ones.
dm = V.dofmap()
first, last = dm.ownership_range()
bdry_dofs_local = np.array(list(bc.get_boundary_values().keys()))
bdry_dofs_global = first + bdry_dofs_local
# Tadaaa
owned = np.where(np.logical_and(bdry_dofs_global >= first, bdry_dofs_global < last))[0]
bdry_dofs_owned = bdry_dofs_local[owned]

# We not want to set these degrees of freedom such that it holds that
# u|gamma = ub. For Lagrange elements this means that it suffices to
# to evaluated ub at xi, the coordinates associated with i-th degree
# of freedom and set the dof for u.
gdim = mesh.geometry().dim()
x = V.tabulate_dof_coordinates().reshape((-1, gdim))

# The function we want to buld
u = Function(V)
u_vec = u.vector()
u_array = u_vec.get_local()
# Let's get ub(xi) ...
print(f'Process {MPI.comm_world.rank} asking for {len(owned)} dofs')
# ... but only compute where ti makes sense

if len(bdry_dofs_owned):
    # The sort here is used to make it easier to get dofs for vector components
    bdry_dofs_owned = np.sort(bdry_dofs_owned)
    # Scalar we take all of ub(xi) ...
    bdry_values = np.array([ub(xi) for xi in x[bdry_dofs_owned]])
    # ... but for vector `bdry_values` array is n-times longer as
    # it has all ub(xi) while for V.sub(i) space we only need ub(xi)[i]
    if V.num_sub_spaces() > 1:
        nsubs = V.num_sub_spaces()
        # Here we exploit dof ordering of vector elements
        bdry_values = np.ravel(np.column_stack([bdry_values[sub::nsubs, sub]
                                                for sub in range(V.num_sub_spaces())]))

    assert len(bdry_values) == len(bdry_dofs_owned)
    # We set those we owned to computed values
    u_array[bdry_dofs_owned] = bdry_values
# So here some u_array are filled and other can be empty but every body
# should participate in setting in values nevertheless
u_vec.set_local(u_array)
u_vec.apply('insert')

as_backend_type(u_vec).update_ghost_values()

# Now we check that ub matches u on the boundary in the sense
# that the difference is 0 in L^2
e = sqrt(abs(assemble(inner(u - f, u - f)*ds(1))))
assert e < 1E-15, e

# Function u can now be used in construction bcs as follows
bc = DirichletBC(V, u, facet_f, 1)

# And we check that if such bcs are applied we get f on the
# boundary
foo = interpolate(one, V)
bc.apply(foo.vector())

e = sqrt(abs(assemble(inner(foo - f, foo - f)*ds(1))))
assert e < 1E-15
# (xii) mirok@evalApply:sandbox|(master)*$ mpirun -np 4 python submesh_bcs.py 
# Process 0 asking for 26 dofs
# Process 1 asking for 0 dofs
# Process 2 asking for 0 dofs
# Process 3 asking for 8 dofs
