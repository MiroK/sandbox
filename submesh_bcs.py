# We illustrate how to use data on boundary meshes (and similar)
# to set boundary conditions. In general we assume u in V=V(Omega)
# is a function which we want to partially reconstruct using
# ub = Vb(Gamma). Here Gamma is some subset of Omega.
from dolfin import *
import numpy as np
import os

f = Expression('x[0]+x[1]', degree=1)

# Generate data for only part of the mesh where they should be
# used. Here we choose to have data on the entire boundary and
# the idea is then to use these to construct boundary conditions
# for functions on the entire(enclosed by the boundary) domain.
if not os.path.exists('bdry_mesh.xdmf'):
    assert MPI.comm_world.size == 1

    mesh = UnitSquareMesh(32, 32)  # Omega
    bmesh = BoundaryMesh(mesh, 'exterior')  # Gamma
    # Now suppose there is data on Gamma ...
    Vb = FunctionSpace(bmesh, 'CG', 1)
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
Vb = FunctionSpace(bmesh, 'CG', 1)
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
CompiledSubDomain('near(x[0]*(1-x[0]), 0) || near(x[1]*(1-x[1]), 0)').mark(facet_f, 1)

V = FunctionSpace(mesh, 'CG', 1)
# We can use DirichletBC to extract degree of freedom of V that
# reside on gamma, that is, where facet_f == 1
bc = DirichletBC(V, Constant(0), facet_f, 1)
# Due to parallelism, the process does not own (has write acccess) to
# all the dofs it can read so we need to get only the owned ones
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
# Let's get ub(xi)
print(f'Process {MPI.comm_world.rank} asking for {len(owned)} dofs')
bdry_values = np.array([ub(xi) for xi in x[bdry_dofs_owned]])

# It remains to set them
u = Function(V)
u_vec = u.vector()
u_array = u_vec.get_local()
# We set those we owned to computed values
u_array[bdry_dofs_owned] = bdry_values
# And assign
u_vec.set_local(u_array)
u_vec.apply('insert')

as_backend_type(u_vec).update_ghost_values()

# Now we check that ub matches u on the boundary in the sense
# that the difference is 0 in L^2
e = sqrt(abs(assemble(inner(u - f, u - f)*ds)))
assert e < 1E-15

# Function u can now be used in construction bcs as follows
bc = DirichletBC(V, u, facet_f, 1)

# And we check that if such bcs are applied we get f on the
# boundary
foo = interpolate(Constant(1), V)
bc.apply(foo.vector())

e = sqrt(abs(assemble(inner(foo - f, foo - f)*ds)))
assert e < 1E-15

# Expected output
# (gmshnics) mirok@evalApply:sandbox|$ dolfin-version
# 2019.1.0
# (gmshnics) mirok@evalApply:sandbox|$ mpirun -np 4 python submesh_bcs.py
# Process 0 asking for 29 dofs
# Process 2 asking for 30 dofs
# Process 3 asking for 36 dofs
# Process 1 asking for 33 dofs
