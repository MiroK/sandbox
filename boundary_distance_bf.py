from dolfin import *
import numpy as np
import operator, os, json
from functools import reduce
from scipy.spatial import cKDTree
from fenics2grid import *


def harmonic_extension(facet_f, tags, elm=None):
    '''Solve -Delta u = 1 with u = 0 on bdries with tags'''
    mesh = facet_f.mesh()
    
    if elm is None:
        elm = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, elm)
    
    u, v = TrialFunction(V), TestFunction(V)
    f = Constant(1)
    bcs = [DirichletBC(V, Constant(0), facet_f, tag) for tag in tags]

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    A, b = assemble_system(a, L, bcs)

    solver = PETScKrylovSolver('cg', 'hypre_amg')
    solver.parameters['relative_tolerance'] = 1E-12

    A, b = map(as_backend_type, (A, b))
    solver.set_operators(A, A)

    uh = Function(V)
    solver.solve(uh.vector(), b)

    return uh


def boundary_distance_bf(facet_f, tags, p=2, degree=2, elm=None, **tree_kwargs):
    '''
    Compute appoximate l^p distance function from tagged facets. The 
    approximation is based on CG^degree space and produces a function
    in elm-space
    '''
    mesh = facet_f.mesh()
    if elm is None:
        elm = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, elm)
    phi = Function(V)
    
    gdim = mesh.geometry().dim()
    # For boundary we use some CG space and pick its boundary dofs
    Vbdry = FunctionSpace(mesh, 'CG', degree)

    bcs = [DirichletBC(Vbdry, phi, facet_f, tag) for tag in tags]
    x = Vbdry.tabulate_dof_coordinates().reshape((-1, gdim))
    # Coordinates of local boundary dofs
    bcs_dofs = np.fromiter(
        reduce(operator.or_, (bc.get_boundary_values().keys() for bc in bcs)),
        dtype='uintp'
    )
    first, last = Vbdry.dofmap().ownership_range()
    bcs_dofs = bcs_dofs[((first + bcs_dofs) < last)*(first >= 0)]

    comm = mesh.mpi_comm()    

    x_local = x[bcs_dofs]
    x_all = comm.alltoall([x_local]*comm.size)
    x_all = np.row_stack(x_all)  # All bdry dofs from all cpus

    # Now we query for dofs to build the function in V
    x = V.tabulate_dof_coordinates().reshape((-1, gdim))
    
    tree = cKDTree(x_all)  # NOTE: KDTree is pure python and slow
    values, _ = tree.query(x, p=p, **tree_kwargs)
    # values = np.array([np.min(np.linalg.norm(x_all - xi, 2, axis=1)) for xi in x])

    f_vec = phi.vector()
    f_vec.set_local(values)
    f_vec.apply('insert')
    as_backend_type(f_vec).update_ghost_values()

    Q = FunctionSpace(mesh, 'DG', 0)
    q = TestFunction(Q)
    hK = CellVolume(mesh)

    # We were solving |grad(phi)| - 1 = 0 is this is very crude way of
    # checking
    dx0 = Measure('dx', metadata={'quadrature_degree': 0})
    error = Function(Q)
    assemble((1/hK)*inner(q, abs(sqrt(dot(grad(phi), grad(phi)))-Constant(1)))*dx0,
             tensor=error.vector())
    
    return phi, error


def square_bdry_distance(facet, tags, elm=None):
    '''Distance function of [0, 1]^2'''
    mesh = facet_f.mesh()
    assert mesh.geometry().dim() == 2
    assert all(t in (1, 2, 3, 4) for t in tags)

    if elm is None:
        elm = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, elm)
    
    f = Function(V)
    dofs_x = V.tabulate_dof_coordinates().reshape((-1, 2))
    x, y = dofs_x.T
    # Pick only the relevant slices
    possible = np.c_[x, 1-x, y, 1-y][:, np.array(tags)-1]
    dist = np.min(possible, axis=1)

    f.vector().set_local(dist)

    return f

# --------------------------------------------------------------------

if __name__ == '__main__':
    
    boundaries = [CompiledSubDomain('near(x[0], 0)'),
                  CompiledSubDomain('near(x[0], 1)'),
                  CompiledSubDomain('near(x[1], 0)'),
                  CompiledSubDomain('near(x[1], 1)')]

    # tags = (1, 2, 3)
    # Check convergence
    # for n in (8, 16, 32, 64, 128, 256):
    #     mesh = UnitSquareMesh(n, n, 'crossed')
    #     facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    #     for tag, domain in enumerate(boundaries, 1):
    #         domain.mark(facet_f, tag)
        
    #     # dist = harmonic_extension(facet_f, tags)
    #     t = Timer('distance')
    #     dist, error = boundary_distance_bf(facet_f, tags)
    #     dt = t.stop()

    #     dist0 = square_bdry_distance(facet_f, tags, elm=dist.function_space().ufl_element())
        
    #     if mesh.mpi_comm().rank == 0:
    #         print(mesh.num_cells(), '->', dt)
    #     File('dist.pvd') << dist
    #     File('error.pvd') << error

    #     e = dist - dist0        
    #     print(sqrt(abs(assemble(inner(e, e)*dx))))

    
    # meshfile = './191_parenchyma_mesh.h5'
    # mesh = Mesh()
    # with HDF5File(mesh.mpi_comm(), meshfile, 'r') as in_:
    #     in_.read(mesh, '/mesh', False)
    # meshfile = "/home/basti/Programming/meshslice/meshmp1.00e-01/mesh.xml"

    p = 1

    path = "/home/basti/Programming/FEniCS/parenchyma/code/2d_simulation_inputs/"
    meshfile = path + "testmesh.xml"
    outfolder = path + "distancefunctions/p" + str(p) + "/"
    mesh = Mesh(meshfile)

    facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    DomainBoundary().mark(facet_f, 1)
    
    t = Timer('distance')
    dist, error = boundary_distance_bf(facet_f, tags=(1, ), p=p)
    dt = t.stop()
   
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    # save dist interpolated to npy grid
    sol = togrid(f=dist, mesh=mesh, N=256)
    np.save(outfolder + "distancefun.npy", sol)



    with open(outfolder + 'mesh_used.json', 'w') as fp:
        json.dump({"meshpath": meshfile}, fp)

    if mesh.mpi_comm().rank == 0:
        print(dist.function_space().dim(), '->', dt)

    data_h5 = HDF5File(mesh.mpi_comm(), outfolder + 'dist_brain.h5', 'w')
    data_h5.write(dist, 'distancefun')
    data_h5.close()

    File(outfolder + 'dist_brain.pvd') << dist



    # File(outfolder + 'error_brain.pvd') << error

    # dist = harmonic_extension(facet_f, tags=(1, ), elm=dist.function_space().ufl_element())
    # File(outfolder + 'harm_brain.pvd') << dist
