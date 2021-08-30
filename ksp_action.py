from dolfin import *
from petsc4py import PETSc

mesh = UnitSquareMesh(5, 5)
V = FunctionSpace(mesh, 'DG', 0)
u, v = TrialFunction(V), TestFunction(V)

a = inner(u, v)*dx
A = PETScMatrix()
assemble(a, A)

hK = CellVolume(mesh)
ia = (1/hK**2)*inner(u, v)*dx
iA = PETScMatrix()
assemble(ia, iA)

b = interpolate(Constant(1), V).vector()
x = as_backend_type(b)

ksp = PETSc.KSP().create()
ksp.setType('preonly')
ksp.setOperators(A.mat(), None)

#pc = ksp.getPC()
#pc.setType('none')
# pc.setUp()

ksp.setUp()

Ax, y = x.copy(), x.copy()
ksp.solve(y.vec(), x.vec())

A.mult(x, Ax)

print((Ax - y).norm('l2'))
