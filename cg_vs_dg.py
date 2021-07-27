from dolfin import *
import numpy as np

mesh = UnitSquareMesh(32, 32)

facet_f = MeshFunction('size_t', mesh, 1, 0)
facet_f.array()[:20] = 1

dS = Measure('dS', domain=mesh, subdomain_data=facet_f)

V = FunctionSpace(mesh, 'CG', 1)
v = Function(V)
v.vector().set_local(np.random.rand(v.vector().local_size()))

L = inner(v('+') - v('-'), v('+') - v('-'))*dS(1)
print(assemble(L))


V = FunctionSpace(mesh, 'DG', 1)
v = Function(V)
v.vector().set_local(np.random.rand(v.vector().local_size()))

L = inner(v('+') - v('-'), v('+') - v('-'))*dS(1)
print(assemble(L))

class Random(UserExpression):
    def value_shape(self):
        return (2, )

    def eval(self, values, x):
        values[:] = np.random.rand(2)

r = Random()

V = FunctionSpace(mesh, 'RT', 1)
v = interpolate(r, V)

n = FacetNormal(mesh)
L = inner(jump(v, n), jump(v, n))*dS(1)
print(assemble(L))

R = Constant(((0, 1), (-1, 0)))
tangent = dot(R, n)

n = FacetNormal(mesh)
L = inner(jump(v, tangent), jump(v, tangent))*dS(1)
print(assemble(L))

V = VectorFunctionSpace(mesh, 'CG', 1)
v = interpolate(r, V)

n = FacetNormal(mesh)
L = inner(jump(v, n), jump(v, n))*dS(1)
print(assemble(L))

R = Constant(((0, 1), (-1, 0)))
tangent = dot(R, n)

n = FacetNormal(mesh)
L = inner(jump(v, tangent), jump(v, tangent))*dS(1)
print(assemble(L))
