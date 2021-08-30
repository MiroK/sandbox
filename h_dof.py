from dolfin import *
import matplotlib.pyplot as plt
import numpy as np


def get_dofs(dim, ncells):
    meshgen = {1: UnitIntervalMesh,
               2: UnitSquareMesh,
               3: UnitCubeMesh}[dim]

    hs, dofs = [], []
    for n in ncells:
        mesh = meshgen(*(n, )*dim)
        V = FunctionSpace(mesh, 'DG', 0)

        hs.append(mesh.hmin())
        dofs.append(V.dim())

    return hs, dofs

ncells = tuple(2**i for i in range(7))

fig, ax = plt.subplots()

for dim in (1, 2, 3):
    hs, dofs = get_dofs(dim, ncells)
    ax.loglog(hs, dofs, '-x', label=str(dim))
    print(dim, np.polyfit(np.log(hs), np.log(dofs), deg=1))
plt.legend()
plt.show()
    
