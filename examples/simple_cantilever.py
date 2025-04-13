import meshzoo
import numpy as np
import torch
import gpytop

from torchfem import Planar
from torchfem.elements import linear_to_quadratic
from torchfem.materials import IsotropicElasticityPlaneStress

# Set double precision
torch.set_default_dtype(torch.float64)

# Material model (plane stress)
material = IsotropicElasticityPlaneStress(E=1000.0, nu=0.3)

points, cells = meshzoo.rectangle_quad(
    np.linspace(0.0, 2.0, 13),
    np.linspace(0.0, 1.0, 7),
    cell_type="quad4",
)

nodes = torch.tensor(points, dtype=torch.get_default_dtype())
elements = torch.tensor(cells.tolist())

# Create model
cantilever = Planar(nodes, elements, material)

# Load at tip
tip = (nodes[:, 0] == 2.0) & (nodes[:, 1] == 0.5)
cantilever.forces[tip, 1] = -1.0

# Constrained displacement at left end
left = nodes[:, 0] == 0.0
cantilever.constraints[left, :] = True

# Thickness
cantilever.thickness[:] = 0.1

# Solve
u, f, sigma, F, alpha = cantilever.solve()

gpytop.save_as_xdmf(
    "cantilever",
    u,
    nodes,
    elements,
    celltype="quad",
    path="./",
)