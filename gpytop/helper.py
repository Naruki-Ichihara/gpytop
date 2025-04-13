import numpy as np
import torch
import meshio

def save_as_xdmf(name: str, data: torch.Tensor, nodes: torch.Tensor, elements: torch.Tensor, celltype: str = "quad", path=""):
    """
    Save mesh and data to an XDMF file.

    Args:
        name (str): Name of the output file.
        data (torch.Tensor): Data to be saved.
        nodes (torch.Tensor): Node coordinates.
        elements (torch.Tensor): Element connectivity.
        celltype (str): Type of the mesh cells. Default is "quad".
        path (str): Path to save the file. Default is current directory.
    """

    x3d = np.hstack([nodes.numpy(), np.zeros((nodes.shape[0], 1))])
    u3d = np.hstack([data.numpy(), np.zeros((data.shape[0], 1))])

    # Create a mesh object
    mesh = meshio.Mesh(
        points=x3d,
        cells=[(celltype, elements.numpy())],
        point_data={name: u3d},
    )

    # Write the mesh to an XDMF file
    mesh.write(path + name + ".xdmf")