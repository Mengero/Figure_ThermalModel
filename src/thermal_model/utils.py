"""
Utility functions for the thermal model.
"""

import numpy as np

def tridiag(n, sub_val=-1, main_val=2, super_val=-1):
    """
    Create a dense tridiagonal matrix of size n x n.

    Parameters:
        n (int): Size of the matrix (number of rows/columns).
        sub_val (float): Value for the sub-diagonal elements (default: -1).
        main_val (float): Value for the main diagonal elements (default: 2).
        super_val (float): Value for the super-diagonal elements (default: -1).

    Returns:
        A (ndarray): A tridiagonal matrix.
    """
    sub_diag = sub_val * np.ones(n-1)
    main_diag = main_val * np.ones(n)
    super_diag = super_val * np.ones(n-1)
    
    A = np.diag(main_diag, 0) + np.diag(sub_diag, -1) + np.diag(super_diag, 1)
    return A

def add_index(index, indices_array):
    """
    Check if 'index' is already in 'indices_array'. 
    If yes, raise an error; otherwise, append and return the new array.
    
    Parameters:
        index (int): The index to add
        indices_array (ndarray): Array of existing indices
        
    Returns:
        ndarray: Updated array with the new index appended
        
    Raises:
        ValueError: If the index is already in the array
    """
    if np.any(indices_array == index):
        print(f"Error: index {index} already used.")
        raise ValueError("Index used twice")
    return np.append(indices_array, index)

def calculate_expected_boundary_indices(nx, ny, nz):
    """
    Calculate the expected number of boundary indices.
    
    Parameters:
        nx, ny, nz (int): Number of cells in each direction
        
    Returns:
        int: Expected number of boundary indices
    """
    # Count vertices (8)
    vertices = 8
    
    # Count edges: 4 edges each for x, y, and z directions
    edges_x = 4 * (nx - 2)
    edges_y = 4 * (ny - 2)
    edges_z = 4 * (nz - 2)
    
    # Count faces: 2 faces each for xy, xz, and yz planes
    faces_xy = 2 * (nx - 2) * (ny - 2)
    faces_xz = 2 * (nx - 2) * (nz - 2)
    faces_yz = 2 * (ny - 2) * (nz - 2)
    
    return vertices + edges_x + edges_y + edges_z + faces_xy + faces_xz + faces_yz
