import numpy as np
import scipy
from scipy.sparse import diags, block_diag, kron, identity, lil_matrix
from scipy.sparse.linalg import LinearOperator
from typing import Dict, Tuple, List, Optional
from thermal_parameters import ThermalParameters
from calculate_elements import calculate_region_elements

class ElementCoordinates:
    """Class for managing element coordinates in a layer."""
    
    def __init__(self, params: ThermalParameters, region_id: str):
        """
        Initialize element coordinates for a region.
        Coordinates are set up with:
        - x centered around 0 (-width/2 to +width/2)
        - y from bottom (0 to height)
        - z from bottom (0 to thickness)
        
        Args:
            params: ThermalParameters object
            region_id: ID of the region
        """
        self.region_id = region_id
        self.params = params
        
        # Get region dimensions
        self.Lx, self.Ly, self.Lz = params.get_region_dimensions(region_id)
        
        # Calculate number of points and spacing
        _, self.Nx, self.Ny, self.Nz, nominal_dx, nominal_dy, nominal_dz, _ = calculate_region_elements(params, region_id)
        
        # Recalculate actual element sizes to ensure they divide the dimensions evenly
        self.dx = self.Lx / (self.Nx - 1) if self.Nx > 1 else self.Lx
        self.dy = self.Ly / (self.Ny - 1) if self.Ny > 1 else self.Ly
        self.dz = self.Lz / (self.Nz - 1) if self.Nz > 1 else self.Lz
        
        # Initialize starting points for coordinates
        self.x_ini = -self.Lx/2  # Start x from -width/2 to center around 0
        self.y_ini = 0  # Start y from bottom
        self.z_ini = 0  # Start z from bottom
        
        # Pre-calculate all point coordinates
        self.coordinates = []
        for k in range(self.Nz):
            for j in range(self.Ny):
                for i in range(self.Nx):
                    # Calculate coordinates with x centered around 0
                    x = self.x_ini + i * self.dx
                    y = self.y_ini + j * self.dy
                    z = self.z_ini + k * self.dz
                    self.coordinates.append((x, y, z))
    
    def get_global_index(self, nx:int, ny:int, i: int, j: int, k: int) -> int:
        """
        Get the global index from 3D indices (i,j,k).
        
        Args:
            i: x-direction index
            j: y-direction index
            k: z-direction index
            
        Returns:
            Global index
        """
        return k * ny * nx + j * nx + i

    def get_coordinates(self, global_index: int) -> Tuple[float, float, float]:
        """
        Get the (x,y,z) coordinates of a point element.
        
        Args:
            global_index: Global index of the element
            
        Returns:
            Tuple of (x,y,z) coordinates
        """
        return self.coordinates[global_index]

    def get_3d_indices(self, global_index: int) -> Tuple[int, int, int]:
        """
        Get the 3D indices (i,j,k) from a global index.
        
        Args:
            global_index: Global index of the element
            
        Returns:
            Tuple of (i,j,k) indices
        """
        k = global_index // (self.Ny * self.Nx)
        remainder = global_index % (self.Ny * self.Nx)
        j = remainder // self.Nx
        i = remainder % self.Nx
        return (i, j, k)

    def get_coordinates_from_3d(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """
        Get the (x,y,z) coordinates from 3D indices.
        Returns coordinates with:
        - x centered around 0 (-width/2 to +width/2)
        - y from bottom (0 to height)
        - z from bottom (0 to thickness)
        
        Args:
            i: x-direction index
            j: y-direction index
            k: z-direction index
            
        Returns:
            Tuple of (x,y,z) coordinates
        """
        # Calculate x coordinate centered around 0
        x = self.x_ini + i * self.dx
        
        # Calculate y and z from bottom
        y = self.y_ini + j * self.dy
        z = self.z_ini + k * self.dz
        
        return (x, y, z)

def create_tridiagonal_matrix(n, sub_val=-1, main_val=2, super_val=-1):
    """
    Create a sparse tridiagonal matrix of size n x n.
    
    Parameters:
        n (int): Size of the matrix
        sub_val (float): Value for sub-diagonal
        main_val (float): Value for main diagonal
        super_val (float): Value for super-diagonal
        
    Returns:
        scipy.sparse matrix: Tridiagonal matrix
    """
    sub_diag = sub_val * np.ones(n-1)
    main_diag = main_val * np.ones(n)
    super_diag = super_val * np.ones(n-1)
    
    return diags([sub_diag, main_diag, super_diag], [-1, 0, 1], shape=(n, n))

def create_layer_matrix(nx, ny, nz, dx, dy, dz, k):
    """
    Create the 3D Laplacian matrix for a single layer.
    
    Parameters:
        nx, ny, nz (int): Number of elements in each direction
        dx, dy, dz (float): Element sizes in each direction
        k (float): Thermal conductivity
        
    Returns:
        scipy.sparse matrix: 3D Laplacian matrix for the layer
    """
    # Create 1D matrices
    Ax = create_tridiagonal_matrix(nx) / dx**2
    Ay = create_tridiagonal_matrix(ny) / dy**2
    Az = create_tridiagonal_matrix(nz) / dz**2
    
    # Create identity matrices
    Ix = identity(nx)
    Iy = identity(ny)
    Iz = identity(nz)
    
    # Create 3D Laplacian using Kronecker product
    A = kron(kron(Az, Iy), Ix) + kron(kron(Iz, Ay), Ix) + kron(kron(Iz, Iy), Ax)
    A *= k
    return A

def create_layer_coordinates(params: ThermalParameters) -> ElementCoordinates:
    """
    Create coordinate system for metal layer.
    
    Args:
        params: ThermalParameters object
        
    Returns:
        ElementCoordinates for metal layer
    """
    # Get metal layer parameters
    metal_coords = ElementCoordinates(params, "metal_layer")
    return metal_coords

def _apply_thermal_conductivity_modifications(A, global_idx, i, j, k, info, thermal_conductivity, Nx, Ny, Nz, dx_m, dy_m, dz_m, element_type='corner'):
    """
    Apply thermal conductivity modifications based on USERDEF_CONDUCTION boundary conditions.
    Updates both off-diagonal (neighbor) terms and diagonal terms.
    
    Args:
        A: Matrix to modify
        global_idx: Global index of current element
        i, j, k: 3D indices of current element
        info: Region information containing boundary conditions
        thermal_conductivity: Base thermal conductivity
        Nx, Ny, Nz: Grid dimensions
        dx_m, dy_m, dz_m: Element dimensions in meters
        element_type: Type of element ('corner', 'edge', 'surface', 'volume')
    """
    # Check if current element has USERDEF_CONDUCTION boundary condition
    current_element_has_userdef = False
    if "USERDEF_CONDUCTION" in info['boundary_indices']:
        for element in info['boundary_indices']["USERDEF_CONDUCTION"]:
            if element['global_idx'] == global_idx:
                # Get new thermal conductivity and apply to whole row
                new_conductivity = element['bc_data']['conductivity']
                A[global_idx, :] *= new_conductivity / thermal_conductivity
                current_element_has_userdef = True
                break
    
    # If current element doesn't have USERDEF_CONDUCTION, check neighbors
    if not current_element_has_userdef and "USERDEF_CONDUCTION" in info['boundary_indices']:
        # Get neighbor indices and their directions
        neighbor_info = _get_neighbor_indices_with_directions(global_idx, i, j, k, Nx, Ny, Nz, element_type)
        
        # Track diagonal modifications
        diagonal_correction = 0.0
        
        # Check each neighbor and apply new conductivity if neighbor has USERDEF_CONDUCTION
        for element in info['boundary_indices']["USERDEF_CONDUCTION"]:
            neighbor_idx = element['global_idx']
            new_conductivity = element['bc_data']['conductivity']
            
            # Find if this neighbor is in our neighbor list and get its direction
            for neighbor_data in neighbor_info:
                if neighbor_data['idx'] == neighbor_idx:
                    direction = neighbor_data['direction']
                    
                    # Apply new conductivity to off-diagonal term
                    A[global_idx, neighbor_idx] *= new_conductivity / thermal_conductivity
                    
                    # Calculate diagonal correction based on direction
                    if direction in ['x+', 'x-']:
                        # X-direction neighbor: diagonal correction for x-term
                        diagonal_correction += (new_conductivity/thermal_conductivity - 1) / dx_m**2
                    elif direction in ['y+', 'y-']:
                        # Y-direction neighbor: diagonal correction for y-term  
                        diagonal_correction += (new_conductivity/thermal_conductivity - 1) / dy_m**2
                    elif direction in ['z+', 'z-']:
                        # Z-direction neighbor: diagonal correction for z-term
                        diagonal_correction += (new_conductivity/thermal_conductivity - 1) / dz_m**2
                    break
        
        # Apply diagonal correction
        if diagonal_correction != 0.0:
            A[global_idx, global_idx] -= diagonal_correction

def _get_neighbor_indices_with_directions(global_idx, i, j, k, Nx, Ny, Nz, element_type):
    """
    Get neighbor indices with direction information for different element types.
    
    Corner elements: 3 neighbors
    Edge elements: 4 neighbors  
    Surface elements: 5 neighbors
    Volume elements: 6 neighbors
    
    Args:
        global_idx: Global index of current element
        i, j, k: 3D indices of current element
        Nx, Ny, Nz: Grid dimensions
        element_type: Type of element ('corner', 'edge', 'surface', 'volume')
        
    Returns:
        List of dictionaries with 'idx' and 'direction' keys
    """
    neighbors = []
    
    if element_type == 'corner':
        # Corner elements have 3 neighbors (3 surfaces exposed, 3 interior connections)
        if i == 0:
            neighbors.append({'idx': global_idx + 1, 'direction': 'x+'})  # +X direction
        else:  # i == Nx-1
            neighbors.append({'idx': global_idx - 1, 'direction': 'x-'})  # -X direction
            
        if j == 0:
            neighbors.append({'idx': global_idx + Nx, 'direction': 'y+'})  # +Y direction
        else:  # j == Ny-1
            neighbors.append({'idx': global_idx - Nx, 'direction': 'y-'})  # -Y direction
            
        if k == 0:
            neighbors.append({'idx': global_idx + Nx*Ny, 'direction': 'z+'})  # +Z direction
        else:  # k == Nz-1
            neighbors.append({'idx': global_idx - Nx*Ny, 'direction': 'z-'})  # -Z direction
            
    elif element_type == 'edge':
        # Edge elements have 4 neighbors (2 surfaces exposed, 4 interior connections)
        # Add neighbors based on which surfaces are NOT exposed
        
        # X-direction neighbors (if not on X boundary)
        if i != 0:
            neighbors.append({'idx': global_idx - 1, 'direction': 'x-'})  # -X direction
        if i != Nx-1:
            neighbors.append({'idx': global_idx + 1, 'direction': 'x+'})  # +X direction
            
        # Y-direction neighbors (if not on Y boundary)  
        if j != 0:
            neighbors.append({'idx': global_idx - Nx, 'direction': 'y-'})  # -Y direction
        if j != Ny-1:
            neighbors.append({'idx': global_idx + Nx, 'direction': 'y+'})  # +Y direction
            
        # Z-direction neighbors (if not on Z boundary)
        if k != 0:
            neighbors.append({'idx': global_idx - Nx*Ny, 'direction': 'z-'})  # -Z direction
        if k != Nz-1:
            neighbors.append({'idx': global_idx + Nx*Ny, 'direction': 'z+'})  # +Z direction
            
    elif element_type == 'surface':
        # Surface elements have 5 neighbors (1 surface exposed, 5 interior connections)
        
        # X-direction neighbors
        if i != 0:
            neighbors.append({'idx': global_idx - 1, 'direction': 'x-'})  # -X direction
        if i != Nx-1:
            neighbors.append({'idx': global_idx + 1, 'direction': 'x+'})  # +X direction
            
        # Y-direction neighbors
        if j != 0:
            neighbors.append({'idx': global_idx - Nx, 'direction': 'y-'})  # -Y direction
        if j != Ny-1:
            neighbors.append({'idx': global_idx + Nx, 'direction': 'y+'})  # +Y direction
            
        # Z-direction neighbors
        if k != 0:
            neighbors.append({'idx': global_idx - Nx*Ny, 'direction': 'z-'})  # -Z direction
        if k != Nz-1:
            neighbors.append({'idx': global_idx + Nx*Ny, 'direction': 'z+'})  # +Z direction
            
    elif element_type == 'volume':
        # Volume elements have 6 neighbors (interior elements, all 6 connections)
        neighbors.append({'idx': global_idx - 1, 'direction': 'x-'})      # -X direction
        neighbors.append({'idx': global_idx + 1, 'direction': 'x+'})      # +X direction
        neighbors.append({'idx': global_idx - Nx, 'direction': 'y-'})     # -Y direction
        neighbors.append({'idx': global_idx + Nx, 'direction': 'y+'})     # +Y direction
        neighbors.append({'idx': global_idx - Nx*Ny, 'direction': 'z-'})  # -Z direction
        neighbors.append({'idx': global_idx + Nx*Ny, 'direction': 'z+'})  # +Z direction
    
    return neighbors

def _get_neighbor_indices(global_idx, i, j, k, Nx, Ny, Nz, element_type):
    """
    Get neighbor indices for different element types (backward compatibility function).
    
    Args:
        global_idx: Global index of current element
        i, j, k: 3D indices of current element
        Nx, Ny, Nz: Grid dimensions
        element_type: Type of element ('corner', 'edge', 'surface', 'volume')
        
    Returns:
        Set of neighbor global indices
    """
    neighbor_info = _get_neighbor_indices_with_directions(global_idx, i, j, k, Nx, Ny, Nz, element_type)
    return {neighbor['idx'] for neighbor in neighbor_info}

def _apply_mapped_boundary_condition(A, global_idx, i, j, k, coords, mapped_bcs, start_idx, Nx, Ny, dx_m, dy_m):
    """
    Apply MAPPED boundary conditions to a specific element if conditions are met.
    
    Args:
        A: Global system matrix
        global_idx: Global index of current element
        i, j, k: 3D indices of current element
        coords: ElementCoordinates object
        mapped_bcs: List of MAPPED boundary conditions
        start_idx: Starting index for this region
        Nx, Ny: Grid dimensions
        dx_m, dy_m: Grid spacing in meters
    """
    if not mapped_bcs:
        return
        
    # Get element coordinates
    x, y, z = coords.get_coordinates_from_3d(i, j, k)
    
    for bc in mapped_bcs:
        face_selection = bc.get('face_selection', '+x')
        start_location = bc.get('start_location', 0.0)
        end_location = bc.get('end_location', 0.0)
        mapping_type = bc.get('mapping_type', 'symmetry')
        symmetry_axis = bc.get('symmetry_axis', 'x')
        
        # Handle directional face selections
        if face_selection == '+x':
            # +X face (right): check if element is at right boundary (i==Nx-1)
            # and if Y coordinate is within [start_location, end_location]
            if (i == Nx-1) and (start_location <= y <= end_location):
                apply_mapping_logic(A, global_idx, start_idx, i, j, k, Nx, Ny, 
                                   mapping_type, symmetry_axis, '+x', dx_m)
                
        elif face_selection == '-x':
            # -X face (left): check if element is at left boundary (i==0)
            # and if Y coordinate is within [start_location, end_location]  
            if (i == 0) and (start_location <= y <= end_location):
                apply_mapping_logic(A, global_idx, start_idx, i, j, k, Nx, Ny, 
                                   mapping_type, symmetry_axis, '-x', dx_m)
                
        elif face_selection == '+y':
            # +Y face (top): check if element is at top boundary (j==Ny-1)
            # and if X coordinate is within [start_location, end_location]
            if (j == Ny-1) and (start_location <= x <= end_location):
                apply_mapping_logic(A, global_idx, start_idx, i, j, k, Nx, Ny, 
                                   mapping_type, symmetry_axis, '+y', dy_m)
                
        elif face_selection == '-y':
            # -Y face (bottom): check if element is at bottom boundary (j==0)
            # and if X coordinate is within [start_location, end_location]
            if (j == 0) and (start_location <= x <= end_location):
                apply_mapping_logic(A, global_idx, start_idx, i, j, k, Nx, Ny, 
                                   mapping_type, symmetry_axis, '-y', dy_m)

def apply_mapping_logic(A, global_idx, start_idx, i, j, k, Nx, Ny, mapping_type, symmetry_axis, face_direction, thermal_conductance):
    """
    Apply symmetry mapping logic based on symmetry axis.
    
    Args:
        A: Global system matrix
        global_idx: Current element global index
        start_idx: Starting index for current region
        i, j, k: Element indices
        Nx, Ny: Grid dimensions
        mapping_type: Type of mapping (always 'symmetry')
        symmetry_axis: Axis for symmetry ('x' or 'y')
        face_direction: Face direction ('+x', '-x', '+y', '-y')
        thermal_conductance: Thermal conductance (1/dx_m**2 or 1/dy_m**2)
    """
    
    # Symmetry mapping: mirror temperature distribution
    A[global_idx, global_idx] -= thermal_conductance
    
    if face_direction == '+x':
        # +X face (right): mirror based on symmetry axis
        if symmetry_axis == 'y':
            # Mirror across X axis: connect to opposite X face (left face)
            opposite_i = 0  # Left face
            opposite_global_idx = start_idx + k * Nx * Ny + j * Nx + opposite_i
        elif symmetry_axis == 'x':
            # Mirror across Y axis: connect to opposite Y position on same face
            opposite_j = Ny - 1 - j
            opposite_global_idx = start_idx + k * Nx * Ny + opposite_j * Nx + i
        else:
            return  # Invalid symmetry axis
            
    elif face_direction == '-x':
        # -X face (left): mirror based on symmetry axis
        if symmetry_axis == 'y':
            # Mirror across X axis: connect to opposite X face (right face)
            opposite_i = Nx - 1  # Right face
            opposite_global_idx = start_idx + k * Nx * Ny + j * Nx + opposite_i
        elif symmetry_axis == 'x':
            # Mirror across Y axis: connect to opposite Y position on same face
            opposite_j = Ny - 1 - j
            opposite_global_idx = start_idx + k * Nx * Ny + opposite_j * Nx + i
        else:
            return  # Invalid symmetry axis
            
    elif face_direction == '+y':
        # +Y face (top): mirror based on symmetry axis
        if symmetry_axis == 'x':
            # Mirror across Y axis: connect to opposite Y face (bottom face)
            opposite_j = 0  # Bottom face
            opposite_global_idx = start_idx + k * Nx * Ny + opposite_j * Nx + i
        elif symmetry_axis == 'y':
            # Mirror across X axis: connect to opposite X position on same face
            opposite_i = Nx - 1 - i
            opposite_global_idx = start_idx + k * Nx * Ny + j * Nx + opposite_i
        else:
            return  # Invalid symmetry axis
            
    elif face_direction == '-y':
        # -Y face (bottom): mirror based on symmetry axis
        if symmetry_axis == 'x':
            # Mirror across Y axis: connect to opposite Y face (top face)
            opposite_j = Ny - 1  # Top face
            opposite_global_idx = start_idx + k * Nx * Ny + opposite_j * Nx + i
        elif symmetry_axis == 'y':
            # Mirror across X axis: connect to opposite X position on same face
            opposite_i = Nx - 1 - i
            opposite_global_idx = start_idx + k * Nx * Ny + j * Nx + opposite_i
        else:
            return  # Invalid symmetry axis
    else:
        return  # Unsupported face direction
        
    A[global_idx, opposite_global_idx] = thermal_conductance

def update_matrix_with_geometries(A, region_info: Dict, actuator_info: Dict) -> scipy.sparse.lil_matrix:
    """
    Update matrix A based on element positions using pre-calculated indices.
    
    Args:
        A: Global system matrix
        region_info: Dictionary with region information
        actuator_info: Dictionary with actuator information
        
    Returns:
        Updated global system matrix
    """
    print("\n" + "=" * 50)
    print("\nUpdating matrix with geometry data...")
    
    # Process each region
    for region_id, info in region_info.items():
        print(f"\nProcessing region: {region_id}")
        coords = info['coords']
        thermal_conductivity = info['region_data']['thermal_conductivity']
        region_data = info['region_data']
        start_idx = info['start_idx']
        
        # Region dimensions
        Nx, Ny, Nz = coords.Nx, coords.Ny, coords.Nz
        dx, dy, dz = coords.dx, coords.dy, coords.dz
        dx_m, dy_m, dz_m = dx * 1e-3, dy * 1e-3, dz * 1e-3
        
        # Get MAPPED boundary conditions for this region
        mapped_bcs = []
        for bc in region_data.get("boundary_conditions", []):
            if bc.get("type") == "MAPPED":
                mapped_bcs.append(bc)
        
        if mapped_bcs:
            print(f"Found {len(mapped_bcs)} MAPPED boundary condition(s) in region {region_id}")
        
        # Get pre-calculated indices
        element_indices = info['element_indices']
        
        # Process corner elements
        for global_idx, i, j, k in element_indices['corner']:
            # Reset row
            A[global_idx, :] = 0
            
            # Corner element - 3 surfaces exposed
            A[global_idx, global_idx] = -1/dx_m**2 - 1/dy_m**2 - 1/dz_m**2
            
            # Add neighboring elements based on position
            if i == 0:
                A[global_idx, global_idx + 1] = 1/dx_m**2
            else:
                A[global_idx, global_idx - 1] = 1/dx_m**2
                
            if j == 0:
                A[global_idx, global_idx + Nx] = 1/dy_m**2
            else:
                A[global_idx, global_idx - Nx] = 1/dy_m**2
                
            if k == 0:
                A[global_idx, global_idx + Nx*Ny] = 1/dz_m**2
            else:
                A[global_idx, global_idx - Nx*Ny] = 1/dz_m**2
            
            # Apply MAPPED boundary conditions
            _apply_mapped_boundary_condition(A, global_idx, i, j, k, coords, mapped_bcs, start_idx, Nx, Ny, dx_m, dy_m)
            # Apply thermal conductivity modifications
            _apply_thermal_conductivity_modifications(A, global_idx, i, j, k, info, thermal_conductivity, Nx, Ny, Nz, dx_m, dy_m, dz_m, element_type='corner')
            # Scale by thermal conductivity
            A[global_idx, :] *= thermal_conductivity 
            
        
        # Process edge elements
        for global_idx, i, j, k in element_indices['edge']:
            # Reset row
            A[global_idx, :] = 0
            
            # Edge element - 2 surfaces exposed
            A[global_idx, global_idx] = -1/dx_m**2 - 1/dy_m**2 - 1/dz_m**2
            
            # Add neighboring elements based on edge type
            if i in [0, Nx-1] and j in [0, Ny-1]:  # Edge parallel to z-axis
                A[global_idx, global_idx] -= 1/dz_m**2
                
                if i == 0:
                    A[global_idx, global_idx + 1] = 1/dx_m**2
                else:
                    A[global_idx, global_idx - 1] = 1/dx_m**2
                    
                if j == 0:
                    A[global_idx, global_idx + Nx] = 1/dy_m**2
                else:
                    A[global_idx, global_idx - Nx] = 1/dy_m**2
                    
                A[global_idx, global_idx + Nx*Ny] = 1/dz_m**2
                A[global_idx, global_idx - Nx*Ny] = 1/dz_m**2
                
            elif i in [0, Nx-1] and k in [0, Nz-1]:  # Edge parallel to y-axis
                A[global_idx, global_idx] -= 1/dy_m**2
                
                if i == 0:
                    A[global_idx, global_idx + 1] = 1/dx_m**2
                else:
                    A[global_idx, global_idx - 1] = 1/dx_m**2
                    
                if k == 0:
                    A[global_idx, global_idx + Nx*Ny] = 1/dz_m**2
                else:
                    A[global_idx, global_idx - Nx*Ny] = 1/dz_m**2
                    
                A[global_idx, global_idx + Nx] = 1/dy_m**2
                A[global_idx, global_idx - Nx] = 1/dy_m**2
                
            elif j in [0, Ny-1] and k in [0, Nz-1]:  # Edge parallel to x-axis
                A[global_idx, global_idx] -= 1/dx_m**2
                
                if j == 0:
                    A[global_idx, global_idx + Nx] = 1/dy_m**2
                else:
                    A[global_idx, global_idx - Nx] = 1/dy_m**2
                    
                if k == 0:
                    A[global_idx, global_idx + Nx*Ny] = 1/dz_m**2
                else:
                    A[global_idx, global_idx - Nx*Ny] = 1/dz_m**2
                    
                A[global_idx, global_idx + 1] = 1/dx_m**2
                A[global_idx, global_idx - 1] = 1/dx_m**2
            
            # Apply MAPPED boundary conditions
            _apply_mapped_boundary_condition(A, global_idx, i, j, k, coords, mapped_bcs, start_idx, Nx, Ny, dx_m, dy_m)
            # Apply thermal conductivity modifications
            _apply_thermal_conductivity_modifications(A, global_idx, i, j, k, info, thermal_conductivity, Nx, Ny, Nz, dx_m, dy_m, dz_m, element_type='edge')
            
            # Scale by thermal conductivity
            A[global_idx, :] *= thermal_conductivity
        
        # Process surface elements
        for global_idx, i, j, k in element_indices['surface']:
            # Reset row
            A[global_idx, :] = 0
            
            # Surface element - 1 surface exposed
            A[global_idx, global_idx] = -1/dx_m**2 - 1/dy_m**2 - 1/dz_m**2
            
            # Add neighboring elements based on surface type
            if i in [0, Nx-1]:  # Surface parallel to y-z plane
                A[global_idx, global_idx] -= 1/dy_m**2 + 1/dz_m**2
                
                A[global_idx, global_idx + Nx] = 1/dy_m**2
                A[global_idx, global_idx - Nx] = 1/dy_m**2
                A[global_idx, global_idx + Nx*Ny] = 1/dz_m**2
                A[global_idx, global_idx - Nx*Ny] = 1/dz_m**2
                if i == 0:
                    A[global_idx, global_idx + 1] = 1/dx_m**2
                else:
                    A[global_idx, global_idx - 1] = 1/dx_m**2
                    
            elif j in [0, Ny-1]:  # Surface parallel to x-z plane
                A[global_idx, global_idx] -= 1/dx_m**2 + 1/dz_m**2
                
                A[global_idx, global_idx + 1] = 1/dx_m**2
                A[global_idx, global_idx - 1] = 1/dx_m**2
                A[global_idx, global_idx + Nx*Ny] = 1/dz_m**2
                A[global_idx, global_idx - Nx*Ny] = 1/dz_m**2
                if j == 0:
                    A[global_idx, global_idx + Nx] = 1/dy_m**2
                else:
                    A[global_idx, global_idx - Nx] = 1/dy_m**2
                    
            elif k in [0, Nz-1]:  # Surface parallel to x-y plane
                A[global_idx, global_idx] -= 1/dx_m**2 + 1/dy_m**2
                
                A[global_idx, global_idx + 1] = 1/dx_m**2
                A[global_idx, global_idx - 1] = 1/dx_m**2
                A[global_idx, global_idx + Nx] = 1/dy_m**2
                A[global_idx, global_idx - Nx] = 1/dy_m**2
                if k == 0:
                    A[global_idx, global_idx + Nx*Ny] = 1/dz_m**2
                else:
                    A[global_idx, global_idx - Nx*Ny] = 1/dz_m**2
            # Apply MAPPED boundary conditions
            _apply_mapped_boundary_condition(A, global_idx, i, j, k, coords, mapped_bcs, start_idx, Nx, Ny, dx_m, dy_m)
            # Apply thermal conductivity modifications
            _apply_thermal_conductivity_modifications(A, global_idx, i, j, k, info, thermal_conductivity, Nx, Ny, Nz, dx_m, dy_m, dz_m, element_type='surface')
            
            # Scale by thermal conductivity
            A[global_idx, :] *= thermal_conductivity

    return A 


def find_adiabatic_element_pairs(coords, bc_data, total_elements):
    """
    Find pairs of neighboring elements that are separated by an adiabatic line
    parallel to the z axis (i.e., a line segment in the x-y plane extruded along z).

    We detect separation by checking if the segment connecting two neighboring
    element centroids intersects the adiabatic line segment defined by
    (point1.x, point1.y) -> (point2.x, point2.y). A robust segment–segment
    intersection test is used (supports collinear overlap and endpoint touches).

    Args:
        coords: ElementCoordinates
        bc_data: Boundary condition data (expects point1/point2; supports legacy source_1)
        total_elements: Starting offset for this region's global indices

    Returns:
        List[dict]: each with element1, element2, and bc_data
    """
    from math import isclose

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1])

    def add(a, b):
        return (a[0] + b[0], a[1] + b[1])

    def mul(v, s):
        return (v[0] * s, v[1] * s)

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def _dist2(a, b):
        dx, dy = a[0] - b[0], a[1] - b[1]
        return dx * dx + dy * dy

    def _point_on_segment(p, a, b, eps):
        ab, ap = sub(b, a), sub(p, a)
        if not isclose(cross(ab, ap), 0.0, abs_tol=eps):
            return False
        return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
                min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)

    def segment_intersection(p1, p2, q1, q2, eps=1e-9):
        r = sub(p2, p1)
        s = sub(q2, q1)
        rxs = cross(r, s)
        qp = sub(q1, p1)

        if isclose(rxs, 0.0, abs_tol=eps):
            if not isclose(cross(qp, r), 0.0, abs_tol=eps):
                return ("none", None)
            rr = dot(r, r)
            if isclose(rr, 0.0, abs_tol=eps):
                return ("point", p1) if _point_on_segment(p1, q1, q2, eps) else ("none", None)
            t0 = dot(sub(q1, p1), r) / rr
            t1 = dot(sub(q2, p1), r) / rr
            a, b = max(0.0, min(t0, t1)), min(1.0, max(t0, t1))
            if a > b + eps:
                return ("none", None)
            A, B = add(p1, mul(r, a)), add(p1, mul(r, b))
            if _dist2(A, B) <= eps * eps:
                return ("point", A)
            return ("overlap", (A, B))
        else:
            t = cross(qp, s) / rxs
            u = cross(qp, r) / rxs
            if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
                I = add(p1, mul(r, t))
                return ("point", I)
            return ("none", None)

    def segments_cross(p1, p2, q1, q2, eps=1e-9):
        kind, _ = segment_intersection(p1, p2, q1, q2, eps)
        return kind != "none"

    # Extract line endpoints (support legacy rectangle -> convert to vertical segment)
    # Also capture optional z-level if provided and consistent between endpoints
    k_target = None  # if set, only build pairs for this k level
    if 'point1' in bc_data and 'point2' in bc_data:
        p1d = bc_data['point1']
        p2d = bc_data['point2']
        q1 = (float(p1d.get('x', 0.0)), float(p1d.get('y', 0.0)))
        q2 = (float(p2d.get('x', 0.0)), float(p2d.get('y', 0.0)))
        # Optional z-level support: if both provided and approximately equal,
        # restrict to the nearest k slice.
        if 'z' in p1d and 'z' in p2d:
            z1 = float(p1d['z'])
            z2 = float(p2d['z'])
            if isclose(z1, z2, abs_tol=1e-9):
                # Find nearest k to this z
                best_k = 0
                best_err = float('inf')
                for kk in range(coords.Nz):
                    _, _, zk = coords.get_coordinates_from_3d(0, 0, kk)
                    err = abs(zk - z1)
                    if err < best_err:
                        best_err = err
                        best_k = kk
                k_target = best_k
            else:
                # If z levels differ, conservatively return no pairs
                return []
    elif 'source_1' in bc_data:
        src = bc_data['source_1']
        x = float(src['centroid']['x'])
        y0 = float(src['centroid']['y']) - float(src['height']) / 2.0
        y1 = float(src['centroid']['y']) + float(src['height']) / 2.0
        q1, q2 = (x, y0), (x, y1)
    else:
        return []

    adiabatic_pairs = []

    # Check both x-neighbors (i,i+1) and y-neighbors (j,j+1) for all k
    for j in range(coords.Ny):
        for i in range(coords.Nx - 1):
            for k in range(coords.Nz):
                if k_target is not None and k != k_target:
                    continue
                x1, y1, _ = coords.get_coordinates_from_3d(i, j, k)
                x2, y2, _ = coords.get_coordinates_from_3d(i + 1, j, k)
                if segments_cross((x1, y1), (x2, y2), q1, q2):
                    idx1 = coords.get_global_index(coords.Nx, coords.Ny, i, j, k)
                    idx2 = coords.get_global_index(coords.Nx, coords.Ny, i + 1, j, k)
                    adiabatic_pairs.append({
                        'element1': {'global_idx': total_elements + idx1, 'i': i, 'j': j, 'k': k, 'x': x1, 'y': y1},
                        'element2': {'global_idx': total_elements + idx2, 'i': i + 1, 'j': j, 'k': k, 'x': x2, 'y': y2},
                        'bc_data': bc_data
                    })

    for i in range(coords.Nx):
        for j in range(coords.Ny - 1):
            for k in range(coords.Nz):
                if k_target is not None and k != k_target:
                    continue
                x1, y1, _ = coords.get_coordinates_from_3d(i, j, k)
                x2, y2, _ = coords.get_coordinates_from_3d(i, j + 1, k)
                if segments_cross((x1, y1), (x2, y2), q1, q2):
                    idx1 = coords.get_global_index(coords.Nx, coords.Ny, i, j, k)
                    idx2 = coords.get_global_index(coords.Nx, coords.Ny, i, j + 1, k)
                    adiabatic_pairs.append({
                        'element1': {'global_idx': total_elements + idx1, 'i': i, 'j': j, 'k': k, 'x': x1, 'y': y1},
                        'element2': {'global_idx': total_elements + idx2, 'i': i, 'j': j + 1, 'k': k, 'x': x2, 'y': y2},
                        'bc_data': bc_data
                    })

    return adiabatic_pairs

def preconditioner(A):
    """
    Create a simple diagonal preconditioner for the matrix A.
    
    Parameters:
        A: Sparse matrix to precondition
        
    Returns:
        LinearOperator: Preconditioner operator
    """
    M_inv = diags(1./A.diagonal())
    return LinearOperator(matvec=lambda x: M_inv @ x, shape=A.shape, dtype=A.dtype)

class LayerMapping:
    def __init__(self, params: ThermalParameters):
        """
        Initialize mappings between regions.
        
        Args:
            params: ThermalParameters object containing mapping information
        """
        self.params = params
        
        # Initialize mappings for each region
        self.region_mappings = {}  # Dictionary of dictionaries for each region
        
        # Create mappings
        self._create_mappings()
    
    def _create_mappings(self):
        """Create element-wise mappings between regions."""
        # Get all mappings
        tolerance = 1e-6
        for mapping_id, mapping in self.params.mappings_by_id.items():
            # Get source and target region information
            source = mapping["source"]
            target = mapping["target"]
            
            # Get region dimensions
            source_dims = source["dimensions"]
            target_dims = target["dimensions"]
            
            # Get region centroid
            source_centroid = source["centroid"]
            target_centroid = target["centroid"]
            
            # Calculate mapped region bounds using absolute coordinates
            source_region = {
                'x_min': source_centroid["x"] - source_dims["Lx"]/2,
                'x_max': source_centroid["x"] + source_dims["Lx"]/2,
                'y_min': source_centroid["y"] - source_dims["Ly"]/2,
                'y_max': source_centroid["y"] + source_dims["Ly"]/2,
                'z_min': source_centroid["z"] - source_dims["Lz"]/2,
                'z_max': source_centroid["z"] + source_dims["Lz"]/2
            }
            
            target_region = {
                'x_min': target_centroid["x"] - target_dims["Lx"]/2,
                'x_max': target_centroid["x"] + target_dims["Lx"]/2,
                'y_min': target_centroid["y"] - target_dims["Ly"]/2,
                'y_max': target_centroid["y"] + target_dims["Ly"]/2,
                'z_min': target_centroid["z"] - target_dims["Lz"]/2,
                'z_max': target_centroid["z"] + target_dims["Lz"]/2
            }
            
            # Initialize lists for this mapping
            source_coords = ElementCoordinates(self.params, source["region_id"])
            target_coords = ElementCoordinates(self.params, target["region_id"])
            
            # Create unique keys for source and target mappings
            source_key = f"{source['region_id']}_source_{mapping_id}"
            target_key = f"{target['region_id']}_target_{mapping_id}"
            
            # Initialize mappings for these regions if not already done
            if source_key not in self.region_mappings:
                self.region_mappings[source_key] = []
            if target_key not in self.region_mappings:
                self.region_mappings[target_key] = []
            
            def find_containing_elements(coords, region, region_dims):
                # Get mesh element sizes
                dx = coords.dx
                dy = coords.dy
                dz = coords.dz
                
                # Check if mapped region is smaller than element size
                if (region_dims["Lx"] < dx and 
                    region_dims["Ly"] < dy and 
                    region_dims["Lz"] < dz):
                    # Find the single element that contains the mapped region
                    i = int((region['x_min'] - coords.x_ini) / dx)
                    j = int((region['y_min'] - coords.y_ini) / dy)
                    k = int((region['z_min'] - coords.z_ini) / dz)
                    return [(i, i, j, j, k, k)]
                else:
                    # Find the range of mesh elements that contain or intersect with the mapped region
                    i_min = max(0, int((region['x_min'] - coords.x_ini) / dx))
                    i_max = min(coords.Nx - 1, int((region['x_max'] - coords.x_ini) / dx))
                    j_min = max(0, int((region['y_min'] - coords.y_ini) / dy))
                    j_max = min(coords.Ny - 1, int((region['y_max'] - coords.y_ini) / dy))
                    k_min = max(0, int((region['z_min'] - coords.z_ini) / dz))
                    k_max = min(coords.Nz - 1, int((region['z_max'] - coords.z_ini) / dz))
                    return [(i_min, i_max, j_min, j_max, k_min, k_max)]
            
            # Find containing elements for source region
            source_ranges = find_containing_elements(source_coords, source_region, source_dims)
            
            # Find containing elements for target region
            target_ranges = find_containing_elements(target_coords, target_region, target_dims)
            
            # Synchronize ranges to the smaller span
            if source_ranges and target_ranges:
                s_i_min, s_i_max, s_j_min, s_j_max, s_k_min, s_k_max = source_ranges[0]
                t_i_min, t_i_max, t_j_min, t_j_max, t_k_min, t_k_max = target_ranges[0]
                
                source_di = s_i_max - s_i_min
                source_dj = s_j_max - s_j_min
                source_dk = s_k_max - s_k_min
                
                target_di = t_i_max - t_i_min
                target_dj = t_j_max - t_j_min
                target_dk = t_k_max - t_k_min
                
                min_di = min(source_di, target_di)
                min_dj = min(source_dj, target_dj)
                min_dk = min(source_dk, target_dk)
                
                # Adjust source range
                new_s_i_max = s_i_min + min_di
                new_s_j_max = s_j_min + min_dj
                new_s_k_max = s_k_min + min_dk
                source_ranges = [(s_i_min, new_s_i_max, s_j_min, new_s_j_max, s_k_min, new_s_k_max)]
                
                # Adjust target range
                new_t_i_max = t_i_min + min_di
                new_t_j_max = t_j_min + min_dj
                new_t_k_max = t_k_min + min_dk
                target_ranges = [(t_i_min, new_t_i_max, t_j_min, new_t_j_max, t_k_min, new_t_k_max)]
            
            else:
                # If either range is empty, skip this mapping
                print(f"Warning: Could not find elements for mapping {mapping_id}. Skipping.")
                continue
            
            # Store source indices
            source_indices = []
            
            # Find points in source region that belong to mapped region
            if source_ranges[0][0] == source_ranges[0][1]:
                self.region_mappings[source_key].append(source_coords.get_global_index(source_coords.Nx, source_coords.Ny, source_ranges[0][0], source_ranges[0][2], source_ranges[0][4]))
            else:
                for i_min, i_max, j_min, j_max, k_min, k_max in source_ranges:
                    for k in range(k_min, k_max + 1):
                        for j in range(j_min, j_max + 1):
                            for i in range(i_min, i_max + 1):
                                # Get coordinates directly from 3D indices
                                x, y, z = source_coords.get_coordinates_from_3d(i, j, k)
                                if (source_region['x_min'] <= x <= source_region['x_max'] and
                                    source_region['y_min'] <= y <= source_region['y_max'] and
                                    source_region['z_min'] - tolerance <= z <= source_region['z_max'] + tolerance):
                                    # Calculate global index from 3D indices
                                    source_idx = source_coords.get_global_index(source_coords.Nx, source_coords.Ny, i, j, k)
                                    self.region_mappings[source_key].append(source_idx)
            
            # Find points in target region that belong to mapped region
            if target_ranges[0][0] == target_ranges[0][1]:
                self.region_mappings[target_key].append(target_coords.get_global_index(target_coords.Nx, target_coords.Ny, target_ranges[0][0], target_ranges[0][2], target_ranges[0][4]))
            else:
                for i_min, i_max, j_min, j_max, k_min, k_max in target_ranges:
                    for k in range(k_min, k_max + 1):
                        for j in range(j_min, j_max + 1):
                            for i in range(i_min, i_max + 1):
                                # Get coordinates directly from 3D indices
                                x, y, z = target_coords.get_coordinates_from_3d(i, j, k)
                                if (target_region['x_min'] <= x <= target_region['x_max'] and
                                    target_region['y_min'] <= y <= target_region['y_max'] and
                                    target_region['z_min'] - tolerance <= z <= target_region['z_max'] + tolerance):
                                    # Calculate global index from 3D indices
                                    target_idx = target_coords.get_global_index(target_coords.Nx, target_coords.Ny, i, j, k)
                                    self.region_mappings[target_key].append(target_idx)
            
            # Verify that we have the same number of mapped points
            if len(self.region_mappings[source_key]) != len(self.region_mappings[target_key]):
                raise ValueError(f"Number of mapped points in regions {source['region_id']} and {target['region_id']} do not match for mapping {mapping_id}")
        print("mapping created")
    
    def get_corresponding_elements(self, region_id: str, idx: int) -> Dict[str, List[int]]:
        """
        Get corresponding points in other regions.
        
        Args:
            region_id: ID of the source region
            idx: Global index in the source region
            
        Returns:
            Dictionary mapping region IDs to lists of corresponding global indices
        """
        correspondences = {}
        for mapping_id, mapping in self.params.mappings_by_id.items():
            if region_id == mapping["source"]["region_id"]:
                if idx in self.region_mappings[f"{region_id}_source_{mapping_id}"]:
                    pos = self.region_mappings[f"{region_id}_source_{mapping_id}"].index(idx)
                    target_id = mapping["target"]["region_id"]
                    if target_id not in correspondences:
                        correspondences[target_id] = []
                    correspondences[target_id].append(self.region_mappings[f"{target_id}_target_{mapping_id}"][pos])
            elif region_id == mapping["target"]["region_id"]:
                if idx in self.region_mappings[f"{region_id}_target_{mapping_id}"]:
                    pos = self.region_mappings[f"{region_id}_target_{mapping_id}"].index(idx)
                    source_id = mapping["source"]["region_id"]
                    if source_id not in correspondences:
                        correspondences[source_id] = []
                    correspondences[source_id].append(self.region_mappings[f"{source_id}_source_{mapping_id}"][pos])
        return correspondences

def calculate_system_size(regions: List[dict], actuators: List[dict], params: ThermalParameters) -> Tuple[Dict, Dict, int, int]:
    """
    Calculate system size and pre-calculate indices for boundary conditions.
    
    Args:
        regions: List of regions from geometry file
        actuators: List of actuators from geometry file
        params: ThermalParameters object
        
    Returns:
        Tuple containing:
        - Dictionary with region information
        - Dictionary with actuator information
        - Total number of elements
        - Total number of unknowns
    """
    print("\nCalculating system size...")
    
    # Initialize dictionaries
    region_info = {}
    actuator_info = {}
    total_elements = 0
    total_unknowns = 0
    
    # Process each region
    for region in regions:
        region_id = region["id"]
        print(f"\nProcessing region: {region_id}")
        
        # Create coordinate system for this region
        coords = ElementCoordinates(params, region_id)
        
        # Pre-calculate geometric element indices
        corner_indices = []
        edge_indices = []
        surface_indices = []
        inner_indices = []
        
        # Calculate indices for each geometric element type
        for i in range(coords.Nx):
            for j in range(coords.Ny):
                for k in range(coords.Nz):
                    local_idx = coords.get_global_index(coords.Nx, coords.Ny, i, j, k)
                    global_idx = total_elements + local_idx
                    
                    # Determine element position
                    if (i in [0, coords.Nx-1] and j in [0, coords.Ny-1] and k in [0, coords.Nz-1]):
                        corner_indices.append((global_idx, i, j, k))
                    elif ((i in [0, coords.Nx-1] and j in [0, coords.Ny-1]) or 
                          (i in [0, coords.Nx-1] and k in [0, coords.Nz-1]) or 
                          (j in [0, coords.Ny-1] and k in [0, coords.Nz-1])):
                        edge_indices.append((global_idx, i, j, k))
                    elif (i in [0, coords.Nx-1] or j in [0, coords.Ny-1] or k in [0, coords.Nz-1]):
                        surface_indices.append((global_idx, i, j, k))
                    else:
                        inner_indices.append((global_idx, i, j, k))
        
        # Initialize boundary condition indices dictionary
        bc_indices = {}
        
        # Initialize adiabatic pairs list
        adiabatic_pairs = []
        
        # Process each boundary condition
        for bc in region.get("boundary_conditions", []):
            bc_type = bc.get("type", "UNKNOWN")
            if bc_type not in bc_indices:
                bc_indices[bc_type] = []
            
            # Get the boundary region dimensions
            bc_width = bc.get('width', 0.0)
            bc_height = bc.get('height', 0.0)
            bc_centroid = bc.get('centroid', {'x': 0, 'y': 0, 'z': 0})
            
            # Calculate the bounds of the boundary condition region
            x_min = bc_centroid['x'] - bc_width/2
            x_max = bc_centroid['x'] + bc_width/2
            y_min = bc_centroid['y'] - bc_height/2
            y_max = bc_centroid['y'] + bc_height/2

            # Add buffer zone for CONST_Q and ACTUATOR_CONNECTED boundary conditions
            if bc_type in ["PLASTIC_COVERED", "NODE_CONNECTED", "CONST_Q", "CONST_T", "ACTUATOR_CONNECTED", "USERDEF_CONVECTION"]:
                # Add dx/2, dy/2 buffer to the region bounds
                x_min -= coords.dx/2
                x_max += coords.dx/2
                y_min -= coords.dy/2
                y_max += coords.dy/2
            
            # Special handling for ADIABATIC boundary conditions
            if bc_type == "ADIABATIC":
                # Find element pairs that have the adiabatic line between them
                pairs = find_adiabatic_element_pairs(coords, bc, total_elements)
                adiabatic_pairs.extend(pairs)
            
            # Find all elements in this boundary region
            for i in range(coords.Nx):
                for j in range(coords.Ny):
                    # Get element coordinates
                    x, y, _ = coords.get_coordinates_from_3d(i, j, 0)
                    
                    # Check if element is within boundary region
                    if (x_min <= x <= x_max and y_min <= y <= y_max):
                        # For surface-type boundary conditions, only add the appropriate surface based on centroid_z
                        surface_bc_types = ["PLASTIC_COVERED", "NODE_CONNECTED", "CONST_Q", "CONST_T", "ACTUATOR_CONNECTED", "USERDEF_CONVECTION"]
                        
                        if bc_type in surface_bc_types:
                            # Determine which surface (top or bottom) based on centroid_z
                            bc_centroid_z = bc_centroid.get('z', 0)
                            region_centroid_z = region["centroid"]["z"]
                            
                            # If BC centroid_z is above region centroid_z, apply to top surface (k = Nz-1)
                            # If BC centroid_z is below region centroid_z, apply to bottom surface (k = 0)
                            if bc_centroid_z >= region_centroid_z:
                                k = coords.Nz - 1  # Top surface
                            else:
                                k = 0  # Bottom surface
                            
                            local_idx = coords.get_global_index(coords.Nx, coords.Ny, i, j, k)
                            global_idx = total_elements + local_idx
                            bc_indices[bc_type].append({
                                'global_idx': global_idx,
                                'i': i, 'j': j, 'k': k,
                                'bc_data': bc  # Store the full boundary condition data
                            })
                        else:
                            # For all other boundary conditions, add both top and bottom surface elements
                            for k in [0, coords.Nz-1]:
                                local_idx = coords.get_global_index(coords.Nx, coords.Ny, i, j, k)
                                global_idx = total_elements + local_idx
                                bc_indices[bc_type].append({
                                    'global_idx': global_idx,
                                    'i': i, 'j': j, 'k': k,
                                    'bc_data': bc  # Store the full boundary condition data
                                })
        
        # Store region information
        region_info[region_id] = {
            'coords': coords,
            'start_idx': total_elements,
            'num_elements': coords.Nx * coords.Ny * coords.Nz,
            'element_indices': {
                'corner': corner_indices,
                'edge': edge_indices,
                'surface': surface_indices,
                'inner': inner_indices
            },
            'boundary_indices': bc_indices,
            'adiabatic_pairs': adiabatic_pairs,  # Store adiabatic pairs
            'region_data': region
        }
        
        # Update total elements
        total_elements += coords.Nx * coords.Ny * coords.Nz
        
        # Print statistics
        print(f"\nRegion {region_id}: {coords.Nx * coords.Ny * coords.Nz} elements (Total: {total_elements})")
        print("Geometric elements:")
        print(f"  - Corner elements: {len(corner_indices)}")
        print(f"  - Edge elements: {len(edge_indices)}")
        print(f"  - Surface elements: {len(surface_indices)}")
        print(f"  - Inner elements: {len(inner_indices)}")
        print("Boundary condition elements:")
        for bc_type, elements in bc_indices.items():
            print(f"  - {bc_type}: {len(elements)} elements")
        if adiabatic_pairs:
            print(f"  - Adiabatic pairs: {len(adiabatic_pairs)} pairs")
    
    # Process each actuator
    for actuator in actuators:
        actuator_id = actuator["id"]
        print(f"\nProcessing actuator: {actuator_id}")
        
        # Store actuator information
        num_unknowns = 3  # Each actuator has 2 temperature unknowns
        actuator_info[actuator_id] = {
            'start_idx': total_elements,
            'num_unknowns': num_unknowns,
            'heat_losses': actuator["heat_losses"],
            'thermal_resistance': actuator["thermal_resistance"]
        }
        
        # Update total unknowns
        total_unknowns += num_unknowns
        print(f"Actuator {actuator_id}: {num_unknowns} unknowns (Total: {total_elements + total_unknowns})")
    
    # Update total unknowns to include elements
    total_unknowns += total_elements
    
    print(f"\nTotal elements: {total_elements}")
    print(f"Total unknowns: {total_unknowns}")
    
    return region_info, actuator_info, total_elements, total_unknowns