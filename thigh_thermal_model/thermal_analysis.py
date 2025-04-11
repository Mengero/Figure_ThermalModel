import numpy as np
from scipy.sparse import diags, block_diag, kron, identity
from scipy.sparse.linalg import LinearOperator
from typing import Dict, Tuple, List, Optional
from thermal_parameters import ThermalParameters
from calculate_elements import calculate_region_elements

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

def create_combined_matrix(metal_coords, plastic_coords, params: ThermalParameters):
    """
    Create the combined matrix for all layers in the thermal model using NumPy arrays.
    
    Parameters:
        params: ThermalParameters object containing model parameters
        
    Returns:
        numpy.ndarray: Combined matrix for all layers
    """
    # Get mesh dimensions for each layer
    metal_nx, metal_ny, metal_nz = metal_coords.Nx, metal_coords.Ny, metal_coords.Nz
    metal_dx, metal_dy, metal_dz = metal_coords.dx, metal_coords.dy, metal_coords.dz
    metal_dx_m = metal_dx*1e-3
    metal_dy_m = metal_dy*1e-3
    metal_dz_m = metal_dz*1e-3
    k_metal = params.get_region_thermal_conductivity("metal_layer")
    
    # Create matrix for metal layer and convert to dense array
    A_metal = create_layer_matrix(metal_nx, metal_ny, metal_nz, metal_dx_m, metal_dy_m, metal_dz_m, k_metal)
    A_metal = A_metal.toarray()  # Convert to dense NumPy array
    
    # Get dimensions for plastic layer
    plastic_nx, plastic_ny, plastic_nz = plastic_coords.Nx, plastic_coords.Ny, plastic_coords.Nz
    plastic_dx, plastic_dy, plastic_dz = plastic_coords.dx, plastic_coords.dy, plastic_coords.dz
    plastic_dx_m = plastic_dx*1e-3
    plastic_dy_m = plastic_dy*1e-3
    plastic_dz_m = plastic_dz*1e-3
    k_plastic = params.get_region_thermal_conductivity("plastic_layer")
    
    # Create matrix for plastic layer and convert to dense array
    A_plastic = create_layer_matrix(plastic_nx, plastic_ny, plastic_nz, plastic_dx_m, plastic_dy_m, plastic_dz_m, k_plastic)
    A_plastic = A_plastic.toarray()  # Convert to dense NumPy array
    
    # Get total dimensions
    total_metal = metal_nx * metal_ny * metal_nz
    total_plastic = plastic_nx * plastic_ny * plastic_nz
    total_size = total_metal + total_plastic
    
    # Create combined matrix
    A_combined = np.zeros((total_size, total_size))
    
    # Fill in metal layer
    A_combined[:total_metal, :total_metal] = A_metal
    
    # Fill in plastic layer
    A_combined[total_metal:, total_metal:] = A_plastic
    
    return A_combined

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

class ElementCoordinates:
    """Class for managing element coordinates in a layer."""
    
    def __init__(self, params: ThermalParameters, region_id: str):
        """
        Initialize element coordinates for a region.
        
        Args:
            params: ThermalParameters object
            region_id: ID of the region
        """
        self.region_id = region_id
        self.params = params
        
        # Get region dimensions and position
        self.Lx, self.Ly, self.Lz = params.get_region_dimensions(region_id)
        self.x0, self.y0, self.z0 = params.get_region_position(region_id)
        
        # Calculate number of points and spacing
        _, self.Nx, self.Ny, self.Nz, self.dx, self.dy, self.dz, _ = calculate_region_elements(params, region_id)
        
        # Pre-calculate all point coordinates
        self.coordinates = []
        for k in range(self.Nz):
            for j in range(self.Ny):
                for i in range(self.Nx):
                    x = self.x0 + i * self.dx
                    y = self.y0 + j * self.dy
                    z = self.z0 + k * self.dz
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
        
        Args:
            i: x-direction index
            j: y-direction index
            k: z-direction index
            
        Returns:
            Tuple of (x,y,z) coordinates
        """
        x = self.x0 + i * self.dx
        y = self.y0 + j * self.dy
        z = self.z0 + k * self.dz
        return (x, y, z)

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
                    i = int((region['x_min'] - coords.x0) / dx)
                    j = int((region['y_min'] - coords.y0) / dy)
                    k = int((region['z_min'] - coords.z0) / dz)
                    return [(i, i, j, j, k, k)]
                else:
                    # Find the range of mesh elements that contain or intersect with the mapped region
                    i_min = max(0, int((region['x_min'] - coords.x0) / dx))
                    i_max = min(coords.Nx - 1, int((region['x_max'] - coords.x0) / dx))
                    j_min = max(0, int((region['y_min'] - coords.y0) / dy))
                    j_max = min(coords.Ny - 1, int((region['y_max'] - coords.y0) / dy))
                    k_min = max(0, int((region['z_min'] - coords.z0) / dz))
                    k_max = min(coords.Nz - 1, int((region['z_max'] - coords.z0) / dz))
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

def create_layer_coordinates(params: ThermalParameters) -> Tuple[ElementCoordinates, ElementCoordinates]:
    """
    Create coordinate systems for both layers.
    
    Args:
        params: ThermalParameters object
        
    Returns:
        Tuple of (metal_coords, plastic_coords)
    """
    # Get metal layer parameters
    metal_coords = ElementCoordinates(params, "metal_layer")
    
    # Get plastic layer parameters
    plastic_coords = ElementCoordinates(params, "plastic_layer")
    
    return metal_coords, plastic_coords

def update_matrix_with_geometries(A, metal_coords, plastic_coords, params: ThermalParameters):
    """Update matrix A based on element positions (corner, edge, surface) for both metal and plastic layers"""
    # Metal layer dimensions
    Nx_metal, Ny_metal, Nz_metal = metal_coords.Nx, metal_coords.Ny, metal_coords.Nz
    dx_metal, dy_metal, dz_metal = metal_coords.dx, metal_coords.dy, metal_coords.dz
    dx_metal_m, dy_metal_m, dz_metal_m = dx_metal * 1e-3, dy_metal * 1e-3, dz_metal * 1e-3
    
    # Plastic layer dimensions
    Nx_plastic, Ny_plastic, Nz_plastic = plastic_coords.Nx, plastic_coords.Ny, plastic_coords.Nz
    dx_plastic, dy_plastic, dz_plastic = plastic_coords.dx, plastic_coords.dy, plastic_coords.dz
    dx_plastic_m, dy_plastic_m, dz_plastic_m = dx_plastic * 1e-3, dy_plastic * 1e-3, dz_plastic * 1e-3
    
    # Total number of elements in each layer
    total_metal = Nx_metal * Ny_metal * Nz_metal
    total_plastic = Nx_plastic * Ny_plastic * Nz_plastic
    
    k_metal = params.get_region_thermal_conductivity("metal_layer")
    k_plastic = params.get_region_thermal_conductivity("plastic_layer")
    
    # Update metal layer
    for i in range(Nx_metal):
        for j in range(Ny_metal):
            for k in range(Nz_metal):
                # Skip inner elements
                if 0 < i < Nx_metal-1 and 0 < j < Ny_metal-1 and 0 < k < Nz_metal-1:
                    continue
                    
                idx = i + j*Nx_metal + k*Nx_metal*Ny_metal
                
                # Determine element position
                is_corner = (i in [0, Nx_metal-1] and j in [0, Ny_metal-1] and k in [0, Nz_metal-1])
                is_edge = ((i in [0, Nx_metal-1] and j in [0, Ny_metal-1]) or 
                          (i in [0, Nx_metal-1] and k in [0, Nz_metal-1]) or 
                          (j in [0, Ny_metal-1] and k in [0, Nz_metal-1]))
                is_surface = (i in [0, Nx_metal-1] or j in [0, Ny_metal-1] or k in [0, Nz_metal-1])
                
                # Reset row
                A[idx, :] = 0
                
                if is_corner:
                    # Corner element - 3 surfaces exposed
                    A[idx, idx] = -1/dx_metal_m**2 - 1/dy_metal_m**2 - 1/dz_metal_m**2
                    
                    # Add neighboring elements based on position
                    if i == 0:
                        A[idx, idx + 1] = 1/dx_metal_m**2
                    else:
                        A[idx, idx - 1] = 1/dx_metal_m**2
                        
                    if j == 0:
                        A[idx, idx + Nx_metal] = 1/dy_metal_m**2
                    else:
                        A[idx, idx - Nx_metal] = 1/dy_metal_m**2
                        
                    if k == 0:
                        A[idx, idx + Nx_metal*Ny_metal] = 1/dz_metal_m**2
                    else:
                        A[idx, idx - Nx_metal*Ny_metal] = 1/dz_metal_m**2
                        
                elif is_edge:
                    # Edge element - 2 surfaces exposed
                    A[idx, idx] = -1/dx_metal_m**2 - 1/dy_metal_m**2 - 1/dz_metal_m**2
                    
                    # Add neighboring elements based on edge type
                    if i in [0, Nx_metal-1] and j in [0, Ny_metal-1]:  # Edge parallel to z-axis
                        A[idx, idx] -= 1/dz_metal_m**2
                        
                        if i == 0:
                            A[idx, idx + 1] = 1/dx_metal_m**2
                        else:
                            A[idx, idx - 1] = 1/dx_metal_m**2
                            
                        if j == 0:
                            A[idx, idx + Nx_metal] = 1/dy_metal_m**2
                        else:
                            A[idx, idx - Nx_metal] = 1/dy_metal_m**2
                            
                        A[idx, idx + Nx_metal*Ny_metal] = 1/dz_metal_m**2
                        A[idx, idx - Nx_metal*Ny_metal] = 1/dz_metal_m**2
                        
                    elif i in [0, Nx_metal-1] and k in [0, Nz_metal-1]:  # Edge parallel to y-axis
                        A[idx, idx] -= 1/dy_metal_m**2
                        
                        if i == 0:
                            A[idx, idx + 1] = 1/dx_metal_m**2
                        else:
                            A[idx, idx - 1] = 1/dx_metal_m**2
                            
                        if k == 0:
                            A[idx, idx + Nx_metal*Ny_metal] = 1/dz_metal_m**2
                        else:
                            A[idx, idx - Nx_metal*Ny_metal] = 1/dz_metal_m**2
                            
                        A[idx, idx + Nx_metal] = 1/dy_metal_m**2
                        A[idx, idx - Nx_metal] = 1/dy_metal_m**2
                        
                    elif j in [0, Ny_metal-1] and k in [0, Nz_metal-1]:  # Edge parallel to x-axis
                        A[idx, idx] -= 1/dx_metal_m**2
                        
                        if j == 0:
                            A[idx, idx + Nx_metal] = 1/dy_metal_m**2
                        else:
                            A[idx, idx - Nx_metal] = 1/dy_metal_m**2
                            
                        if k == 0:
                            A[idx, idx + Nx_metal*Ny_metal] = 1/dz_metal_m**2
                        else:
                            A[idx, idx - Nx_metal*Ny_metal] = 1/dz_metal_m**2
                            
                        A[idx, idx + 1] = 1/dx_metal_m**2
                        A[idx, idx - 1] = 1/dx_metal_m**2 
                        
                            
                elif is_surface:
                    # Surface element - 1 surface exposed
                    A[idx, idx] = -1/dx_metal_m**2 - 1/dy_metal_m**2 - 1/dz_metal_m**2
                    
                    # Add neighboring elements based on surface type
                    if i in [0, Nx_metal-1]:  # Surface parallel to y-z plane
                        A[idx, idx] -= 1/dy_metal_m**2 + 1/dz_metal_m**2
                        
                        A[idx, idx + Nx_metal] = 1/dy_metal_m**2
                        A[idx, idx - Nx_metal] = 1/dy_metal_m**2
                        A[idx, idx + Nx_metal*Ny_metal] = 1/dz_metal_m**2
                        A[idx, idx - Nx_metal*Ny_metal] = 1/dz_metal_m**2
                        if i == 0:
                            A[idx, idx + 1] = 1/dx_metal_m**2
                        else:
                            A[idx, idx - 1] = 1/dx_metal_m**2
                            
                    elif j in [0, Ny_metal-1]:  # Surface parallel to x-z plane
                        A[idx, idx] -= 1/dx_metal_m**2 + 1/dz_metal_m**2
                        
                        A[idx, idx + 1] = 1/dx_metal_m**2
                        A[idx, idx - 1] = 1/dx_metal_m**2
                        A[idx, idx + Nx_metal*Ny_metal] = 1/dz_metal_m**2
                        A[idx, idx - Nx_metal*Ny_metal] = 1/dz_metal_m**2
                        if j == 0:
                            A[idx, idx + Nx_metal] = 1/dy_metal_m**2
                        else:
                            A[idx, idx - Nx_metal] = 1/dy_metal_m**2
                            
                    elif k in [0, Nz_metal-1]:  # Surface parallel to x-y plane
                        A[idx, idx] -= 1/dx_metal_m**2 + 1/dy_metal_m**2
                        
                        A[idx, idx + 1] = 1/dx_metal_m**2
                        A[idx, idx - 1] = 1/dx_metal_m**2
                        A[idx, idx + Nx_metal] = 1/dy_metal_m**2
                        A[idx, idx - Nx_metal] = 1/dy_metal_m**2
                        if k == 0:
                            A[idx, idx + Nx_metal*Ny_metal] = 1/dz_metal_m**2
                        else:
                            A[idx, idx - Nx_metal*Ny_metal] = 1/dz_metal_m**2
                            
                A[idx, :] *= k_metal

    # Update plastic layer (offset by total_metal)
    for i in range(Nx_plastic):
        for j in range(Ny_plastic):
            for k in range(Nz_plastic):
                # Skip inner elements
                if 0 < i < Nx_plastic-1 and 0 < j < Ny_plastic-1 and 0 < k < Nz_plastic-1:
                    continue
                    
                idx = total_metal + i + j*Nx_plastic + k*Nx_plastic*Ny_plastic
                
                # Determine element position
                is_corner = (i in [0, Nx_plastic-1] and j in [0, Ny_plastic-1] and k in [0, Nz_plastic-1])
                is_edge = ((i in [0, Nx_plastic-1] and j in [0, Ny_plastic-1]) or 
                          (i in [0, Nx_plastic-1] and k in [0, Nz_plastic-1]) or 
                          (j in [0, Ny_plastic-1] and k in [0, Nz_plastic-1]))
                is_surface = (i in [0, Nx_plastic-1] or j in [0, Ny_plastic-1] or k in [0, Nz_plastic-1])
                
                # Reset row
                A[idx, :] = 0
                A[idx, idx] = -1/dx_plastic_m**2 - 1/dy_plastic_m**2 - 1/dz_plastic_m**2
                if is_corner:
                    
                    # Add neighboring elements based on position
                    if i == 0:
                        A[idx, idx + 1] = 1/dx_plastic_m**2
                    else:
                        A[idx, idx - 1] = 1/dx_plastic_m**2
                        
                    if j == 0:
                        A[idx, idx + Nx_plastic] = 1/dy_plastic_m**2
                    else:
                        A[idx, idx - Nx_plastic] = 1/dy_plastic_m**2
                        
                    if k == 0:
                        A[idx, idx + Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                    else:
                        A[idx, idx - Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                        
                elif is_edge:
                    
                    # Add neighboring elements based on edge type
                    if i in [0, Nx_plastic-1] and j in [0, Ny_plastic-1]:  # Edge parallel to z-axis
                        A[idx, idx] -= 1/dz_plastic_m**2
                        
                        if i == 0:
                            A[idx, idx + 1] = 1/dx_plastic_m**2
                        else:
                            A[idx, idx - 1] = 1/dx_plastic_m**2
                            
                        if j == 0:
                            A[idx, idx + Nx_plastic] = 1/dy_plastic_m**2
                        else:
                            A[idx, idx - Nx_plastic] = 1/dy_plastic_m**2
                            
                        A[idx, idx + Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                        A[idx, idx - Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                        
                    elif i in [0, Nx_plastic-1] and k in [0, Nz_plastic-1]:  # Edge parallel to y-axis
                        A[idx, idx] -= 1/dy_plastic_m**2
                        
                        if i == 0:
                            A[idx, idx + 1] = 1/dx_plastic_m**2
                        else:
                            A[idx, idx - 1] = 1/dx_plastic_m**2
                            
                        A[idx, idx + Nx_plastic] = 1/dy_plastic_m**2
                        A[idx, idx - Nx_plastic] = 1/dy_plastic_m**2
                        
                        if k == 0:
                            A[idx, idx + Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                        else:
                            A[idx, idx - Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                            
                    elif j in [0, Ny_plastic-1] and k in [0, Nz_plastic-1]:  # Edge parallel to x-axis
                        A[idx, idx] -= 1/dx_plastic_m**2
                        
                        A[idx, idx + 1] = 1/dx_plastic_m**2
                        A[idx, idx - 1] = 1/dx_plastic_m**2
                        
                        if j == 0:
                            A[idx, idx + Nx_plastic] = 1/dy_plastic_m**2
                        else:
                            A[idx, idx - Nx_plastic] = 1/dy_plastic_m**2
                            
                        if k == 0:
                            A[idx, idx + Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                        else:
                            A[idx, idx - Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                            
                elif is_surface:
                    
                    # Add neighboring elements based on surface type
                    if i in [0, Nx_plastic-1]:  # Surface parallel to y-z plane
                        A[idx, idx] -= 1/dy_plastic_m**2 + 1/dz_plastic_m**2
                        
                        A[idx, idx + Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                        A[idx, idx - Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                        A[idx, idx + Nx_plastic] = 1/dy_plastic_m**2
                        A[idx, idx - Nx_plastic] = 1/dy_plastic_m**2
                        if i == 0:
                            A[idx, idx + 1] = 1/dx_plastic_m**2
                        else:
                            A[idx, idx - 1] = 1/dx_plastic_m**2
                            
                    elif j in [0, Ny_plastic-1]:  # Surface parallel to x-z plane
                        A[idx, idx] -= 1/dx_plastic_m**2 + 1/dz_plastic_m**2
                        
                        A[idx, idx + 1] = 1/dx_plastic_m**2
                        A[idx, idx - 1] = 1/dx_plastic_m**2
                        A[idx, idx + Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                        A[idx, idx - Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                        if j == 0:
                            A[idx, idx + Nx_plastic] = 1/dy_plastic_m**2
                        else:
                            A[idx, idx - Nx_plastic] = 1/dy_plastic_m**2
                            
                    elif k in [0, Nz_plastic-1]:  # Surface parallel to x-y plane
                        A[idx, idx] -= 1/dx_plastic_m**2 + 1/dy_plastic_m**2
                        
                        A[idx, idx + 1] = 1/dx_plastic_m**2
                        A[idx, idx - 1] = 1/dx_plastic_m**2
                        A[idx, idx + Nx_plastic] = 1/dy_plastic_m**2
                        A[idx, idx - Nx_plastic] = 1/dy_plastic_m**2
                        if k == 0:
                            A[idx, idx + Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                        else:
                            A[idx, idx - Nx_plastic*Ny_plastic] = 1/dz_plastic_m**2
                            
                A[idx, :] *= k_plastic
                
    return A

def update_matrix_with_boundary_conditions(A, b, metal_coords, plastic_coords, params: ThermalParameters, metal_boundary, plastic_boundary, mapping):
    """Update matrix A based on boundary conditions for both metal and plastic layers"""
    # Metal layer dimensions
    Nx_metal, Ny_metal, Nz_metal = metal_coords.Nx, metal_coords.Ny, metal_coords.Nz
    dx_metal, dy_metal, dz_metal = metal_coords.dx, metal_coords.dy, metal_coords.dz
    dx_metal_m, dy_metal_m, dz_metal_m = dx_metal * 1e-3, dy_metal * 1e-3, dz_metal * 1e-3
    
    # Plastic layer dimensions
    Nx_plastic, Ny_plastic, Nz_plastic = plastic_coords.Nx, plastic_coords.Ny, plastic_coords.Nz
    dx_plastic, dy_plastic, dz_plastic = plastic_coords.dx, plastic_coords.dy, plastic_coords.dz
    dx_plastic_m, dy_plastic_m, dz_plastic_m = dx_plastic * 1e-3, dy_plastic * 1e-3, dz_plastic * 1e-3
    # Get thermal conductivities
    k_metal = params.get_region_thermal_conductivity("metal_layer")
    k_plastic = params.get_region_thermal_conductivity("plastic_layer")
    
    # Update metal layer
    for source_id, elements in metal_boundary.heat_source_elements.items():
        # Get heat source parameters
        heat_source = next((hs for hs in params.heat_sources if hs["id"] == source_id), None)
        if not heat_source:
            continue
            
        # Check if this is a metal layer heat source
        if not heat_source["region_id"] == "metal_layer":
            continue
            
        for idx in elements:
            # Get 3D indices from global index
            k = idx // (Nx_metal * Ny_metal)
            remainder = idx % (Nx_metal * Ny_metal)
            j = remainder // Nx_metal
            i = remainder % Nx_metal
            
            if heat_source["type"] == "CONVECTIVE":
                htc = heat_source["heat_transfer_coefficient"]
                T_inf = heat_source["ambient_temperature"]
                
                # Update diagonal term
                A[idx, idx] -= htc / dz_metal_m
                b[idx] = -htc * T_inf / dz_metal_m
                
            elif heat_source["type"] == "CONST_Qflux":
                q = heat_source["power"]
                Lx_tmp = heat_source["dimensions"]["Lx"]*1e-3
                Ly_tmp = heat_source["dimensions"]["Ly"]*1e-3
                qFlux = q / (Lx_tmp * Ly_tmp)
                b[idx] = -qFlux / dz_metal_m
    
    # Update plastic layer (offset by total_metal)
    total_metal = Nx_metal * Ny_metal * Nz_metal
    for source_id, elements in plastic_boundary.heat_source_elements.items():
        # Get heat source parameters
        heat_source = next((hs for hs in params.heat_sources if hs["id"] == source_id), None)
        if not heat_source:
            continue
            
        # Check if this is a plastic layer heat source
        if not heat_source["region_id"] == "plastic_layer":
            continue
            
        for idx in elements:
            # Convert to global index
            global_idx = total_metal + idx
            
            # Get 3D indices from local index
            k = idx // (Nx_plastic * Ny_plastic)
            remainder = idx % (Nx_plastic * Ny_plastic)
            j = remainder // Nx_plastic
            i = remainder % Nx_plastic
            
            if heat_source["type"] == "CONVECTIVE":
                htc = heat_source["heat_transfer_coefficient"]
                T_inf = heat_source["ambient_temperature"]
                
                # Update diagonal term
                A[global_idx, global_idx] -= htc / dz_plastic_m
                b[global_idx] = -htc * T_inf / dz_plastic_m

            elif heat_source["type"] == "CONST_Qflux":
                q = heat_source["power"]
                Lx_tmp = heat_source["dimensions"]["Lx"]*1e-3
                Ly_tmp = heat_source["dimensions"]["Ly"]*1e-3
                qFlux = q / (Lx_tmp * Ly_tmp)
                b[global_idx] = -qFlux / dz_plastic_m

    # Update matrix for mapped elements
    for mapping_id, mapping_data in params.mappings_by_id.items():
        source_region = mapping_data["source"]["region_id"]
        target_region = mapping_data["target"]["region_id"]
        
        # Get the mapping indices
        source_key = f"{source_region}_source_{mapping_id}"
        target_key = f"{target_region}_target_{mapping_id}"
        
        if mapping_id == "KNEE_ACTUATOR_CONNECTION":
            # Leave space for custom knee actuator connection handling
            k_kneeACT = mapping_data["thermal_conductivity"]
            L_kneeACT = mapping_data["L"]*1e-3
            for source_idx, target_idx in zip(mapping.region_mappings[source_key],
                                                             mapping.region_mappings[target_key]):
                A[source_idx, source_idx] -= k_kneeACT / (L_kneeACT * dz_metal_m)
                A[source_idx, target_idx] = k_kneeACT / (L_kneeACT * dz_metal_m)
                A[target_idx, source_idx] = k_kneeACT / (L_kneeACT * dz_metal_m)
                A[target_idx, target_idx] -= k_kneeACT / (L_kneeACT * dz_metal_m)
            
        else:
            # Handle standard mappings
            for source_idx, target_idx in zip(mapping.region_mappings[source_key],
                                                             mapping.region_mappings[target_key]):
                
                # Adjust target_idx if it's in plastic layer
                if target_region == "plastic_layer":
                    target_idx += total_metal
                    k_target = k_plastic
                    Nx_target = Nx_plastic
                    Ny_target = Ny_plastic
                else:
                    k_target = k_metal
                    Nx_target = Nx_metal
                    Ny_target = Ny_metal
                    
                # Adjust source_idx if it's in plastic layer  
                if source_region == "plastic_layer":
                    source_idx += total_metal
                    k_source = k_plastic
                    Nx_source = Nx_plastic
                    Ny_source = Ny_plastic
                else:
                    k_source = k_metal
                    Nx_source = Nx_metal
                    Ny_source = Ny_metal
                
                # Set coupling terms
                A[source_idx, target_idx] -= k_target / (dz_plastic_m * dz_metal_m)
                A[source_idx, target_idx-Nx_target*Ny_target] = k_target / (dz_plastic_m * dz_metal_m)
                A[target_idx, source_idx] -= k_source / (dz_plastic_m * dz_metal_m)
                A[target_idx, source_idx-Nx_source*Ny_source] = k_source / (dz_plastic_m * dz_metal_m)
            
    return A, b