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

def create_layer_matrix(nx, ny, nz, dx, dy, dz):
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
    return A

def create_combined_matrix(params: ThermalParameters):
    """
    Create the combined matrix for all layers in the thermal model.
    
    Parameters:
        params: ThermalParameters object containing model parameters
        
    Returns:
        scipy.sparse matrix: Combined matrix for all layers
    """
    # Get mesh dimensions for each layer
    metal_nx, metal_ny, metal_nz = params.get_mesh_dimensions()
    metal_Lx, metal_Ly, metal_Lz = params.get_region_dimensions("metal_layer")
    metal_dx = metal_Lx / metal_nx
    metal_dy = metal_Ly / metal_ny
    metal_dz = metal_Lz / metal_nz
    
    # Create matrix for metal layer
    A_metal = create_layer_matrix(metal_nx, metal_ny, metal_nz, metal_dx, metal_dy, metal_dz)
    
    # Get dimensions for plastic layer
    plastic_Lx, plastic_Ly, plastic_Lz = params.get_region_dimensions("plastic_layer")
    plastic_nx = int(round(plastic_Lx / metal_dx))  # Use same dx as metal layer
    plastic_ny = int(round(plastic_Ly / metal_dy))  # Use same dy as metal layer
    plastic_nz = metal_nz  # Use same nz as metal layer
    plastic_dz = plastic_Lz / plastic_nz
    
    # Create matrix for plastic layer
    A_plastic = create_layer_matrix(plastic_nx, plastic_ny, plastic_nz, metal_dx, metal_dy, plastic_dz)
    
    # Combine matrices in block diagonal form
    A_combined = block_diag([A_metal, A_plastic])
    
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