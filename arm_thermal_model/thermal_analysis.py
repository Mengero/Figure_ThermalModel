import numpy as np
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

def update_matrix_with_geometries(A, region_info: Dict, actuator_info: Dict, metal_conductivity: float) -> lil_matrix:
    """
    Update matrix A based on element positions using pre-calculated indices.
    
    Args:
        A: Global system matrix
        region_info: Dictionary with region information
        actuator_info: Dictionary with actuator information
        metal_conductivity: Metal thermal conductivity
        
    Returns:
        Updated global system matrix
    """
    print("\nUpdating matrix with geometry data...")
    
    # Process each region
    for region_id, info in region_info.items():
        print(f"\nProcessing region: {region_id}")
        coords = info['coords']
        
        # Region dimensions
        Nx, Ny, Nz = coords.Nx, coords.Ny, coords.Nz
        dx, dy, dz = coords.dx, coords.dy, coords.dz
        dx_m, dy_m, dz_m = dx * 1e-3, dy * 1e-3, dz * 1e-3
        
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
            
            # Scale by thermal conductivity
            A[global_idx, :] *= metal_conductivity
        
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
            
            # Scale by thermal conductivity
            A[global_idx, :] *= metal_conductivity
        
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
            
            # Scale by thermal conductivity
            A[global_idx, :] *= metal_conductivity
    
    # Initialize actuator blocks
    for actuator_id, info in actuator_info.items():
        print(f"\nInitializing actuator block: {actuator_id}")
        start_idx = info['start_idx']
        num_unknowns = info['num_unknowns']
        end_idx = start_idx + num_unknowns
        
        # Create a 4x4 block for the actuator's thermal model
        actuator_block = np.zeros((num_unknowns, num_unknowns))
        A[start_idx:end_idx, start_idx:end_idx] = actuator_block
    
    return A

def find_adiabatic_element_pairs(coords, bc_data, total_elements):
    """
    Find pairs of elements that have an adiabatic line between them.
    
    Args:
        coords: ElementCoordinates object
        bc_data: Boundary condition data containing adiabatic line info
        total_elements: Current total number of elements
        
    Returns:
        List of tuples (element1, element2) that are adiabatic to each other
    """
    adiabatic_pairs = []
    source_1 = bc_data.get('source_1', {})
    if not source_1:
        return adiabatic_pairs
        
    # Get the vertical line coordinates
    x_line = source_1['centroid']['x']+1e-3
    y_min = source_1['centroid']['y'] - source_1['height']/2
    y_max = source_1['centroid']['y'] + source_1['height']/2
    
    # For each element, check if the adiabatic line passes between it and its neighbor
    for i in range(coords.Nx - 1):  # Stop at Nx-1 since we check i and i+1
        for j in range(coords.Ny):
            for k in range(coords.Nz):
                # Get coordinates of current element and its right neighbor
                x1, y1, z1 = coords.get_coordinates_from_3d(i, j, k)
                x2, y2, z2 = coords.get_coordinates_from_3d(i+1, j, k)
                
                # Check if adiabatic line passes between these elements
                # The line should be between x1 and x2, and y should be within range
                if (x1 <= x_line <= x2 and 
                    y_min <= y1 <= y_max):  # Use half dz as tolerance
                    
                    # Calculate element indices
                    idx1 = coords.get_global_index(coords.Nx, coords.Ny, i, j, k)
                    idx2 = coords.get_global_index(coords.Nx, coords.Ny, i+1, j, k)
                    
                    # Add to adiabatic pairs with global indices
                    adiabatic_pairs.append({
                        'element1': {
                            'global_idx': total_elements + idx1,
                            'i': i, 'j': j, 'k': k,
                            'x': x1, 'y': y1, 'z': z1
                        },
                        'element2': {
                            'global_idx': total_elements + idx2,
                            'i': i+1, 'j': j, 'k': k,
                            'x': x2, 'y': y2, 'z': z2
                        },
                        'line_x': x_line,
                        'bc_data': bc_data
                    })
    
    return adiabatic_pairs

def update_matrix_with_boundary_conditions(A: lil_matrix, b: np.ndarray, region_info: Dict, actuator_info: Dict, metal_conductivity: float) -> Tuple[lil_matrix, np.ndarray, Dict]:
    """
    Update matrix A and vector b based on boundary conditions using pre-calculated indices.
    Different actuator types (PITCHYAW, ROLL) are handled differently.
    
    Args:
        A: Global system matrix
        b: Global system vector
        region_info: Dictionary with region information
        actuator_info: Dictionary with actuator information
        metal_conductivity: Metal thermal conductivity
        
    Returns:
        Tuple containing:
        - Updated global system matrix
        - Updated global system vector
        - Dictionary containing actuator elements mapping
    """
    print("\nUpdating matrix with boundary conditions...")
    
    # Initialize dictionaries to store elements for each actuator
    actuator_elements = {}
    
    # First pass: Collect all boundary elements for each actuator
    for actuator_id, act_info in actuator_info.items():
        actuator_elements[actuator_id] = {
            'housing_elements': [],
            'gearbox_elements': [],
            'motor_elements': []
        }
        
        # Loop through all regions to find connected elements
        for region_id, info in region_info.items():
            if 'boundary_indices' not in info:
                continue
                
            for bc_type, elements in info['boundary_indices'].items():
                if bc_type == "ACTUATOR_CONNECTED":
                    for element in elements:
                        global_idx = element['global_idx']
                        bc_data = element['bc_data']
                        # Check if this element connects to current actuator
                        if bc_data.get('actuator_id') == actuator_id:
                            connecting_loc = bc_data.get('connecting_location')
                            if connecting_loc == 'housing':
                                actuator_elements[actuator_id]['housing_elements'].append(global_idx)
                            elif connecting_loc == 'gearbox':
                                actuator_elements[actuator_id]['gearbox_elements'].append(global_idx)
                            elif connecting_loc == 'motor':
                                actuator_elements[actuator_id]['motor_elements'].append(global_idx)
    
    # Second pass: Apply actuator couplings using collected elements
    for actuator_id, info in actuator_info.items():
        # Get actuator indices
        start_idx = info['start_idx']
        actuator_type = info['type']
        
        # Get thermal resistances
        R1 = info['thermal_resistance']['R1']  # Gearbox to Winding
        R2 = info['thermal_resistance']['R2']  # Winding to Housing
        R3 = info['thermal_resistance']['R3']  # Housing to Motor
        R4 = info['thermal_resistance']['R4']  # Housing to FETs
        R5 = info['thermal_resistance'].get('R5', 0.0)  # Additional resistance for PITCHYAW
        
        # Get heat sources
        Q_GEARBOX = info['heat_losses'].get('gearbox', 0.0)
        Q_FETS = info['heat_losses'].get('FETs', 0.0)
        Q_MOTOR = info['heat_losses'].get('motor', 0.0)

        # Get elements for this actuator
        housing_elements = actuator_elements[actuator_id]['housing_elements']
        gearbox_elements = actuator_elements[actuator_id]['gearbox_elements']
        motor_elements = actuator_elements[actuator_id]['motor_elements']

        # Both types have 2 unknowns
        T2_idx = start_idx      # Gearbox temperature
        T4_idx = start_idx + 1  # Motor internal temperature
        
        # Equation 1: (T2-T1)/R1 + (T2-T3)/R2 = Q_GEARBOX
        A[T2_idx, T2_idx] = 1/R1 + 1/R2
        for elem_idx in gearbox_elements:
            A[T2_idx, elem_idx] = -1/(R1 * len(gearbox_elements))
        for elem_idx in housing_elements:
            A[T2_idx, elem_idx] = -1/(R2 * len(housing_elements))
        b[T2_idx] = Q_GEARBOX
        
        # Equation 2: (T4-T3)/R3 = Q_MOTOR
        A[T4_idx, T4_idx] = 1/R3
        for elem_idx in housing_elements:
            A[T4_idx, elem_idx] = -1/(R3 * len(housing_elements))
        b[T4_idx] = Q_MOTOR
    
    
    # Process each region
    for current_region_id, info in region_info.items():
        print(f"\nProcessing region: {current_region_id}")
        coords = info['coords']
        
        # Get region dimensions
        Nx, Ny, Nz = coords.Nx, coords.Ny, coords.Nz
        dx, dy, dz = coords.dx, coords.dy, coords.dz
        dx_m, dy_m, dz_m = dx * 1e-3, dy * 1e-3, dz * 1e-3
        
        # Track elements that already have boundary conditions applied
        elements_with_bc = set()
        
        if info['adiabatic_pairs'] is not None:
            for element_pair in info['adiabatic_pairs']:
                # Get indices and data
                idx1 = element_pair['element1']['global_idx']
                idx2 = element_pair['element2']['global_idx']
                # Zero out coupling terms between these elements
                A[idx1, idx2] -= metal_conductivity/dx_m**2
                A[idx1, idx1] += metal_conductivity/dx_m**2
                A[idx2, idx2] += metal_conductivity/dx_m**2
                A[idx2, idx1] -= metal_conductivity/dx_m**2
        
        # Process mapped regions
        if "MAPPED" in info['boundary_indices']:
            for source_element in info['boundary_indices']["MAPPED"]:
                source_idx = source_element['global_idx']
                source_i = source_element['i']
                source_j = source_element['j']
                source_k = source_element['k']
                bc_data = source_element['bc_data']
                
                # Get symmetry axis and region dimensions
                symmetry_axis = bc_data.get('symmetry_axis', 'y')  # default to y-axis symmetry
                
                # Calculate center indices
                center_i = coords.Nx // 2
                center_j = coords.Ny // 2
                
                # Calculate target indices based on symmetry axis
                if symmetry_axis == 'y':
                    # For y-axis symmetry, reflect across vertical line (i changes, j stays same)
                    distance_from_center = source_i - center_i
                    target_i = center_i - distance_from_center
                    target_j = source_j
                    target_k = source_k
                else:  # x-axis symmetry
                    # For x-axis symmetry, reflect across horizontal line (j changes, i stays same)
                    distance_from_center = source_j - center_j
                    target_i = source_i
                    target_j = center_j - distance_from_center
                    target_k = source_k
                
                # Calculate target global index
                target_local_idx = coords.get_global_index(coords.Nx, coords.Ny, target_i, target_j, target_k)
                target_idx = info['start_idx'] + target_local_idx
                
                # Update matrix elements using 1/dx²*k format for thermal coupling
                if symmetry_axis == 'y':
                    coupling_factor = metal_conductivity / (dx_m * dx_m)
                else:  # x-axis
                    coupling_factor = metal_conductivity / (dy_m * dy_m)
                
                # Set up symmetric coupling
                A[source_idx, source_idx] -= coupling_factor
                A[source_idx, target_idx] += coupling_factor
                A[target_idx, source_idx] += coupling_factor
                A[target_idx, target_idx] -= coupling_factor
        
        # Process other boundary conditions
        for bc_type, elements in info['boundary_indices'].items():
            if bc_type == "PLASTIC_COVERED":
                for element in elements:
                    global_idx = element['global_idx']
                    elements_with_bc.add(global_idx)
                    bc_data = element['bc_data']
                    
                    # Get boundary condition parameters
                    plastic_thickness = bc_data.get('plastic_thickness', 1.0) * 1e-3  # Convert to meters
                    plastic_conductivity = bc_data.get('plastic_conductivity', 0.3)  # W/mK
                    htc = bc_data.get('heat_transfer_coefficient', 7.5)  # W/m²K
                    T_inf = bc_data.get('ambient_temperature', 30.0)  # °C
                    
                    # Calculate effective heat transfer coefficient
                    h_eff = 1.0 / (1.0/htc + plastic_thickness/plastic_conductivity)
                    
                    # Update matrix and vector
                    A[global_idx, global_idx] -= h_eff / dz_m
                    b[global_idx] -= h_eff * T_inf / dz_m
            
            elif bc_type == "CONST_Q":
                for element in elements:
                    global_idx = element['global_idx']
                    elements_with_bc.add(global_idx)
                    bc_data = element['bc_data']
                    
                    # Get heat transfer rate and calculate heat flux
                    Q = bc_data.get('q', 0.0)  # W (total heat transfer rate)
                    width = bc_data.get('width', 0.0)  # mm
                    height = bc_data.get('height', 0.0)  # mm
                    area = (width * height) * 1e-6  # Convert to m^2
                    q = Q / area  # W/m^2 (heat flux)
                    
                    # Update vector with heat flux
                    b[global_idx] -= q / dz_m
            
            elif bc_type == "ACTUATOR_CONNECTED":
                for element in elements:
                    global_idx = element['global_idx']
                    elements_with_bc.add(global_idx)
                    bc_data = element['bc_data']
                    
                    # Get actuator info
                    actuator_id = bc_data.get('actuator_id')
                    connecting_location = bc_data.get('connecting_location')
                    if actuator_id not in actuator_info:
                        continue
                        
                    act_info = actuator_info[actuator_id]
                    act_start_idx = act_info['start_idx']
                    actuator_type = act_info['type']
                    
                    # Get actuator-specific elements
                    housing_elements = actuator_elements[actuator_id]['housing_elements']
                    gearbox_elements = actuator_elements[actuator_id]['gearbox_elements']
                    motor_elements = actuator_elements[actuator_id]['motor_elements']
                    
                    # Calculate area
                    width = bc_data.get('width', 0.0)  # mm
                    height = bc_data.get('height', 0.0)  # mm
                    area = (width * height) * 1e-6  # Convert to m^2
                    
                    # Get heat generation and thermal resistances
                    Q_fets = act_info.get('heat_losses', {}).get('FETs', 0.0)
                    q_fets = Q_fets / area if area > 0 else 0
                    
                    R1 = act_info['thermal_resistance']['R1']
                    R2 = act_info['thermal_resistance']['R2']
                    R3 = act_info['thermal_resistance']['R3']
                    R4 = act_info['thermal_resistance']['R4']
                    R5 = act_info['thermal_resistance'].get('R5', 0.0)
                    
                    # Both types have 2 unknowns
                    T2_idx = act_start_idx      # Gearbox temperature
                    T4_idx = act_start_idx + 1  # Motor internal temperature
                    
                    if actuator_type == "ROLL":
                        if connecting_location == "housing":
                            # Loop through elements in boundary region to create coupling between all elements
                            for element_idx in housing_elements:
                                A[global_idx, element_idx] -= (1/R2+1/R3)/area/len(housing_elements)/dz_m
                            A[global_idx, T4_idx] += 1/R3/area/dz_m
                            A[global_idx, T2_idx] += 1/R2/area/dz_m
                            b[global_idx] -= q_fets/area/dz_m
                            
                        elif connecting_location == "gearbox":
                            for element_idx in gearbox_elements:
                                A[global_idx, element_idx] -= 1/R1/area/len(gearbox_elements)/dz_m
                            A[global_idx, T2_idx] += 1/R1/area/dz_m
                    
                    else:  # PITCHYAW type
                        if connecting_location == "housing":
                            # Loop through elements in boundary region to create coupling between all elements
                            for element_idx in housing_elements:
                                A[global_idx, element_idx] -= (1/R2+1/R3+1/R5)/area/len(housing_elements)/dz_m
                            for element_idx in motor_elements:
                                A[global_idx, element_idx] += 1/R5/area/len(motor_elements)/dz_m
                            A[global_idx, T4_idx] += 1/R3/area/dz_m
                            A[global_idx, T2_idx] += 1/R2/area/dz_m
                            b[global_idx] -= q_fets/area/dz_m
                            
                        elif connecting_location == "motor":
                            for element_idx in motor_elements:
                                A[global_idx, element_idx] -= 1/R5/area/len(motor_elements)/dz_m
                            for element_idx in housing_elements:
                                A[global_idx, element_idx] += 1/R5/area/len(housing_elements)/dz_m
                            
                        elif connecting_location == "gearbox":
                            for element_idx in gearbox_elements:
                                A[global_idx, element_idx] -= 1/R1/area/len(gearbox_elements)/dz_m
                            A[global_idx, T2_idx] += 1/R1/area/dz_m
            
            else:
                # Get convection parameters from region data
                region_data = info['region_data']
                htc = region_data.get('heat_transfer_coefficient', 7.5)  # W/m²K (default if not specified)
                T_inf = region_data.get('ambient_temperature', 30.0)  # °C (default if not specified)
                
                # Process all surface, edge, and corner elements that don't have other BCs
                for element_type in ['surface', 'edge', 'corner']:
                    for global_idx, i, j, k in info['element_indices'][element_type]:
                        if global_idx not in elements_with_bc:
                            # Only apply convection to elements on top (k=Nz-1) and bottom (k=0) surfaces
                            if k == Nz-1:
                                # Apply convection boundary condition
                                A[global_idx, global_idx] -= htc / dz_m
                                b[global_idx] -= htc * T_inf / dz_m
                            # Other surfaces (x=0, x=Nx-1, y=0, y=Ny-1) are adiabatic by default
                            # No need to modify matrix for adiabatic conditions

    

    return A, b, actuator_elements

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
            bc_type = bc["type"]
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
                        # Add both top and bottom surface elements if they exist
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
        num_unknowns = 3  # Each actuator has 3 temperature unknowns
        actuator_info[actuator_id] = {
            'start_idx': total_elements,
            'num_unknowns': num_unknowns,
            'type': actuator["type"],
            'heat_losses': actuator["heat_losses"],
            'thermal_resistance': actuator["thermal_resistance"]
        }
        
        # Update total unknowns
        total_unknowns += num_unknowns
        print(f"Actuator {actuator_id} ({actuator['type']}): {num_unknowns} unknowns (Total: {total_elements + total_unknowns})")
    
    # Update total unknowns to include elements
    total_unknowns += total_elements
    
    print(f"\nTotal elements: {total_elements}")
    print(f"Total unknowns: {total_unknowns}")
    
    return region_info, actuator_info, total_elements, total_unknowns