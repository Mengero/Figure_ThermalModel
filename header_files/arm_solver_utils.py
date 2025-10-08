import numpy as np
import matplotlib.pyplot as plt
import json5
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.linalg import norm
import time
from scipy.sparse.linalg import spsolve, spilu

from thermal_utils import read_geo_file
from thermal_parameters import get_parameters, ThermalParameters
from thermal_analysis import (
    create_layer_coordinates, 
    update_matrix_with_geometries,
    preconditioner,
    ElementCoordinates,
    create_layer_matrix,
    calculate_system_size
)
from boundary_conditions import ElementBoundary, BoundaryCondition
from visualization import plot_layer_surface, plot_boundary_conditions
from typing import List, Tuple, Dict

def read_geometry(file_path: str = "GEO.json") -> Tuple[dict, ThermalParameters, List[dict], List[dict], List[dict]]:
    """
    Read and parse the geometry file.
    """
    with open(file_path, "r") as f:
        geo_data = json5.load(f)
    params = get_parameters(geo_data)
    regions = geo_data["regions"]
    actuators = geo_data["actuators"]
    node_networks = geo_data.get("node_networks", [])
    return geo_data, params, regions, actuators, node_networks

def initialize_global_matrix(total_elements: int, region_info: Dict, actuator_info: Dict, node_networks: List[dict]) -> Tuple[scipy.sparse.lil_matrix, np.ndarray, dict]:
    """
    Initialize the global system matrix and vector, including special unknowns for GEARBOX_HAND.
    """
    print("\n" + "=" * 50)
    print("\nInitializing global matrix and vector...")
    total_actuator_unknowns = 0
    for info in actuator_info.values():
        total_actuator_unknowns += 3 # motor, gearbox, fets
    # Node networks: add one unknown per node
    total_node_unknowns = sum(len(net.get('nodes', [])) for net in node_networks)
    total_size = total_elements + total_actuator_unknowns + total_node_unknowns
    A = scipy.sparse.lil_matrix((total_size, total_size))
    b = np.zeros(total_size)
    for region_id, info in region_info.items():
        print(f"\nBuilding matrix block for region: {region_id}")
        coords = info['coords']
        thermal_conductivity = info['region_data']['thermal_conductivity']
        # Region dimensions
        Nx, Ny, Nz = coords.Nx, coords.Ny, coords.Nz
        dx, dy, dz = coords.dx, coords.dy, coords.dz
        dx_m, dy_m, dz_m = dx * 1e-3, dy * 1e-3, dz * 1e-3
        start_idx = info['start_idx']
        num_elements = info['num_elements']
        region_matrix = create_layer_matrix(Nx, Ny, Nz,
                                          dx_m, dy_m, dz_m,
                                          thermal_conductivity)
        end_idx = start_idx + num_elements
        A[start_idx:end_idx, start_idx:end_idx] = region_matrix
        print(f"Added block from index {start_idx} to {end_idx}")
    current_actuator_idx = total_elements
    for actuator_id, info in actuator_info.items():
        print(f"\nInitializing matrix block for actuator: {actuator_id}")
        num_unknowns = 3
        info['start_idx'] = current_actuator_idx
        info['num_unknowns'] = num_unknowns
        end_idx = current_actuator_idx + num_unknowns
        actuator_block = np.zeros((num_unknowns, num_unknowns))
        A[current_actuator_idx:end_idx, current_actuator_idx:end_idx] = actuator_block
        print(f"Added actuator block from index {current_actuator_idx} to {end_idx} (size: {num_unknowns}x{num_unknowns})")
        current_actuator_idx = end_idx
    # Map node IDs to global indices for later coupling
    node_index_map = {}
    current_node_idx = total_elements + total_actuator_unknowns
    for net in node_networks:
        for node in net.get('nodes', []):
            node_index_map[node['id']] = current_node_idx
            current_node_idx += 1
    
    # Store node index mapping in actuator_info for better later usage
    for actuator_id, info in actuator_info.items():
        info['node_index_map'] = node_index_map
    
    # Initialize node network blocks 
    if total_node_unknowns > 0:
        print(f"\nInitializing matrix block for node networks: {total_node_unknowns} nodes")
        node_start_idx = total_elements + total_actuator_unknowns
        node_end_idx = node_start_idx + total_node_unknowns
        # Initialize diagonal entries for nodes (will be modified by boundary conditions)
        for i in range(node_start_idx, node_end_idx):
            A[i, :] = 0.0  # Default diagonal value, will be updated by coupling
            b[i] = 0.0
        print(f"Added node network block from index {node_start_idx} to {node_end_idx}")
    return A, b, node_index_map

def visualize_regions(regions: List[dict], region_info: Dict, params: ThermalParameters) -> None:
    """
    Create visualization plots for all regions.
    """
    num_regions = len(regions)
    num_cols = 3
    num_rows = (num_regions + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6*num_rows))
    axes = axes.flatten()
    for idx, region in enumerate(regions):
        region_id = region["id"]
        print(f"\nProcessing region: {region_id}")
        info = region_info[region_id]
        coords = info['coords']
        adiabatic_pairs = info.get('adiabatic_pairs', [])
        boundary = ElementBoundary(coords, None, region_id, params)
        boundary_conditions = boundary.label_boundary_conditions()
        ax = axes[idx]
        plot_boundary_conditions(coords, boundary_conditions, f"{region_id} (z=0)", 0, ax, adiabatic_pairs)
        width = region["width"]
        height = region["height"]
        thickness = region["thickness"]
        ax.set_title(f"{region_id}\nSize: {width}x{height}x{thickness} mm", fontsize=12, pad=10)
    for idx in range(num_regions, len(axes)):
        fig.delaxes(axes[idx])
    fig.suptitle("Boundary Conditions for All Regions (z=0 surface)", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def prepare_system():
    """
    Prepare the system: read geometry, calculate system size, visualize, and initialize matrices.
    Returns a dictionary with all necessary data for solving.
    """
    geo_data, params, regions, actuators, node_networks = read_geometry()
    region_info, actuator_info, total_elements, actuator_start_idx = calculate_system_size(
        regions, actuators, params
    )
    # visualize_regions(regions, region_info, params)
    A, b, node_index_map = initialize_global_matrix(total_elements, region_info, actuator_info, node_networks)
    A = update_matrix_with_geometries(A, region_info, actuator_info)
    A, b, actuator_elements, node_connected_elements = update_matrix_with_boundary_conditions(A, b, region_info, actuator_info, node_index_map, geo_data)
    return dict(
        geo_data=geo_data,
        params=params,
        regions=regions,
        actuators=actuators,
        region_info=region_info,
        actuator_info=actuator_info,
        total_elements=total_elements,
        actuator_start_idx=actuator_start_idx,
        A=A,
        b=b,
        actuator_elements=actuator_elements,
        node_connected_elements=node_connected_elements,
        node_index_map=node_index_map
    )

def solve_system(system_data):
    """
    Solve the system matrix (direct or iterative) and return the solution and relevant info.
    """
    import numpy as np
    import scipy.sparse
    import time
    from scipy.sparse.linalg import spsolve, spilu, LinearOperator, gmres
    from thermal_analysis import preconditioner
    A = system_data['A']
    b = system_data['b']
    u0 = np.ones_like(b) * 40.0
    rel_tol = 1e-4
    iteration = 0
    solve_time = None
    exitCode = None
    def callback(pr_norm):
        nonlocal iteration
        iteration += 1
        print(f"Iteration {iteration}: residual norm = {pr_norm:.4e}", end="\r")
    A = A.tocsr()
    A_final = scipy.sparse.csr_matrix(A)
    try:
        print("\n" + "=" * 50)
        print("Attempting direct sparse solve...")
        start_time = time.time()
        u = spsolve(A, b)
        solve_time = time.time() - start_time
        print(f"Direct solve completed in {solve_time:.2f} seconds")
        exitCode = 0
    except Exception as e:
        print(f"Direct solve failed: {e}")
        print("Switching to iterative solver with ILU preconditioner...")
        try:
            print("Computing ILU preconditioner...")
            A_csc = A.tocsc()
            ILU = spilu(A_csc, drop_tol=1e-4, fill_factor=20)
            M_x = lambda x: ILU.solve(x)
            M = LinearOperator(A.shape, M_x)
            print("ILU preconditioner ready")
        except Exception as e:
            print(f"ILU preconditioner failed: {e}")
            print("Falling back to diagonal preconditioner...")
            M = preconditioner(A)
        solver = gmres
        name = "GMRES"
        try:
            print(f"\nTrying {solver} solver...")
            start_time = time.time()
            u, exitCode = solver(A, b, M=M, x0=u0, atol=rel_tol, callback=callback, callback_type='pr_norm')
            solve_time = time.time() - start_time
            if exitCode == 0:
                print(f"\n{name} solve succeeded!")
            else:
                print(f"\n{name} did not converge.")
        except Exception as e:
            print(f"{name} solver failed: {e}")
            u = None
    residual = A_final @ u - b
    residual_norm = np.linalg.norm(residual)
    return dict(
        u=u,
        solve_time=solve_time,
        exitCode=exitCode,
        residual_norm=residual_norm,
        **system_data
    )

def postprocess_results(solution_data):
    """
    Print, save, and plot results based on the solution.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    u = solution_data['u']
    solve_time = solution_data['solve_time']
    exitCode = solution_data['exitCode']
    residual_norm = solution_data['residual_norm']
    region_info = solution_data['region_info']
    actuator_info = solution_data['actuator_info']
    actuator_elements = solution_data['actuator_elements']
    special_unknowns = solution_data.get('special_unknowns', {})
    geo_data = solution_data.get('geo_data', {})
    
    # Get global environment parameters
    env_data = geo_data.get('environment', {})
    T_inf_global = env_data.get('ambient_temperature')
    htc_global = env_data.get('heat_transfer_coefficient')
    if T_inf_global is None:
        raise ValueError("Ambient temperature must be specified in the environment data")
    elif htc_global is None:
        raise ValueError("Heat transfer coefficient must be specified in the environment data")
    
    # Print temperatures and heat transfer rates for each region
    print("\n" + "=" * 50)
    print("\nTemperature and Heat Transfer Results:")
    
    # Calculate heat transfer rates for each region
    global_heat_balance = 0.0  # Total energy balance across all regions
    
    for region_id, info in region_info.items():
        ratio_1 = 1
        start_idx = info['start_idx']
        num_elements = info['num_elements']
        region_temps = u[start_idx:start_idx + num_elements]
        coords = info['coords']
        thermal_conductivity = info['region_data']['thermal_conductivity']
        dx_m, dy_m, dz_m = coords.dx * 1e-3, coords.dy * 1e-3, coords.dz * 1e-3
        A_region = info['region_data']['width'] * info['region_data']['height'] * 1e-6 * 2
        A_convection = A_region
        
        print(f"\nRegion: {region_id}")
        print(f"Average Temperature: {np.mean(region_temps):.2f}°C")
        print(f"Min Temperature: {np.min(region_temps):.2f}°C")
        print(f"Max Temperature: {np.max(region_temps):.2f}°C")
        
        # Calculate heat transfer rates for different boundary conditions
        region_heat_transfer = 0.0
        
        # Track elements that have specific boundary conditions applied (to skip in default convection)
        elements_with_bc = set()
        
        # Process PLASTIC_COVERED boundary conditions (identify unique BCs by parameters)
        if "PLASTIC_COVERED" in info['boundary_indices']:
            plastic_heat_transfer = 0.0
            plastic_elements = info['boundary_indices']["PLASTIC_COVERED"]
            
            # First, identify unique boundary conditions by their parameters
            unique_bcs = {}
            for element in plastic_elements:
                bc_data = element['bc_data']
                bc_key = (bc_data.get('plastic_thickness', 1.0), 
                         bc_data.get('plastic_conductivity', 0.25),
                         bc_data.get('heat_transfer_coefficient', htc_global),
                         bc_data.get('ambient_temperature', T_inf_global),
                         bc_data.get('width', 0), 
                         bc_data.get('height', 0),
                         bc_data.get('centroid', {}).get('x', 0),
                         bc_data.get('centroid', {}).get('y', 0),
                         bc_data.get('centroid', {}).get('z', 0))
                
                if bc_key not in unique_bcs:
                    unique_bcs[bc_key] = {
                        'bc_data': bc_data,
                        'comments': bc_data.get('comments', f'PLASTIC_{len(unique_bcs)+1}'),
                        'heat_total': 0.0,
                        'temp_sum': 0.0,
                        'element_count': 0
                    }
            
            print(f"  Processing {len(unique_bcs)} original PLASTIC_COVERED boundary condition(s):")
            
            # Loop through all elements once and assign to appropriate boundary condition
            for element in plastic_elements:
                bc_data = element['bc_data']
                global_idx = element['global_idx']
                
                # Track this element as having a specific boundary condition
                elements_with_bc.add(global_idx)
                
                # Find which boundary condition this element belongs to
                bc_key = (bc_data.get('plastic_thickness', 1.0), 
                         bc_data.get('plastic_conductivity', 0.25),
                         bc_data.get('heat_transfer_coefficient', htc_global),
                         bc_data.get('ambient_temperature', T_inf_global),
                         bc_data.get('width', 0), 
                         bc_data.get('height', 0),
                         bc_data.get('centroid', {}).get('x', 0),
                         bc_data.get('centroid', {}).get('y', 0),
                         bc_data.get('centroid', {}).get('z', 0))
                
                # Calculate heat transfer for this element using matrix physics
                plastic_thickness = bc_data.get('plastic_thickness', 1.0) * 1e-3  # Convert mm to m
                plastic_conductivity = bc_data.get('plastic_conductivity', 0.25)
                htc = bc_data.get('heat_transfer_coefficient', htc_global)
                T_inf = bc_data.get('ambient_temperature', T_inf_global)
                R_contact = bc_data.get('contact_resistance', 0.0)  # m²·K/W
                
                # Same physics as matrix assembly: h_eff = 1/(1/htc + thickness/k + R_contact)
                h_eff = 1.0 / (1.0/htc + plastic_thickness/plastic_conductivity + R_contact)
                
                element_temp = u[global_idx]
                # Heat transfer: Q = h_eff * A * (T - T_inf), matches matrix: -h_eff*(T-T_inf)/dz
                element_heat = -h_eff * (element_temp - T_inf)
                
                # Accumulate for this boundary condition
                unique_bcs[bc_key]['heat_total'] += element_heat
                unique_bcs[bc_key]['temp_sum'] += element_temp
                unique_bcs[bc_key]['element_count'] += 1
            
            # Display results for each boundary condition
            for bc_key, bc_info in unique_bcs.items():
                bc_data = bc_info['bc_data']
                avg_temp = bc_info['temp_sum'] / bc_info['element_count']
                total_area = bc_info['element_count'] * dx_m * dy_m * 1e6  # Convert to mm²
                
                plastic_thickness = bc_data.get('plastic_thickness', 1.0)  # Keep in mm for display
                plastic_conductivity = bc_data.get('plastic_conductivity', 0.25)
                htc = bc_data.get('heat_transfer_coefficient', htc_global)
                h_eff = 1.0 / (1.0/htc + (plastic_thickness*1e-3)/plastic_conductivity)
                
                print(f"    {bc_info['comments']}: t={plastic_thickness:.1f}mm, k={plastic_conductivity:.3f}W/m·K, h_eff={h_eff:.1f}W/m²K, Area={total_area:.1f}mm², Heat={bc_info['heat_total']:.3f}W, Avg_T={avg_temp:.2f}°C")
                plastic_heat_transfer += bc_info['heat_total']
            
            print(f"  Total Plastic Covered Surface Heat Transfer: {plastic_heat_transfer:.2f} W")
            region_heat_transfer += plastic_heat_transfer
        
        # Process CONST_Q boundary conditions (identify unique BCs by parameters)
        if "CONST_Q" in info['boundary_indices']:
            const_q_heat = 0.0
            const_q_elements = info['boundary_indices']["CONST_Q"]
            
            # First, identify unique boundary conditions by their parameters
            unique_bcs = {}
            for element in const_q_elements:
                bc_data = element['bc_data']
                bc_key = (bc_data.get('q', 0), 
                         bc_data.get('width', 0), 
                         bc_data.get('height', 0),
                         bc_data.get('centroid', {}).get('x', 0),
                         bc_data.get('centroid', {}).get('y', 0),
                         bc_data.get('centroid', {}).get('z', 0))
                
                if bc_key not in unique_bcs:
                    unique_bcs[bc_key] = {
                        'bc_data': bc_data,
                        'comments': bc_data.get('comments', f'CONST_Q_{len(unique_bcs)+1}'),
                        'heat_total': 0.0,
                        'temp_sum': 0.0,
                        'element_count': 0
                    }
            
            print(f"  Processing {len(unique_bcs)} original CONST_Q boundary condition(s):")
            
            # Loop through all elements once and assign to appropriate boundary condition
            for element in const_q_elements:
                bc_data = element['bc_data']
                global_idx = element['global_idx']
                
                # Find which boundary condition this element belongs to
                bc_key = (bc_data.get('q', 0), 
                         bc_data.get('width', 0), 
                         bc_data.get('height', 0),
                         bc_data.get('centroid', {}).get('x', 0),
                         bc_data.get('centroid', {}).get('y', 0),
                         bc_data.get('centroid', {}).get('z', 0))
                
                # Calculate heat transfer for this element using matrix physics
                # From matrix: Q is total heat transfer rate (W), q is heat flux (W/m²)
                q_tmp = bc_data['q'] / (bc_data['width'] * bc_data['height'] * 1e-6) # W (total heat transfer rate for this BC)
                
                element_temp = u[global_idx]
                
                # Accumulate for this boundary condition
                unique_bcs[bc_key]['heat_total'] += q_tmp * dx_m * dy_m
                unique_bcs[bc_key]['temp_sum'] += element_temp
                unique_bcs[bc_key]['element_count'] += 1
            
            # Display results for each boundary condition
            for bc_key, bc_info in unique_bcs.items():
                avg_temp = bc_info['temp_sum'] / bc_info['element_count']
                total_area = bc_info['bc_data']['width'] * bc_info['bc_data']['height'] # Convert to mm²
                E_const_Q = bc_info['heat_total']
                heat_total = bc_info['heat_total'] * total_area * 1e-6 / (bc_info['element_count'] * dx_m * dy_m)
                # ratio_1 = heat_total / E_const_Q
                
                print(f"    {bc_info['comments']}: Area={total_area:.1f} mm², Heat={E_const_Q:.3f} W, Avg_T={avg_temp:.2f}°C")
                const_q_heat += E_const_Q
            
            print(f"  Total Constant Heat Flux Surface Heat Transfer: {const_q_heat:.2f} W")
            region_heat_transfer += const_q_heat
        
        # Process ACTUATOR_CONNECTED boundary conditions (group by actuator_id and connecting_location)
        if "ACTUATOR_CONNECTED" in info['boundary_indices']:
            actuator_heat_transfer = 0.0
            actuator_boundary_elements = info['boundary_indices']["ACTUATOR_CONNECTED"]
            A_actuator = 0
            
            # Group elements by actuator_id and connecting_location only
            actuator_groups = {}
            for element in actuator_boundary_elements:
                bc_data = element['bc_data']
                actuator_id = bc_data.get('actuator_id', '')
                connecting_location = bc_data.get('connecting_location', '')
                group_key = (actuator_id, connecting_location)
                
                if group_key not in actuator_groups:
                    actuator_groups[group_key] = {
                        'actuator_id': actuator_id,
                        'connecting_location': connecting_location,
                        'temp_sum': 0.0,
                        'element_count': 0
                    }
                
                # Accumulate temperature and count for averaging
                global_idx = element['global_idx']
                element_temp = u[global_idx]
                
                # Track this element as having a specific boundary condition
                elements_with_bc.add(global_idx)
                
                actuator_groups[group_key]['temp_sum'] += element_temp
                actuator_groups[group_key]['element_count'] += 1
            
            print(f"  Processing {len(actuator_groups)} ACTUATOR_CONNECTED group(s):")
            
            # Calculate heat transfer for each actuator-location group
            for group_key, group_info in actuator_groups.items():
                actuator_id = group_info['actuator_id']
                connecting_location = group_info['connecting_location']
                
                # Calculate average temperature for this actuator-location group
                avg_temp = group_info['temp_sum'] / group_info['element_count']
                total_area = group_info['element_count'] * dx_m * dy_m * 1e6  # Convert to mm²
                A_actuator += total_area * 1e-6
                heat_transfer = 0.0
                if actuator_id in actuator_info:
                    T_FETs = u[actuator_info[actuator_id]['start_idx']]
                    T_Motor = u[actuator_info[actuator_id]['start_idx'] + 1]
                    T_Gearbox = u[actuator_info[actuator_id]['start_idx'] + 2]
                    R1 = actuator_info[actuator_id]['thermal_resistance']['R1']
                    R2 = actuator_info[actuator_id]['thermal_resistance']['R2']
                    R3 = actuator_info[actuator_id]['thermal_resistance']['R3']
                    R4 = actuator_info[actuator_id]['thermal_resistance']['R4']
                    R5 = actuator_info[actuator_id]['thermal_resistance']['R5']
                    
                    # Calculate heat transfer using average temperature
                    if connecting_location == 'Housing':
                        heat_transfer = ((T_FETs - avg_temp)/R1 + (T_Motor - avg_temp)/R2 + (T_Gearbox - avg_temp)/R4)
                    elif connecting_location == 'Output':
                        heat_transfer = (T_Gearbox - avg_temp)/R5
                
                print(f"    ACTUATOR_{actuator_id}_{connecting_location}: Area={total_area:.1f}mm², Heat={heat_transfer:.3f}W, Avg_T={avg_temp:.2f}°C")
                actuator_heat_transfer += heat_transfer
            
            print(f"  Total Actuator Connected Surface Heat Transfer: {actuator_heat_transfer:.2f} W")
            region_heat_transfer += actuator_heat_transfer
        
        # Process CONST_T boundary conditions (identify unique BCs by parameters)
        # if "CONST_T" in info['boundary_indices']:
        #     const_t_heat = 0.0
        #     const_t_elements = info['boundary_indices']["CONST_T"]
            
        #     # First, identify unique boundary conditions by their parameters
        #     unique_bcs = {}
        #     for element in const_t_elements:
        #         bc_data = element['bc_data']
        #         bc_key = (bc_data.get('temperature', 0.0),
        #                  bc_data.get('width', 0), 
        #                  bc_data.get('height', 0),
        #                  bc_data.get('centroid', {}).get('x', 0),
        #                  bc_data.get('centroid', {}).get('y', 0),
        #                  bc_data.get('centroid', {}).get('z', 0))
                
        #         if bc_key not in unique_bcs:
        #             unique_bcs[bc_key] = {
        #                 'bc_data': bc_data,
        #                 'comments': bc_data.get('comments', f'CONST_T_{bc_data.get("temperature", 0):.1f}C'),
        #                 'heat_total': 0.0,
        #                 'temp_sum': 0.0,
        #                 'element_count': 0
        #             }
            
        #     print(f"  Processing {len(unique_bcs)} original CONST_T boundary condition(s):")
            
        #     # Loop through all elements once and assign to appropriate boundary condition
        #     for element in const_t_elements:
        #         bc_data = element['bc_data']
        #         global_idx = element['global_idx']
                
        #         # Find which boundary condition this element belongs to
        #         bc_key = (bc_data.get('temperature', 0.0),
        #                  bc_data.get('width', 0), 
        #                  bc_data.get('height', 0),
        #                  bc_data.get('centroid', {}).get('x', 0),
        #                  bc_data.get('centroid', {}).get('y', 0),
        #                  bc_data.get('centroid', {}).get('z', 0))
                
        #         element_temp = u[global_idx]
        #         set_temperature = bc_data.get('temperature', 0.0)
                
        #         # For CONST_T, calculate the heat transfer needed to maintain the fixed temperature
        #         # This is done by calculating what heat would flow in/out based on temperature differences
        #         # with neighboring elements (same physics as matrix assembly)
                
        #         # Get element position and neighbors
        #         coords = info['coords']
        #         start_idx = info['start_idx']
        #         thermal_conductivity = info['region_data']['thermal_conductivity']
                
        #         # Find the 3D position of this element
        #         local_idx = global_idx - start_idx
        #         Nx, Ny, Nz = coords.Nx, coords.Ny, coords.Nz
                
        #         # Convert local index to i,j,k coordinates
        #         k = local_idx // (Nx * Ny)
        #         remainder = local_idx % (Nx * Ny)
        #         j = remainder // Nx
        #         i = remainder % Nx
                
        #         # Calculate heat flow to/from neighbors to maintain set temperature
        #         element_heat = 0.0
                
        #         # Check each direction for neighbors and calculate conduction heat transfer
        #         # X-direction neighbors
        #         if i > 0:  # Left neighbor
        #             neighbor_idx = start_idx + coords.get_global_index(Nx, Ny, i-1, j, k)
        #             neighbor_temp = u[neighbor_idx]
        #             area = dy_m * dz_m
        #             q_x = thermal_conductivity * area * (neighbor_temp - set_temperature) / dx_m
        #             element_heat += q_x
                
        #         if i < Nx-1:  # Right neighbor
        #             neighbor_idx = start_idx + coords.get_global_index(Nx, Ny, i+1, j, k)
        #             neighbor_temp = u[neighbor_idx]
        #             area = dy_m * dz_m
        #             q_x = thermal_conductivity * area * (neighbor_temp - set_temperature) / dx_m
        #             element_heat += q_x
                
        #         # Y-direction neighbors
        #         if j > 0:  # Bottom neighbor
        #             neighbor_idx = start_idx + coords.get_global_index(Nx, Ny, i, j-1, k)
        #             neighbor_temp = u[neighbor_idx]
        #             area = dx_m * dz_m
        #             q_y = thermal_conductivity * area * (neighbor_temp - set_temperature) / dy_m
        #             element_heat += q_y
                
        #         if j < Ny-1:  # Top neighbor
        #             neighbor_idx = start_idx + coords.get_global_index(Nx, Ny, i, j+1, k)
        #             neighbor_temp = u[neighbor_idx]
        #             area = dx_m * dz_m
        #             q_y = thermal_conductivity * area * (neighbor_temp - set_temperature) / dy_m
        #             element_heat += q_y
                
        #         # Z-direction neighbors
        #         if k > 0:  # Bottom neighbor
        #             neighbor_idx = start_idx + coords.get_global_index(Nx, Ny, i, j, k-1)
        #             neighbor_temp = u[neighbor_idx]
        #             area = dx_m * dy_m
        #             q_z = thermal_conductivity * area * (neighbor_temp - set_temperature) / dz_m
        #             element_heat += q_z
                
        #         if k < Nz-1:  # Top neighbor
        #             neighbor_idx = start_idx + coords.get_global_index(Nx, Ny, i, j, k+1)
        #             neighbor_temp = u[neighbor_idx]
        #             area = dx_m * dy_m
        #             q_z = thermal_conductivity * area * (neighbor_temp - set_temperature) / dz_m
        #             element_heat += q_z
                
        #         # Accumulate for this boundary condition
        #         unique_bcs[bc_key]['heat_total'] += element_heat
        #         unique_bcs[bc_key]['temp_sum'] += element_temp
        #         unique_bcs[bc_key]['element_count'] += 1
            
        #     # Display results for each boundary condition
        #     for bc_key, bc_info in unique_bcs.items():
        #         total_area = bc_info['bc_data']['width'] * bc_info['bc_data']['height']  # Convert to mm²
        #         ratio_1 = total_area * 1e-6 / (bc_info['element_count'] * dx_m * dy_m)
        #         avg_temp = bc_info['temp_sum'] / bc_info['element_count']
        #         heat_total = bc_info['heat_total'] * ratio_1
                
        #         print(f"    {bc_info['comments']}: Area={total_area:.1f}mm², Heat={heat_total:.3f}W, Avg_T={avg_temp:.2f}°C")
        #         const_t_heat += heat_total
            
        #     print(f"  Total Constant Temperature Heat Transfer: {const_t_heat:.2f} W")
        #     region_heat_transfer += const_t_heat
        
        # Process NODE_CONNECTED boundary conditions (identify unique BCs by parameters)
        if "NODE_CONNECTED" in info['boundary_indices']:
            node_heat_transfer = 0.0
            node_elements = info['boundary_indices']["NODE_CONNECTED"]
            
            # First, identify unique boundary conditions by their parameters
            unique_bcs = {}
            for element in node_elements:
                bc_data = element['bc_data']
                bc_key = (bc_data.get('node_id', ''),
                         bc_data.get('thermal_resistance', 1.0),
                         bc_data.get('width', 0), 
                         bc_data.get('height', 0),
                         bc_data.get('centroid', {}).get('x', 0),
                         bc_data.get('centroid', {}).get('y', 0),
                         bc_data.get('centroid', {}).get('z', 0))
                
                if bc_key not in unique_bcs:
                    unique_bcs[bc_key] = {
                        'bc_data': bc_data,
                        'comments': bc_data.get('comments', f'NODE_{bc_data.get("node_id", "UNKNOWN")}'),
                        'heat_total': 0.0,
                        'temp_sum': 0.0,
                        'element_count': 0
                    }
            
            print(f"  Processing {len(unique_bcs)} original NODE_CONNECTED boundary condition(s):")
            
            # First pass: collect temperatures for each boundary condition
            for element in node_elements:
                bc_data = element['bc_data']
                global_idx = element['global_idx']
                
                # Track this element as having a specific boundary condition
                elements_with_bc.add(global_idx)
                
                # Find which boundary condition this element belongs to
                bc_key = (bc_data.get('node_id', ''),
                         bc_data.get('thermal_resistance', 1.0),
                         bc_data.get('width', 0), 
                         bc_data.get('height', 0),
                         bc_data.get('centroid', {}).get('x', 0),
                         bc_data.get('centroid', {}).get('y', 0),
                         bc_data.get('centroid', {}).get('z', 0))
                
                element_temp = u[global_idx]
                
                # Accumulate temperature and count for averaging
                unique_bcs[bc_key]['temp_sum'] += element_temp
                unique_bcs[bc_key]['element_count'] += 1
            
            # Second pass: calculate heat transfer using average temperature for each BC
            for bc_key, bc_info in unique_bcs.items():
                bc_data = bc_info['bc_data']
                
                # Calculate average temperature for this boundary condition
                avg_temp = bc_info['temp_sum'] / bc_info['element_count']
                
                thermal_resistance = bc_data.get('thermal_resistance', 1.0)
                node_id = bc_data.get('node_id')
                
                node_idx = solution_data['node_index_map'][node_id]
                node_temp = u[node_idx]
                
                # Calculate heat transfer: Q = (T_surface - T_node) / R_thermal
                bc_info['heat_total'] = -(avg_temp - node_temp) / thermal_resistance
            
            # Display results for each boundary condition
            for bc_key, bc_info in unique_bcs.items():
                avg_temp = bc_info['temp_sum'] / bc_info['element_count']
                total_area = bc_info['element_count'] * dx_m * dy_m * 1e6  # Convert to mm²
                
                print(f"    {bc_info['comments']}: Area={total_area:.1f}mm², Heat={bc_info['heat_total']:.3f}W, Avg_T={avg_temp:.2f}°C")
                # node_heat_transfer += bc_info['heat_total']
                node_heat_transfer += bc_info['heat_total'] * total_area * 1e-6 / (bc_info['bc_data']['width'] * bc_info['bc_data']['height'] * 1e-6)
            
            print(f"  Total Node Connected Heat Transfer: {node_heat_transfer:.2f} W")
            region_heat_transfer += node_heat_transfer
        
        # Process USERDEF_CONVECTION boundary conditions (identify unique BCs by parameters)
        if "USERDEF_CONVECTION" in info['boundary_indices']:
            convection_heat_transfer = 0.0
            convection_elements = info['boundary_indices']["USERDEF_CONVECTION"]
            
            # First, identify unique boundary conditions by their parameters
            unique_bcs = {}
            for element in convection_elements:
                bc_data = element['bc_data']
                bc_key = (bc_data.get('heat_transfer_coefficient', htc_global),
                         bc_data.get('ambient_temperature', T_inf_global),
                         bc_data.get('width', 0), 
                         bc_data.get('height', 0),
                         bc_data.get('centroid', {}).get('x', 0),
                         bc_data.get('centroid', {}).get('y', 0),
                         bc_data.get('centroid', {}).get('z', 0))
                
                if bc_key not in unique_bcs:
                    unique_bcs[bc_key] = {
                        'bc_data': bc_data,
                        'comments': bc_data.get('comments', f'USERDEF_CONV_{len(unique_bcs)+1}'),
                        'heat_total': 0.0,
                        'temp_sum': 0.0,
                        'element_count': 0
                    }
            
            print(f"  Processing {len(unique_bcs)} original USERDEF_CONVECTION boundary condition(s):")
            
            # Loop through all elements once and assign to appropriate boundary condition
            for element in convection_elements:
                bc_data = element['bc_data']
                global_idx = element['global_idx']
                
                # Track this element as having a specific boundary condition
                elements_with_bc.add(global_idx)
                
                # Find which boundary condition this element belongs to
                bc_key = (bc_data.get('heat_transfer_coefficient', htc_global),
                         bc_data.get('ambient_temperature', T_inf_global),
                         bc_data.get('width', 0), 
                         bc_data.get('height', 0),
                         bc_data.get('centroid', {}).get('x', 0),
                         bc_data.get('centroid', {}).get('y', 0),
                         bc_data.get('centroid', {}).get('z', 0))
                
                # Calculate heat transfer using matrix physics
                htc = bc_data.get('heat_transfer_coefficient', htc_global)
                T_inf = bc_data.get('ambient_temperature', T_inf_global)
                element_temp = u[global_idx]
                element_area = dx_m * dy_m  # Element area in m²
                # Heat transfer: Q = htc * A * (T - T_inf), matches matrix: -htc*(T-T_inf)/dz
                element_heat = htc * element_area * (element_temp - T_inf)
                
                # Accumulate for this boundary condition
                unique_bcs[bc_key]['heat_total'] += element_heat
                unique_bcs[bc_key]['temp_sum'] += element_temp
                unique_bcs[bc_key]['element_count'] += 1
            
            # Display results for each boundary condition
            for bc_key, bc_info in unique_bcs.items():
                avg_temp = bc_info['temp_sum'] / bc_info['element_count']
                total_area = bc_info['element_count'] * dx_m * dy_m * 1e6  # Convert to mm²
                
                print(f"    {bc_info['comments']}: Area={total_area:.1f}mm², Heat={bc_info['heat_total']:.3f}W, Avg_T={avg_temp:.2f}°C")
                convection_heat_transfer += bc_info['heat_total']
            
            print(f"  Total User-Defined Convection Heat Transfer: {convection_heat_transfer:.2f} W")
            region_heat_transfer += convection_heat_transfer
        
        # Process default convection for surface elements without specific boundary conditions
        default_convection_heat_tmp = 0.0
        surface_element_count = 0
        
        # Get default convection parameters
        default_htc = env_data['heat_transfer_coefficient']
        default_T_inf = env_data['ambient_temperature']
        
        region_data = info['region_data']
        htc = region_data.get('heat_transfer_coefficient', htc_global)
        T_inf = region_data.get('ambient_temperature', T_inf_global)
        element_area = dx_m * dy_m  # Element area in m²
        
        # Process surface, edge, and corner elements that don't have specific BCs
        for element_type in ['surface', 'edge', 'corner']:
            if element_type in info['element_indices']:
                for global_idx, i, j, k in info['element_indices'][element_type]:
                    # Only process elements that don't have specific boundary conditions
                    if global_idx not in elements_with_bc:
                        # Only apply to top (k=Nz-1) and bottom (k=0) surfaces
                        if k == 0 or k == coords.Nz-1:
                            element_temp = u[global_idx]
                            # Default convection: Q = htc * A * (T - T_inf)
                            element_heat = -htc * element_area * (element_temp - T_inf)
                            default_convection_heat_tmp += element_heat
                            surface_element_count += 1
        
        if surface_element_count > 0:
            default_convection_heat = ratio_1 * default_convection_heat_tmp
            print(f"  Default Convection Heat Transfer: {default_convection_heat:.2f} W")
            print(f"    Surface Elements: {surface_element_count}, Area: {A_convection*1e6:.1f}mm², HTC: {default_htc:.1f}W/m²K")
            region_heat_transfer += default_convection_heat
        
        print(f"Region {region_id} Total Heat Transfer: {region_heat_transfer:.2f} W")
        global_heat_balance += region_heat_transfer
        
        # Detailed boundary condition temperatures
        print(f"\nBoundary Condition Temperatures for Region {region_id}:")
        if 'boundary_indices' in info:
            for bc_type, elements in info['boundary_indices'].items():
                if elements:  # Only process if there are elements
                    bc_temps = [u[elem['global_idx']] for elem in elements]
                    avg_temp = np.mean(bc_temps)
                    min_temp = np.min(bc_temps)
                    max_temp = np.max(bc_temps)
                    
                    print(f"  {bc_type} Boundary:")
                    print(f"    Elements: {len(elements)}")
                    print(f"    Average Temperature: {avg_temp:.2f}°C")
                    print(f"    Min Temperature: {min_temp:.2f}°C")
                    print(f"    Max Temperature: {max_temp:.2f}°C")
                    
                    # Add specific boundary condition parameters
                    if elements:
                        if bc_type == 'CONST_Q':
                            # Show all individual CONST_Q heat flux values
                            q_values = [elem['bc_data'].get('q', 0.0) for elem in elements]
                            if len(set(q_values)) == 1:
                                print(f"    Heat Flux: {q_values[0]:.2f} W/m² (all elements)")
                            else:
                                print(f"    Heat Flux: {min(q_values):.2f} to {max(q_values):.2f} W/m² (range)")
                                print(f"    Individual values: {[f'{q:.2f}' for q in q_values]}")
                        else:
                            # For other BC types, use the first element's bc_data
                            bc_data = elements[0]['bc_data']
                            if bc_type == 'CONST_T':
                                set_temp = bc_data.get('temperature', 0.0)
                                print(f"    Set Temperature: {set_temp:.2f}°C")
                            elif bc_type == 'PLASTIC_COVERED':
                                thickness = bc_data.get('plastic_thickness', 0.0)
                                conductivity = bc_data.get('plastic_conductivity', 0.0)
                                print(f"    Plastic Thickness: {thickness:.3f} mm")
                                print(f"    Plastic Conductivity: {conductivity:.3f} W/m·K")
                            elif bc_type == 'ACTUATOR_CONNECTED':
                                actuator_id = bc_data.get('actuator_id', 'Unknown')
                                location = bc_data.get('connecting_location', 'Unknown')
                                print(f"    Connected to Actuator: {actuator_id}")
                                print(f"    Connection Location: {location}")
                            elif bc_type == 'NODE_CONNECTED':
                                node_id = bc_data.get('node_id', 'Unknown')
                                resistance = bc_data.get('thermal_resistance', 0.0)
                                print(f"    Connected to Node: {node_id}")
                                print(f"    Thermal Resistance: {resistance:.2f} K/W")
                            elif bc_type == 'USERDEF_CONVECTION':
                                htc = bc_data.get('heat_transfer_coefficient', 0.0)
                                t_amb = bc_data.get('ambient_temperature', 0.0)
                                print(f"    Heat Transfer Coefficient: {htc:.2f} W/m²·K")
                                print(f"    Ambient Temperature: {t_amb:.2f}°C")
                            elif bc_type == 'USERDEF_CONDUCTION':
                                conductivity = bc_data.get('conductivity', 0.0)
                                print(f"    Conductivity: {conductivity:.2f} W/m·K")
    
    # Process node network temperatures
    if 'node_index_map' in solution_data:
        node_index_map = solution_data['node_index_map']
        node_networks = geo_data.get('node_networks', [])
        
        if node_networks:
            print(f"\n" + "=" * 50)
            print("\nNode Network Temperature Results:")
            
            for network in node_networks:
                network_id = network.get('id', 'Unknown')
                print(f"\nNetwork: {network_id}")
                print(f"Description: {network.get('description', 'No description')}")
                
                # Process each node in the network
                for node in network.get('nodes', []):
                    node_id = node['id']
                    if node_id in node_index_map:
                        node_idx = node_index_map[node_id]
                        node_temp = u[node_idx]
                        heat_source = node.get('heat_source', 0.0)
                        
                        print(f"  Node '{node_id}': {node_temp:.2f}°C")
                        if heat_source != 0:
                            print(f"    Heat Source: {heat_source:.2f} W")
                        if node.get('name'):
                            print(f"    Name: {node['name']}")
                    else:
                        print(f"  Node '{node_id}': Index not found in solution")
                
                # Show connections for context
                connections = network.get('connections', [])
                if connections:
                    print(f"  Network Connections ({len(connections)}):")
                    for conn in connections:
                        from_node = conn.get('from_node', 'Unknown')
                        to_node = conn.get('to_node', 'Unknown')
                        resistance = conn.get('thermal_resistance', 0.0)
                        print(f"    {from_node} → {to_node}: {resistance:.2f} K/W")

    # Process actuator temperatures and heat transfer
    for actuator_id, info in actuator_info.items():
        start_idx = info['start_idx']
        TFETs_idx = start_idx      # FETs temperature
        Tmotor_idx = start_idx + 1  # Motor internal temperature
        Tgearbox_idx = start_idx + 2 # Gearbox internal temperature
        num_unknowns = info['num_unknowns']
        actuator_temps = u[start_idx:start_idx + num_unknowns]
        housing_elements = actuator_elements[actuator_id]['housing_elements']
        output_elements = actuator_elements[actuator_id]['output_elements']
        
        avg_housing_temp = np.mean([u[idx] for idx in housing_elements]) if housing_elements else None
        avg_output_temp = np.mean([u[idx] for idx in output_elements]) if output_elements else None
        FETs_temp = u[TFETs_idx]
        Motor_temp = u[Tmotor_idx]
        Gearbox_temp = u[Tgearbox_idx]
        
        print(f"\nActuator: {actuator_id}")
        print(f"  FETs Temperature: {FETs_temp:.2f}°C (Heat: {info['heat_losses']['FETs']:.2f} W)")
        print(f"  Motor Temperature: {Motor_temp:.2f}°C (Heat: {info['heat_losses']['motor']:.2f} W)")
        print(f"  Gearbox Temperature: {Gearbox_temp:.2f}°C (Heat: {info['heat_losses']['gearbox']:.2f} W)")
        
        if avg_housing_temp is not None:
            print(f"  Housing Structure Temperature: {avg_housing_temp:.2f}°C")
        if avg_output_temp is not None:
            print(f"  Output Structure Temperature: {avg_output_temp:.2f}°C")
        
        # Calculate actuator heat losses
        heat_losses = info['heat_losses']
        total_heat = sum(heat_losses.values())
        print(f"  Total Actuator Heat Generation: {total_heat:.2f} W")
        
        # Show thermal resistances
        resistances = info['thermal_resistance']
        print(f"  Thermal Resistances: R1={resistances['R1']:.2f}, R2={resistances['R2']:.2f}, R3={resistances['R3']:.2f}, R4={resistances['R4']:.2f}, R5={resistances['R5']:.2f} K/W")
    
    # Print special unknowns (back hand and WY housing temperatures)
    if special_unknowns:
        T_WY_HS_idx = special_unknowns.get('T_WY_HS', None)
        T_BH_idx = special_unknowns.get('T_BH', None)
        if T_WY_HS_idx is not None:
            print(f"\nWrist Yaw (WY) Housing Temperature (T_WY_HS): {u[T_WY_HS_idx]:.2f}°C")
        if T_BH_idx is not None:
            print(f"Back Hand Temperature (T_BH): {u[T_BH_idx]:.2f}°C")
    
    # Print temperature distribution for the special boundary condition region (GEARBOX_HAND)
    special_bc_temps = []
    for region_id, info in region_info.items():
        if "GEARBOX_HAND" in info['boundary_indices']:
            gbh_elements = info['boundary_indices']["GEARBOX_HAND"]
            special_bc_temps.extend([u[elem['global_idx']] for elem in gbh_elements])
    if special_bc_temps:
        print(f"\nSpecial Boundary Condition (GEARBOX_HAND) Region:")
        print(f"  Average Temperature: {np.mean(special_bc_temps):.2f}°C")
        print(f"  Min Temperature: {np.min(special_bc_temps):.2f}°C")
        print(f"  Max Temperature: {np.max(special_bc_temps):.2f}°C")
    
    # Save results to file
    with open('sim_results.txt', 'w') as f:
        f.write("Arm Thermal Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Solve time: {solve_time:.2f} seconds\n")
        f.write(f"Final residual norm: {residual_norm:.2e}\n")
        if exitCode == 0:
            f.write("Solution converged successfully!\n")
        else:
            f.write(f"Warning: Solution did not converge, exit code: {exitCode}\n")
        
        # Write region results
        f.write("\nRegion Results:\n")
        f.write("-" * 50 + "\n")
        for region_id, info in region_info.items():
            start_idx = info['start_idx']
            num_elements = info['num_elements']
            region_temps = u[start_idx:start_idx + num_elements]
            f.write(f"\nRegion: {region_id}\n")
            f.write(f"Average Temperature: {np.mean(region_temps):.2f}°C\n")
            f.write(f"Min Temperature: {np.min(region_temps):.2f}°C\n")
            f.write(f"Max Temperature: {np.max(region_temps):.2f}°C\n")
            
            # Write heat transfer rates
            total_heat_transfer = 0.0
            if "PLASTIC_COVERED" in info['boundary_indices']:
                plastic_heat_transfer = 0.0
                for element in info['boundary_indices']["PLASTIC_COVERED"]:
                    global_idx = element['global_idx']
                    bc_data = element['bc_data']
                    plastic_thickness = bc_data['plastic_thickness'] * 1e-3
                    plastic_conductivity = bc_data.get('plastic_conductivity', 0.25)
                    R_contact = bc_data['contact_resistance'] # m^2C/W
                    htc = bc_data.get('heat_transfer_coefficient', htc_global)
                    T_inf = bc_data.get('ambient_temperature', T_inf_global)
                    h_eff = 1.0 / (1.0/htc + plastic_thickness/plastic_conductivity + R_contact)
                    element_temp = u[global_idx]
                    element_area = info['coords'].dx * info['coords'].dy * 1e-6
                    q_element = h_eff * element_area * (element_temp - T_inf)
                    plastic_heat_transfer += q_element
                f.write(f"Plastic Covered Surface Heat Transfer: {plastic_heat_transfer:.2f} W\n")
                total_heat_transfer += plastic_heat_transfer
            
            if "CONST_Q" in info['boundary_indices']:
                const_q_heat = sum(elem['bc_data'].get('heat_flux', 0.0) * 
                                 info['coords'].dx * info['coords'].dy * 1e-6 
                                 for elem in info['boundary_indices']["CONST_Q"])
                f.write(f"Constant Heat Flux Surface Heat Transfer: {const_q_heat:.2f} W\n")
                total_heat_transfer += const_q_heat
            
            if "ACTUATOR_CONNECTED" in info['boundary_indices']:
                actuator_heat_transfer = 0.0
                for element in info['boundary_indices']["ACTUATOR_CONNECTED"]:
                    global_idx = element['global_idx']
                    bc_data = element['bc_data']
                    actuator_id = bc_data.get('actuator_id')
                    if actuator_id in actuator_info:
                        actuator_temp = u[actuator_info[actuator_id]['start_idx']]
                        element_temp = u[global_idx]
                        # Get the appropriate thermal resistance based on connecting location
                        connecting_location = bc_data.get('connecting_location', 'housing')
                        thermal_resistance = actuator_info[actuator_id]['thermal_resistance'].get(connecting_location, 1.0)
                        element_area = info['coords'].dx * info['coords'].dy * 1e-6
                        q_element = (actuator_temp - element_temp) * element_area / thermal_resistance
                        actuator_heat_transfer += q_element
                f.write(f"Actuator Connected Surface Heat Transfer: {actuator_heat_transfer:.2f} W\n")
                total_heat_transfer += actuator_heat_transfer
            
            f.write(f"Total Heat Transfer Rate: {total_heat_transfer:.2f} W\n")
        
            # Write detailed boundary condition temperatures
            f.write(f"\nBoundary Condition Temperatures for Region {region_id}:\n")
            if 'boundary_indices' in info:
                for bc_type, elements in info['boundary_indices'].items():
                    if elements:  # Only process if there are elements
                        bc_temps = [u[elem['global_idx']] for elem in elements]
                        avg_temp = np.mean(bc_temps)
                        min_temp = np.min(bc_temps)
                        max_temp = np.max(bc_temps)
                        
                        f.write(f"  {bc_type} Boundary:\n")
                        f.write(f"    Elements: {len(elements)}\n")
                        f.write(f"    Average Temperature: {avg_temp:.2f}°C\n")
                        f.write(f"    Min Temperature: {min_temp:.2f}°C\n")
                        f.write(f"    Max Temperature: {max_temp:.2f}°C\n")
                        
                        # Add specific boundary condition parameters
                        if elements:
                            bc_data = elements[0]['bc_data']
                            if bc_type == 'CONST_Q':
                                q_flux = bc_data.get('q', 0.0)
                                f.write(f"    Heat Flux: {q_flux:.2f} W/m²\n")
                            elif bc_type == 'CONST_T':
                                set_temp = bc_data.get('temperature', 0.0)
                                f.write(f"    Set Temperature: {set_temp:.2f}°C\n")
                            elif bc_type == 'NODE_CONNECTED':
                                node_id = bc_data.get('node_id', 'Unknown')
                                resistance = bc_data.get('thermal_resistance', 0.0)
                                f.write(f"    Connected to Node: {node_id}\n")
                                f.write(f"    Thermal Resistance: {resistance:.2f} K/W\n")
        
        # Write node network results
        if 'node_index_map' in solution_data:
            node_index_map = solution_data['node_index_map']
            node_networks = geo_data.get('node_networks', [])
            
            if node_networks:
                f.write("\nNode Network Temperature Results:\n")
                f.write("-" * 50 + "\n")
                
                for network in node_networks:
                    network_id = network.get('id', 'Unknown')
                    f.write(f"\nNetwork: {network_id}\n")
                    f.write(f"Description: {network.get('description', 'No description')}\n")
                    
                    # Process each node in the network
                    for node in network.get('nodes', []):
                        node_id = node['id']
                        if node_id in node_index_map:
                            node_idx = node_index_map[node_id]
                            node_temp = u[node_idx]
                            heat_source = node.get('heat_source', 0.0)
                            
                            f.write(f"  Node '{node_id}': {node_temp:.2f}°C\n")
                            if heat_source != 0:
                                f.write(f"    Heat Source: {heat_source:.2f} W\n")
                            if node.get('name'):
                                f.write(f"    Name: {node['name']}\n")
                        else:
                            f.write(f"  Node '{node_id}': Index not found in solution\n")
                    
                    # Show connections for context
                    connections = network.get('connections', [])
                    if connections:
                        f.write(f"  Network Connections ({len(connections)}):\n")
                        for conn in connections:
                            from_node = conn.get('from_node', 'Unknown')
                            to_node = conn.get('to_node', 'Unknown')
                            resistance = conn.get('thermal_resistance', 0.0)
                            f.write(f"    {from_node} → {to_node}: {resistance:.2f} K/W\n")

        # Write actuator results  
        f.write("\nActuator Results:\n")
        f.write("-" * 50 + "\n")
        for actuator_id, info in actuator_info.items():
            start_idx = info['start_idx']
            TFETs_idx = start_idx      # FETs temperature
            Tmotor_idx = start_idx + 1  # Motor internal temperature
            Tgearbox_idx = start_idx + 2 # Gearbox internal temperature
            
            FETs_temp = u[TFETs_idx]
            Motor_temp = u[Tmotor_idx]
            Gearbox_temp = u[Tgearbox_idx]
            
            f.write(f"\nActuator: {actuator_id}\n")
            f.write(f"  FETs Temperature: {FETs_temp:.2f}°C (Heat: {info['heat_losses']['FETs']:.2f} W)\n")
            f.write(f"  Motor Temperature: {Motor_temp:.2f}°C (Heat: {info['heat_losses']['motor']:.2f} W)\n")
            f.write(f"  Gearbox Temperature: {Gearbox_temp:.2f}°C (Heat: {info['heat_losses']['gearbox']:.2f} W)\n")
            
            # Calculate actuator heat losses
            heat_losses = info['heat_losses']
            total_heat = sum(heat_losses.values())
            f.write(f"  Total Actuator Heat Generation: {total_heat:.2f} W\n")
            
            # Show thermal resistances
            resistances = info['thermal_resistance']
            f.write(f"  Thermal Resistances: R1={resistances['R1']:.2f}, R2={resistances['R2']:.2f}, R3={resistances['R3']:.2f}, R4={resistances['R4']:.2f}, R5={resistances['R5']:.2f} K/W\n")
            
            # Add connected element temperatures if available
            if actuator_id in actuator_elements:
                housing_elements = actuator_elements[actuator_id]['housing_elements']
                output_elements = actuator_elements[actuator_id]['output_elements']
                
                if housing_elements:
                    avg_housing_temp = np.mean([u[idx] for idx in housing_elements])
                    f.write(f"  Housing Structure Temperature: {avg_housing_temp:.2f}°C\n")
                if output_elements:
                    avg_output_temp = np.mean([u[idx] for idx in output_elements])
                    f.write(f"  Output Structure Temperature: {avg_output_temp:.2f}°C\n")
    
    print("\nResults have been saved to 'sim_results.txt'")
    
    # Plot temperature distributions for both bottom and top surfaces
    num_regions = len(region_info)
    
    # Only create plots if there are regions to plot
    if num_regions == 0:
        print("\nNo regions found - skipping temperature distribution plots")
        return
    
    # Create subplots: 2 rows per region (bottom and top surface), up to 3 regions per row
    num_cols = min(3, num_regions)
    num_rows = 2 * ((num_regions + num_cols - 1) // num_cols)  # 2 rows per region row
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 4*num_rows))
    
    # Handle single region case
    if num_regions == 1:
        if num_rows == 2:
            axes = axes.reshape(2, 1)
        else:
            axes = np.array([[axes]])
    
    # Collect all temperatures for consistent color scaling
    all_temps = []
    for region_id, info in region_info.items():
        start_idx = info['start_idx']
        coords = info['coords']
        # Bottom surface (z=0)
        bottom_temps = u[start_idx:start_idx + coords.Nx*coords.Ny].reshape(coords.Ny, coords.Nx)
        all_temps.extend(bottom_temps.flatten())
        # Top surface (z=max)
        if coords.Nz > 1:
            top_start_idx = start_idx + (coords.Nz - 1) * coords.Nx * coords.Ny
            top_temps = u[top_start_idx:top_start_idx + coords.Nx*coords.Ny].reshape(coords.Ny, coords.Nx)
            all_temps.extend(top_temps.flatten())
    
    vmin, vmax = np.min(all_temps), np.max(all_temps)
    
    # Plot each region's bottom and top surfaces
    for idx, (region_id, info) in enumerate(region_info.items()):
        coords = info['coords']
        start_idx = info['start_idx']
        col = idx % num_cols
        base_row = 2 * (idx // num_cols)
        
        # Bottom surface (z=0)
        bottom_temps = u[start_idx:start_idx + coords.Nx*coords.Ny].reshape(coords.Ny, coords.Nx)
        im_bottom = axes[base_row, col].imshow(bottom_temps, cmap='jet', interpolation='nearest', 
                                            origin='lower', vmin=vmin, vmax=vmax)
        axes[base_row, col].set_title(f'{region_id} - Bottom Surface (z=0)')
        plt.colorbar(im_bottom, ax=axes[base_row, col], label='Temperature (°C)')
        
        # Top surface (z=max)
        if coords.Nz > 1:
            top_start_idx = start_idx + (coords.Nz - 1) * coords.Nx * coords.Ny
            top_temps = u[top_start_idx:top_start_idx + coords.Nx*coords.Ny].reshape(coords.Ny, coords.Nx)
            im_top = axes[base_row + 1, col].imshow(top_temps, cmap='jet', interpolation='nearest', 
                                                  origin='lower', vmin=vmin, vmax=vmax)
            axes[base_row + 1, col].set_title(f'{region_id} - Top Surface (z=max)')
            plt.colorbar(im_top, ax=axes[base_row + 1, col], label='Temperature (°C)')
        else:
            # If only one layer, show same data but indicate it's the only layer
            im_top = axes[base_row + 1, col].imshow(bottom_temps, cmap='jet', interpolation='nearest', 
                                                  origin='lower', vmin=vmin, vmax=vmax)
            axes[base_row + 1, col].set_title(f'{region_id} - Single Layer')
            plt.colorbar(im_top, ax=axes[base_row + 1, col], label='Temperature (°C)')
    
    # Remove unused subplots
    total_used = num_regions * 2
    total_subplots = num_rows * num_cols
    for idx in range(total_used, total_subplots):
        row = idx // num_cols
        col = idx % num_cols
        fig.delaxes(axes[row, col])
    
    plt.suptitle('Temperature Distribution - Bottom and Top Surfaces', fontsize=16)
    plt.tight_layout()
    plt.savefig('temperature_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Final Energy Balance Check
    print("\n" + "=" * 60)
    print("GLOBAL ENERGY BALANCE CHECK")
    print("=" * 60)
    print(f"Total Heat Transfer across all regions: {global_heat_balance:.6f} W")
    
    # Check if energy balance is satisfied (should be close to zero)
    tolerance = 1e-3  # 1 mW tolerance
    if abs(global_heat_balance) < tolerance:
        print(f"✅ ENERGY BALANCE SATISFIED: |{global_heat_balance:.6f}| < {tolerance} W")
    else:
        print(f"❌ ENERGY BALANCE NOT SATISFIED: |{global_heat_balance:.6f}| ≥ {tolerance} W")
        print("   Check boundary conditions and heat sources/sinks!")
    
    print("=" * 60)

def calculate_conduction_heat_transfer(region_info, u, dx_m, dy_m, dz_m):
    """
    Calculate conduction heat transfer within a region using the same physics as matrix assembly.
    
    Args:
        region_info: Region information with coordinates and boundary conditions
        u: Solution vector with temperatures
        dx_m, dy_m, dz_m: Element dimensions in meters
        
    Returns:
        Total internal conduction heat transfer (W)
    """
    coords = region_info['coords']
    thermal_conductivity = region_info['region_data']['thermal_conductivity']
    start_idx = region_info['start_idx']
    
    Nx, Ny, Nz = coords.Nx, coords.Ny, coords.Nz
    total_conduction_heat = 0.0
    
    # Get adiabatic pairs (where conduction is blocked)
    adiabatic_pairs = region_info.get('adiabatic_pairs', [])
    adiabatic_blocked = set()
    for pair in adiabatic_pairs:
        idx1, idx2 = pair['element1']['global_idx'], pair['element2']['global_idx']
        adiabatic_blocked.add((min(idx1, idx2), max(idx1, idx2)))
    
    # Get USERDEF_CONDUCTION elements (modified conductivity)
    userdef_conductivity = {}
    if "USERDEF_CONDUCTION" in region_info['boundary_indices']:
        for element in region_info['boundary_indices']["USERDEF_CONDUCTION"]:
            global_idx = element['global_idx']
            new_k = element['bc_data']['conductivity']
            userdef_conductivity[global_idx] = new_k
    
    # Calculate conduction heat transfer between neighboring elements
    # Same physics as matrix assembly: Q = k * A * (T2 - T1) / distance
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                local_idx = coords.get_global_index(Nx, Ny, i, j, k)
                global_idx = start_idx + local_idx
                T1 = u[global_idx]
                
                # Get thermal conductivity for this element
                k1 = userdef_conductivity.get(global_idx, thermal_conductivity)
                
                # Check x-direction neighbor (i+1)
                if i < Nx - 1:
                    neighbor_local_idx = coords.get_global_index(Nx, Ny, i+1, j, k)
                    neighbor_global_idx = start_idx + neighbor_local_idx
                    
                    # Check if this connection is blocked by adiabatic BC
                    pair_key = (min(global_idx, neighbor_global_idx), max(global_idx, neighbor_global_idx))
                    if pair_key not in adiabatic_blocked:
                        T2 = u[neighbor_global_idx]
                        k2 = userdef_conductivity.get(neighbor_global_idx, thermal_conductivity)
                        k_avg = (k1 + k2) / 2.0  # Average conductivity at interface
                        
                        # Heat transfer: Q = k * A * (T2 - T1) / dx
                        area = dy_m * dz_m  # Cross-sectional area
                        q_x = k_avg * area * (T2 - T1) / dx_m
                        total_conduction_heat += abs(q_x)
                
                # Check y-direction neighbor (j+1)
                if j < Ny - 1:
                    neighbor_local_idx = coords.get_global_index(Nx, Ny, i, j+1, k)
                    neighbor_global_idx = start_idx + neighbor_local_idx
                    
                    # Check if this connection is blocked by adiabatic BC
                    pair_key = (min(global_idx, neighbor_global_idx), max(global_idx, neighbor_global_idx))
                    if pair_key not in adiabatic_blocked:
                        T2 = u[neighbor_global_idx]
                        k2 = userdef_conductivity.get(neighbor_global_idx, thermal_conductivity)
                        k_avg = (k1 + k2) / 2.0  # Average conductivity at interface
                        
                        # Heat transfer: Q = k * A * (T2 - T1) / dy
                        area = dx_m * dz_m  # Cross-sectional area
                        q_y = k_avg * area * (T2 - T1) / dy_m
                        total_conduction_heat += abs(q_y)
                
                # Check z-direction neighbor (k+1)
                if k < Nz - 1:
                    neighbor_local_idx = coords.get_global_index(Nx, Ny, i, j, k+1)
                    neighbor_global_idx = start_idx + neighbor_local_idx
                    
                    # Check if this connection is blocked by adiabatic BC
                    pair_key = (min(global_idx, neighbor_global_idx), max(global_idx, neighbor_global_idx))
                    if pair_key not in adiabatic_blocked:
                        T2 = u[neighbor_global_idx]
                        k2 = userdef_conductivity.get(neighbor_global_idx, thermal_conductivity)
                        k_avg = (k1 + k2) / 2.0  # Average conductivity at interface
                        
                        # Heat transfer: Q = k * A * (T2 - T1) / dz
                        area = dx_m * dy_m  # Cross-sectional area
                        q_z = k_avg * area * (T2 - T1) / dz_m
                        total_conduction_heat += abs(q_z)
    
    return total_conduction_heat

def update_matrix_with_boundary_conditions(A: scipy.sparse.lil_matrix, b: np.ndarray, region_info: Dict, actuator_info: Dict, node_index_map: dict, geo_data: dict) -> Tuple[scipy.sparse.lil_matrix, np.ndarray, Dict, Dict]:
    """
    Update matrix A and vector b based on boundary conditions using pre-calculated indices.
    Different actuator types (PITCHYAW, ROLL) are handled differently.
    Args:
        A: Global system matrix
        b: Global system vector
        region_info: Dictionary with region information
        actuator_info: Dictionary with actuator information
        special_unknowns: Dictionary with indices for special unknowns (T_WY_HS, T_BH)
    Returns:
        Tuple containing:
        - Updated global system matrix
        - Updated global system vector
        - Dictionary containing actuator elements mapping
    """
    print("\n" + "=" * 50)
    print("\nUpdating matrix with boundary conditions...")
    actuator_elements = {}
    
    env_data = geo_data['environment']
    T_inf_global = env_data['ambient_temperature']
    htc_global = env_data['heat_transfer_coefficient']
    
    # Collect element indices connected to each node via NODE_CONNECTED boundary conditions
    node_connected_elements = {}
    for region_id, info in region_info.items():
        if 'boundary_indices' in info and 'NODE_CONNECTED' in info['boundary_indices']:
            N_elements = len(info['boundary_indices']['NODE_CONNECTED'])
            for element in info['boundary_indices']['NODE_CONNECTED']:
                bc_data = element['bc_data']
                node_id = bc_data['node_id']
                if node_id:
                    if node_id not in node_connected_elements:
                        node_connected_elements[node_id] = {
                            'elements': [],
                            'regions': []
                        }
                    node_connected_elements[node_id]['elements'].append(element['global_idx'])
                    if region_id not in node_connected_elements[node_id]['regions']:
                        node_connected_elements[node_id]['regions'].append(region_id)

    # First pass: Collect all boundary elements for each actuator
    for actuator_id, act_info in actuator_info.items():
        actuator_elements[actuator_id] = {
            'housing_elements': [],
            'output_elements': [],
            'housing_area': 0.0,
            'output_area': 0.0
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
                            connecting_loc = bc_data['connecting_location']
                            if connecting_loc == 'Housing':
                                actuator_elements[actuator_id]['housing_elements'].append(global_idx)
                                actuator_elements[actuator_id]['housing_area'] = bc_data['width'] * bc_data['height'] * 1e-6
                            elif connecting_loc == 'Output':
                                actuator_elements[actuator_id]['output_elements'].append(global_idx)
                                actuator_elements[actuator_id]['output_area'] = bc_data['width'] * bc_data['height'] * 1e-6
                        
                    
                                
        act_start_idx = act_info['start_idx']
                    
        # Thermal Resistance in the actuator
        R1 = act_info['thermal_resistance']['R1']
        R2 = act_info['thermal_resistance']['R2']
        R3 = act_info['thermal_resistance']['R3']
        R4 = act_info['thermal_resistance']['R4']
        R5 = act_info['thermal_resistance']['R5']
        
        # Get actuator-specific elements
        housing_elements = actuator_elements[actuator_id]['housing_elements']
        output_elements = actuator_elements[actuator_id]['output_elements']
        
        # Get heat generation and thermal resistances
        Q_fets = act_info['heat_losses']['FETs']
        Q_motor = act_info['heat_losses']['motor']
        Q_gearbox = act_info['heat_losses']['gearbox']
        
        # Both types have 2 unknowns
        TFETs_idx = act_start_idx      # FETs temperature
        Tmotor_idx = act_start_idx + 1  # Motor internal temperature
        Tgearbox_idx = act_start_idx + 2 # Gearbox internal temperature
        
        # Fets
        # Equation: (THousing - TFETs)/R1 + Q_FETs = 0  
        A[TFETs_idx, TFETs_idx] -= 1/R1
        b[TFETs_idx] -= Q_fets
        # Motor
        # Equation: (THousing - TMotor)/R2 + Q_Motor = 0
        A[Tmotor_idx, Tmotor_idx] -= 1/R2
        b[Tmotor_idx] -= Q_motor
        # Gearbox
        # Equation: (THousing - TGearbox)/R4 + (TOutput - TGearbox)/R5 = 0
        A[Tgearbox_idx, Tgearbox_idx] -= 1/R4+1/R5
        b[Tgearbox_idx] -= Q_gearbox
        
        for element_idx in housing_elements:
            # Thousing = sum(Thousing_elem) / len(housing_elements), average of the housing elements
            A[TFETs_idx, element_idx] += 1/R1 /len(housing_elements)
            A[Tmotor_idx, element_idx] += 1/R2 /len(housing_elements)
            A[Tgearbox_idx, element_idx] += 1/R4 /len(housing_elements)
        for element_idx in output_elements:
            # Thousing = sum(TOutput) / len(output_elements), average of the output elements
            A[Tgearbox_idx, element_idx] += 1/R5 /len(output_elements)
    
    # Node Network Internal Physics: Process each node and its connections
    # Logic: For each node, find all its connections and apply heat generation
    
    for node_network in geo_data['node_networks']:
        # Process each node: apply heat generation and find connections
        for node in node_network['nodes']:
            node_id = node['id']
            node_idx = node_index_map[node_id]
                
            # Get node properties
            heat_source = node['heat_source']  # W
            # Apply heat generation Q to the node
            b[node_idx] -= heat_source
            
        # Find all connections for this node and update matrix
        for connection in node_network['connections']:
            from_node_id = connection['from_node']
            to_node_id = connection['to_node']
            thermal_resistance = connection['thermal_resistance']
            
            # Handle air connections
            if to_node_id == 'air' or from_node_id == 'air':
                if to_node_id == 'air':
                    # Connection to ambient air: (T_node - T_amb) / R
                    from_node_idx = node_index_map[from_node_id]
                    A[from_node_idx, from_node_idx] -= 1 / thermal_resistance
                    b[from_node_idx] -= T_inf_global / thermal_resistance
                else:
                    to_node_idx = node_index_map[to_node_id]
                    A[to_node_idx, to_node_idx] -= 1 / thermal_resistance
                    b[to_node_idx] -= T_inf_global / thermal_resistance  
            elif "region" not in from_node_id and "region" not in to_node_id:
                from_node_idx = node_index_map[from_node_id]
                to_node_idx = node_index_map[to_node_id]
                # Connection between two nodes: (T_node - T_connected) / R
                A[from_node_idx, to_node_idx] += 1 / thermal_resistance
                A[to_node_idx, from_node_idx] += 1 / thermal_resistance
                A[from_node_idx, from_node_idx] -= 1 / thermal_resistance
                A[to_node_idx, to_node_idx] -= 1 / thermal_resistance

    # Process each region
    for current_region_id, info in region_info.items():
        print(f"\nProcessing region: {current_region_id}")
        coords = info['coords']
        
        # Get region dimensions
        Nx, Ny, Nz = coords.Nx, coords.Ny, coords.Nz
        dx, dy, dz = coords.dx, coords.dy, coords.dz
        dx_m, dy_m, dz_m = dx * 1e-3, dy * 1e-3, dz * 1e-3
        
        # Track elements that already have boundary conditions applied
        elements_with_bc_to_skip = set()
        
        if info.get('adiabatic_pairs'):
            for element_pair in info['adiabatic_pairs']:
                # Global indices of the neighboring elements that are separated by an adiabatic line
                idx1 = element_pair['element1']['global_idx']
                idx2 = element_pair['element2']['global_idx']

                # Read the CURRENT assembled coupling coefficients. These already
                # include any USERDEF_CONDUCTION adjustments that changed the
                # effective conductance between neighbors. By using the current
                # matrix values, we remain consistent with heterogeneous thermal
                # conductivities.
                k12 = A[idx1, idx2]
                k21 = A[idx2, idx1]
                k_tmp1 = A[idx1, idx1]
                k_tmp2 = A[idx2, idx2]

                if k12 != 0:
                    # Off-diagonals are typically negative conductances. Remove the
                    # coupling and add its magnitude to the corresponding diagonal to
                    # preserve energy balance.
                    A[idx1, idx2] = 0.0
                    A[idx1, idx1] += k12

                if k21 != 0:
                    A[idx2, idx1] = 0.0
                    A[idx2, idx2] += k21
        
        
        
        # Process other boundary conditions
        for bc_type, elements in info['boundary_indices'].items():
            
            if bc_type == "PLASTIC_COVERED":
                for element in elements:
                    global_idx = element['global_idx']
                    elements_with_bc_to_skip.add(global_idx)
                    bc_data = element['bc_data']
                    
                    # Get boundary condition parameters
                    plastic_thickness = bc_data['plastic_thickness'] * 1e-3  # Convert to meters
                    plastic_conductivity = bc_data.get('plastic_conductivity', 0.25)  # W/mK
                    htc = bc_data.get('heat_transfer_coefficient', htc_global)  # W/m²K
                    T_inf = bc_data.get('ambient_temperature', T_inf_global)  # C
                    R_contact = bc_data['contact_resistance'] # m^2C/W
                    
                    # Calculate effective heat transfer coefficient
                    h_eff = 1.0 / (1.0/htc + plastic_thickness/plastic_conductivity + R_contact)
                    
                    # Update matrix and vector
                    A[global_idx, global_idx] -= h_eff / dz_m
                    b[global_idx] -= h_eff * T_inf / dz_m
            
            elif bc_type == "CONST_Q":
                for element in elements:
                    global_idx = element['global_idx']
                    # elements_with_bc.add(global_idx)
                    bc_data = element['bc_data']
                    
                    # Get heat transfer rate and calculate heat flux
                    Q = bc_data.get('q', 0.0)  # W (total heat transfer rate)
                    width = bc_data.get('width', 0.0)  # mm
                    height = bc_data.get('height', 0.0)  # mm
                    area = (width * height) * 1e-6  # Convert to m^2
                    q = Q / area  # W/m^2 (heat flux)
                    
                    # Update vector with heat flux
                    b[global_idx] -= q / dz_m
                    
            elif bc_type == "USERDEF_CONVECTION":
                for element in elements:
                    global_idx = element['global_idx']
                    userdef_htc = elements[0]['bc_data']['heat_transfer_coefficient']
                    userdef_T_inf = elements[0]['bc_data']['ambient_temperature']
                    
                    A[global_idx, global_idx] -= userdef_htc / dz_m
                    b[global_idx] -= userdef_htc  * userdef_T_inf / dz_m
                
            elif bc_type == "CONST_T":
                for element in elements:
                    global_idx = element['global_idx']
                    elements_with_bc_to_skip.add(global_idx)
                    T_tmp = elements[0]['bc_data']['temperature']
                    
                    A[global_idx, :] = 0
                    A[global_idx, global_idx] = 1
                    b[global_idx] = T_tmp
                    
            elif bc_type == "ACTUATOR_CONNECTED":
                for element in elements:
                    global_idx = element['global_idx']
                    elements_with_bc_to_skip.add(global_idx)
                    bc_data = element['bc_data']
                    
                    # Get actuator info
                    actuator_id = bc_data.get('actuator_id')
                    connecting_location = bc_data.get('connecting_location')
                    if actuator_id not in actuator_info:
                        continue
                        
                    act_info = actuator_info[actuator_id]
                    act_start_idx = act_info['start_idx']
                    
                    # Thermal Resistance in the actuator
                    R1 = act_info['thermal_resistance']['R1']
                    R2 = act_info['thermal_resistance']['R2']
                    R3 = act_info['thermal_resistance']['R3']
                    R4 = act_info['thermal_resistance']['R4']
                    R5 = act_info['thermal_resistance']['R5']
                    
                    # Get actuator-specific elements
                    housing_elements = actuator_elements[actuator_id]['housing_elements']
                    output_elements = actuator_elements[actuator_id]['output_elements']
                    
                    # Get heat generation and thermal resistances
                    Q_fets = act_info['heat_losses']['FETs']
                    Q_motor = act_info['heat_losses']['motor']
                    Q_gearbox = act_info['heat_losses']['gearbox']
                    
                    # Both types have 2 unknowns
                    TFETs_idx = act_start_idx      # FETs temperature
                    Tmotor_idx = act_start_idx + 1  # Motor internal temperature
                    Tgearbox_idx = act_start_idx + 2 # Gearbox internal temperature
                    
                    if connecting_location == "Housing":
                        # Loop through elements in boundary region to create coupling between all elements
                        for element_idx in housing_elements:
                            # Thousing = sum(Thousing_elem) / len(housing_elements), average of the housing elements
                            # Equation: (TFETs - THousing)/R1 + (TMotor - THousing)/R2 + (TOutput - THousing)/R3 + (TGearbox - THousing)/R4 + (TNeighbors - THousing)/Rcond + (Tamb - THousing)/Rhtc = 0
                            A[global_idx, element_idx] -= (1/R1+1/R2+1/R3+1/R4) /len(housing_elements)/dz_m / actuator_elements[actuator_id]['housing_area']
                        for element_idx in output_elements:
                            # Equation: (THousing - TOutput)/R3 + Q_Output = 0
                            A[global_idx, element_idx] += (1/R3) /len(output_elements)/dz_m / actuator_elements[actuator_id]['housing_area']
                        A[global_idx, TFETs_idx] += 1/R1/dz_m / actuator_elements[actuator_id]['housing_area']
                        A[global_idx, Tmotor_idx] += 1/R2/dz_m / actuator_elements[actuator_id]['housing_area']
                        A[global_idx, Tgearbox_idx] += 1/R4/dz_m / actuator_elements[actuator_id]['housing_area']
                        
                    elif connecting_location == "Output":
                        for element_idx in output_elements:
                            A[global_idx, element_idx] -= (1/R3+1/R5) /len(output_elements)/dz_m / actuator_elements[actuator_id]['output_area']
                        for element_idx in housing_elements:
                            A[global_idx, element_idx] += 1/R3 /len(housing_elements)/dz_m/ actuator_elements[actuator_id]['output_area']
                        A[global_idx, Tgearbox_idx] += 1/R5 /dz_m/ actuator_elements[actuator_id]['output_area']

            elif bc_type == "NODE_CONNECTED":
                for element in elements:
                    global_idx = element['global_idx']
                    elements_with_bc_to_skip.add(global_idx)
                    bc_data = element['bc_data']
                    node_id = bc_data['node_id']
                    thermal_resistance = bc_data['thermal_resistance']
                    area = bc_data['width'] * bc_data['height'] * 1e-6
                    
                    if node_id in node_index_map:
                        node_global_idx = node_index_map[node_id]
                        # Add symmetric conductance G = 1/R
                        for element_idx in node_connected_elements[node_id]['elements']:
                            A[global_idx, element_idx] -= 1.0 / thermal_resistance / len(elements) / dz_m / area
                        A[global_idx, node_global_idx] += 1.0 / thermal_resistance / dz_m / area
                        A[node_global_idx, global_idx] += 1.0 / thermal_resistance / len(elements)
                A[node_global_idx, node_global_idx] -= 1.0 / thermal_resistance
            
            
        # Get convection parameters from region data
        region_data = info['region_data']
        htc = region_data.get('heat_transfer_coefficient', htc_global)  # W/m²K (default if not specified)
        T_inf = region_data.get('ambient_temperature',T_inf_global)  # °C
                
               
        # Process all surface, edge, and corner elements that don't have other BCs
        for element_type in ['surface', 'edge', 'corner']:
            for global_idx, i, j, k in info['element_indices'][element_type]:
                if global_idx not in elements_with_bc_to_skip:
                    # Only apply convection to elements on top (k=Nz-1) and bottom (k=0) surfaces
                    if k == 0 or k == Nz-1:
                        # Apply convection boundary condition
                        # Check if this element has USERDEF_CONDUCTION boundary condition
                        A[global_idx, global_idx] -= htc / dz_m
                        b[global_idx] -= htc * T_inf / dz_m


    return A, b, actuator_elements, node_connected_elements 