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

def read_geometry(file_path: str = "GEO.json") -> Tuple[dict, ThermalParameters, List[dict], List[dict]]:
    """
    Read and parse the geometry file.
    """
    with open(file_path, "r") as f:
        geo_data = json5.load(f)
    params = get_parameters(geo_data)
    regions = geo_data["regions"]
    actuators = geo_data["actuators"]
    return geo_data, params, regions, actuators

def initialize_global_matrix(total_elements: int, region_info: Dict, actuator_info: Dict) -> Tuple[scipy.sparse.lil_matrix, np.ndarray, dict]:
    """
    Initialize the global system matrix and vector, including special unknowns for GEARBOX_HAND.
    """
    print("\nInitializing global matrix and vector...")
    total_actuator_unknowns = 0
    for info in actuator_info.values():
        total_actuator_unknowns += 2
    # Add 2 for GEARBOX_HAND unknowns
    total_special_unknowns = 2
    total_size = total_elements + total_actuator_unknowns + total_special_unknowns
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
        region_matrix = create_layer_matrix(coords.Nx, coords.Ny, coords.Nz,
                                          coords.dx*1e-3, coords.dy*1e-3, coords.dz*1e-3,
                                          thermal_conductivity)
        end_idx = start_idx + num_elements
        A[start_idx:end_idx, start_idx:end_idx] = region_matrix
        print(f"Added block from index {start_idx} to {end_idx}")
    current_actuator_idx = total_elements
    for actuator_id, info in actuator_info.items():
        print(f"\nInitializing matrix block for actuator: {actuator_id}")
        num_unknowns = 2
        info['start_idx'] = current_actuator_idx
        info['num_unknowns'] = num_unknowns
        end_idx = current_actuator_idx + num_unknowns
        actuator_block = np.zeros((num_unknowns, num_unknowns))
        A[current_actuator_idx:end_idx, current_actuator_idx:end_idx] = actuator_block
        print(f"Added {info['type']} actuator block from index {current_actuator_idx} to {end_idx} (size: {num_unknowns}x{num_unknowns})")
        current_actuator_idx = end_idx
    # Add special unknowns for GEARBOX_HAND
    special_unknowns = {}
    special_unknowns['T_WY_OUTPUT'] = total_elements + total_actuator_unknowns
    special_unknowns['T_BH'] = total_elements + total_actuator_unknowns + 1
    print(f"Added special unknowns: T_WY_OUTPUT at {special_unknowns['T_WY_OUTPUT']}, T_BH at {special_unknowns['T_BH']}")
    return A, b, special_unknowns

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
    geo_data, params, regions, actuators = read_geometry()
    region_info, actuator_info, total_elements, actuator_start_idx = calculate_system_size(
        regions, actuators, params
    )
    visualize_regions(regions, region_info, params)
    A, b, special_unknowns = initialize_global_matrix(total_elements, region_info, actuator_info)
    A = update_matrix_with_geometries(A, region_info, actuator_info)
    A, b, actuator_elements = update_matrix_with_boundary_conditions(A, b, region_info, actuator_info, special_unknowns)
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
        special_unknowns=special_unknowns
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
    u0 = np.ones_like(b) * 30.0
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
    # Print temperatures for each region and actuator
    print("\nTemperature Results:")
    print("-" * 50)
    for region_id, info in region_info.items():
        start_idx = info['start_idx']
        num_elements = info['num_elements']
        region_temps = u[start_idx:start_idx + num_elements]
        print(f"\nRegion: {region_id}")
        print(f"Average Temperature: {np.mean(region_temps):.2f}°C")
        print(f"Min Temperature: {np.min(region_temps):.2f}°C")
        print(f"Max Temperature: {np.max(region_temps):.2f}°C")
        if "CONST_Q" in info['boundary_indices']:
            const_q_elements = info['boundary_indices']["CONST_Q"]
            const_q_temps = [u[elem['global_idx']] for elem in const_q_elements]
            print(f"Average CONST_Q Temperature: {np.mean(const_q_temps):.2f}°C")
            print(f"Min CONST_Q Temperature: {np.min(const_q_temps):.2f}°C")
            print(f"Max CONST_Q Temperature: {np.max(const_q_temps):.2f}°C")
    for actuator_id, info in actuator_info.items():
        start_idx = info['start_idx']
        num_unknowns = info['num_unknowns']
        actuator_temps = u[start_idx:start_idx + num_unknowns]
        housing_elements = actuator_elements[actuator_id]['housing_elements']
        gearbox_elements = actuator_elements[actuator_id]['gearbox_elements']
        motor_elements = actuator_elements[actuator_id]['motor_elements']
        avg_housing_temp = np.mean([u[idx] for idx in housing_elements]) if housing_elements else None
        avg_gearbox_temp = np.mean([u[idx] for idx in gearbox_elements]) if gearbox_elements else None
        avg_motor_temp = np.mean([u[idx] for idx in motor_elements]) if motor_elements else None
        print(f"\nActuator: {actuator_id} ({info['type']})")
        print(f"Gearbox Temperature (T2): {actuator_temps[0]:.2f}°C")
        print(f"Motor Temperature (T4): {actuator_temps[1]:.2f}°C")
        if avg_housing_temp is not None:
            print(f"Average Housing Structure Temperature: {avg_housing_temp:.2f}°C")
        if avg_gearbox_temp is not None:
            print(f"Average Gearbox Side Structure Temperature: {avg_gearbox_temp:.2f}°C")
        if avg_motor_temp is not None:
            print(f"Average Motor Side Structure Temperature: {avg_motor_temp:.2f}°C")
    # Print special unknowns (back hand and WY housing temperatures)
    if special_unknowns:
        T_WY_OUTPUT_idx = special_unknowns.get('T_WY_OUTPUT', None)
        T_BH_idx = special_unknowns.get('T_BH', None)
        if T_WY_OUTPUT_idx is not None:
            print(f"\nWrist Yaw (WY) Housing Temperature (T_WY_OUTPUT): {u[T_WY_OUTPUT_idx]:.2f}°C")
        if T_BH_idx is not None:
            print(f"Back Hand Temperature (T_BH): {u[T_BH_idx]:.2f}°C")
    # Print temperature distribution for the special boundary condition regions
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
    # Print temperature distribution for the wrist region with special boundary conditions
    wrist_special_bc_temps = []
    for region_id, info in region_info.items():
        if region_id == "wrist":
            # Check for both types of special boundary conditions
            for bc_type in ["GEARBOX_HAND", "HOUSING_HAND"]:
                if bc_type in info['boundary_indices']:
                    elements = info['boundary_indices'][bc_type]
                    temps = [u[elem['global_idx']] for elem in elements]
                    wrist_special_bc_temps.extend(temps)
                    print(f"\nWrist Region {bc_type} Elements:")
                    print(f"  Average Temperature: {np.mean(temps):.2f}°C")
                    print(f"  Min Temperature: {np.min(temps):.2f}°C")
                    print(f"  Max Temperature: {np.max(temps):.2f}°C")
                    print(f"  Number of Elements: {len(temps)}")
    # Save results to file (same as before)
    with open('arm_thermal_results.txt', 'w') as f:
        f.write("Arm Thermal Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Solve time: {solve_time:.2f} seconds\n")
        f.write(f"Final residual norm: {residual_norm:.2e}\n")
        if exitCode == 0:
            f.write("Solution converged successfully!\n")
        else:
            f.write(f"Warning: Solution did not converge, exit code: {exitCode}\n")
        f.write("\nRegion Temperatures:\n")
        f.write("-" * 50 + "\n")
        for region_id, info in region_info.items():
            start_idx = info['start_idx']
            num_elements = info['num_elements']
            region_temps = u[start_idx:start_idx + num_elements]
            f.write(f"\nRegion: {region_id}\n")
            f.write(f"Average Temperature: {np.mean(region_temps):.2f}°C\n")
            f.write(f"Min Temperature: {np.min(region_temps):.2f}°C\n")
            f.write(f"Max Temperature: {np.max(region_temps):.2f}°C\n")
            if "CONST_Q" in info['boundary_indices']:
                const_q_elements = info['boundary_indices']["CONST_Q"]
                const_q_temps = [u[elem['global_idx']] for elem in const_q_elements]
                f.write(f"Average CONST_Q Temperature: {np.mean(const_q_temps):.2f}°C\n")
                f.write(f"Min CONST_Q Temperature: {np.min(const_q_temps):.2f}°C\n")
                f.write(f"Max CONST_Q Temperature: {np.max(const_q_temps):.2f}°C\n")
        f.write("\nActuator Temperatures:\n")
        f.write("-" * 50 + "\n")
        for actuator_id, info in actuator_info.items():
            start_idx = info['start_idx']
            num_unknowns = info['num_unknowns']
            actuator_temps = u[start_idx:start_idx + num_unknowns]
            housing_elements = actuator_elements[actuator_id]['housing_elements']
            gearbox_elements = actuator_elements[actuator_id]['gearbox_elements']
            motor_elements = actuator_elements[actuator_id]['motor_elements']
            avg_housing_temp = np.mean([u[idx] for idx in housing_elements]) if housing_elements else None
            avg_gearbox_temp = np.mean([u[idx] for idx in gearbox_elements]) if gearbox_elements else None
            avg_motor_temp = np.mean([u[idx] for idx in motor_elements]) if motor_elements else None
            f.write(f"\nActuator: {actuator_id} ({info['type']})\n")
            f.write(f"Gearbox Temperature (T2): {actuator_temps[0]:.2f}°C\n")
            f.write(f"Motor Temperature (T4): {actuator_temps[1]:.2f}°C\n")
            if avg_housing_temp is not None:
                f.write(f"Average Housing Structure Temperature: {avg_housing_temp:.2f}°C\n")
            if avg_gearbox_temp is not None:
                f.write(f"Average Gearbox Structure Temperature: {avg_gearbox_temp:.2f}°C\n")
            if avg_motor_temp is not None:
                f.write(f"Average Motor Structure Temperature: {avg_motor_temp:.2f}°C\n")
    print("\nResults have been saved to 'arm_thermal_results.txt'")
    # Plot temperature distributions for each region (same as before)
    num_regions = len(region_info)
    num_cols = min(3, num_regions)
    num_rows = (num_regions + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 5*num_rows))
    if num_regions == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    all_temps = []
    for region_id, info in region_info.items():
        start_idx = info['start_idx']
        coords = info['coords']
        bottom_temps = u[start_idx:start_idx + coords.Nx*coords.Ny].reshape(coords.Ny, coords.Nx)
        all_temps.extend(bottom_temps.flatten())
    vmin, vmax = np.min(all_temps), np.max(all_temps)
    for idx, (region_id, info) in enumerate(region_info.items()):
        coords = info['coords']
        start_idx = info['start_idx']
        bottom_temps = u[start_idx:start_idx + coords.Nx*coords.Ny].reshape(coords.Ny, coords.Nx)
        im = axes[idx].imshow(bottom_temps, cmap='jet', interpolation='nearest', 
                            origin='lower', vmin=vmin, vmax=vmax)
        axes[idx].set_title(f'{region_id}\n(z=0)')
        plt.colorbar(im, ax=axes[idx], label='Temperature (°C)')
    for idx in range(num_regions, len(axes)):
        fig.delaxes(axes[idx])
    plt.suptitle('Temperature Distribution for All Regions', fontsize=16)
    plt.tight_layout()
    plt.savefig('temperature_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    print("\nUpdating matrix with geometry data...")
    
    # Process each region
    for region_id, info in region_info.items():
        print(f"\nProcessing region: {region_id}")
        coords = info['coords']
        thermal_conductivity = info['region_data']['thermal_conductivity']
        
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
            
            # Scale by thermal conductivity
            A[global_idx, :] *= thermal_conductivity
    
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

def update_matrix_with_boundary_conditions(A: scipy.sparse.lil_matrix, b: np.ndarray, region_info: Dict, actuator_info: Dict, special_unknowns: dict) -> Tuple[scipy.sparse.lil_matrix, np.ndarray, Dict]:
    """
    Update matrix A and vector b based on boundary conditions using pre-calculated indices.
    Different actuator types (PITCHYAW, ROLL) are handled differently.
    Args:
        A: Global system matrix
        b: Global system vector
        region_info: Dictionary with region information
        actuator_info: Dictionary with actuator information
        special_unknowns: Dictionary with indices for special unknowns (T_WY_OUTPUT, T_BH)
    Returns:
        Tuple containing:
        - Updated global system matrix
        - Updated global system vector
        - Dictionary containing actuator elements mapping
    """
    print("\nUpdating matrix with boundary conditions...")
    actuator_elements = {}
    
    R_contact = 2
    
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
        
        R1 = act_info['thermal_resistance']['R1']
        R2 = act_info['thermal_resistance']['R2']
        R3 = act_info['thermal_resistance']['R3']
        R4 = act_info['thermal_resistance']['R4']
        R5 = act_info['thermal_resistance']['R4']
        R6 = act_info['thermal_resistance']['R6']
        
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
        
        if actuator_type == "ROLL_IV":
            # Equation 1: (T2-T3)/(R1+R2) + (T2-T3)/R3 = Q_GEARBOX
            A[T2_idx, T2_idx] = 1/(R1+R2) + 1/R3
            for elem_idx in housing_elements:
                A[T2_idx, elem_idx] = -(1/(R1+R2) + 1/R3) / len(housing_elements)
            b[T2_idx] = Q_GEARBOX
        else:
            # Equation 1: (T2-T1)/R1 + (T2-T3)/R3 = Q_GEARBOX
            A[T2_idx, T2_idx] = 1/R1 + 1/R3
            for elem_idx in gearbox_elements:
                A[T2_idx, elem_idx] = -1/(R1 * len(gearbox_elements))
            for elem_idx in housing_elements:
                A[T2_idx, elem_idx] = -1/(R3 * len(housing_elements))
            b[T2_idx] = Q_GEARBOX
        
        # Equation 2: (T4-T3)/R4 = Q_MOTOR
        A[T4_idx, T4_idx] = 1/R4
        for elem_idx in housing_elements:
            A[T4_idx, elem_idx] = -1/(R4 * len(housing_elements))
        b[T4_idx] = Q_MOTOR
    
    
    # Process each region
    for current_region_id, info in region_info.items():
        print(f"\nProcessing region: {current_region_id}")
        coords = info['coords']
        thermal_conductivity = info['region_data']['thermal_conductivity']
        
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
                A[idx1, idx2] -= thermal_conductivity/dx_m**2
                A[idx1, idx1] += thermal_conductivity/dx_m**2
                A[idx2, idx2] += thermal_conductivity/dx_m**2
                A[idx2, idx1] -= thermal_conductivity/dx_m**2
        
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
                    coupling_factor = thermal_conductivity / (dx_m * dx_m)
                else:  # x-axis
                    coupling_factor = thermal_conductivity / (dy_m * dy_m)
                
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
                    htc = bc_data.get('heat_transfer_coefficient', 7)  # W/m²K
                    T_inf = bc_data.get('ambient_temperature', 21)  # °C
                    
                    # Calculate effective heat transfer coefficient
                    h_eff = 1.0 / (1.0/htc + plastic_thickness/plastic_conductivity)
                    
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
                    
            elif bc_type == "HOUSING_HAND":
                R1_tmp = 1.0+R_contact  # C/W
                
                R2_tmp = 1.9  # C/W
                
                # heatpiped design
                R2_tmp = 1.9/3
                
                # Only flipped Yaw
                R3_tmp = 2.898*0.65  # C/W
                R4_tmp = 7.7+5  # C/W
                
                # Flipped with glove on, elbow installed, both fan blowing, hand fan pos
                R3_tmp = 2.898 + 1.5 - 2.898*0.55  # C/W
                R4_tmp = 7.7 + 5 + 1.5  # C/W
                
                Q1 = 5.5       # W
                Q2 = 24.56-2.5
                region_data = info['region_data']
                T_amb = region_data.get('ambient_temperature', 21)
                htc = region_data.get('heat_transfer_coefficient', 7)  # W/m²K (default if not specified)
                N_element = len(elements)
                T_WY_OUTPUT_idx = special_unknowns['T_WY_OUTPUT']
                T_BH_idx = special_unknowns['T_BH']
                # 1. For each element in the boundary, couple to T_WY_OUTPUT
                # Equation for the housing surface: (T_element - T_output)/R1_tmp + Q
                for element in elements:
                    global_idx = element['global_idx']
                    elements_with_bc.add(global_idx)
                    A[global_idx, global_idx] -= 1.0 / R1_tmp / N_element / dx_m / dy_m / dz_m  
                    A[global_idx, T_WY_OUTPUT_idx] += 1.0 / R1_tmp / N_element / dx_m / dy_m / dz_m
                    b[global_idx] -= Q1 / N_element / dx_m / dy_m / dz_m
                # 2. Equation for T_WY_OUTPUT: sum over boundary elements (T_element - T_OUTPUT)/R1_tmp + (T_BH - T_OUTPUT)/R2_tmp + (T_amb - T_OUTPUT)/R4_tmp + Q = 0
                #    T_OUTPUT is T_WY_OUTPUT_idx, T_BH is T_BH_idx
                #    sum over all boundary elements: (T_element - T_WY_OUTPUT)/R1_tmp
                #    (T_BH - T_WY_OUTPUT)/R2_tmp
                #    (T_amb - T_OUTPUT)/R4_tmp
                #    + Q = 0
                #    Place this equation at row T_WY_OUTPUT_idx
                for element in elements:
                    global_idx = element['global_idx']
                    A[T_WY_OUTPUT_idx, global_idx] += 1.0 / R1_tmp / N_element
                    A[T_WY_OUTPUT_idx, T_WY_OUTPUT_idx] -= 1.0 / R1_tmp / N_element
                A[T_WY_OUTPUT_idx, T_BH_idx] += 1.0 / R2_tmp
                A[T_WY_OUTPUT_idx, T_WY_OUTPUT_idx] -= 1.0 / R2_tmp
                A[T_WY_OUTPUT_idx, T_WY_OUTPUT_idx] -= 1.0 / R4_tmp
                b[T_WY_OUTPUT_idx] -= T_amb / R4_tmp
                # 3. Equation for T_BH: (T_WY_OUTPUT - T_BH)/R2_tmp + (T_amb - T_BH)/R3_tmp + Q2 = 0
                #    Place this equation at row T_BH_idx
                
                A[T_BH_idx, T_WY_OUTPUT_idx] += 1.0 / R2_tmp
                A[T_BH_idx, T_BH_idx] -= 1.0 / R2_tmp
                A[T_BH_idx, T_BH_idx] -= 1.0 / R3_tmp
                b[T_BH_idx] -= T_amb / R3_tmp + Q2
            
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
                    
                    # Both types have 2 unknowns
                    T2_idx = act_start_idx      # Gearbox temperature
                    T4_idx = act_start_idx + 1  # Motor internal temperature
                    
                    if actuator_type == "ROLL":
                        if connecting_location == "housing":
                            # Loop through elements in boundary region to create coupling between all elements
                            for element_idx in housing_elements:
                                A[global_idx, element_idx] -= (1/(R2+R_contact)+1/R3+1/R4)/area/len(housing_elements)/dz_m
                            for element_idx in gearbox_elements:
                                A[global_idx, element_idx] += 1/(R2+R_contact)/area/len(gearbox_elements)/dz_m
                            A[global_idx, T4_idx] += 1/R4/area/dz_m
                            A[global_idx, T2_idx] += 1/R3/area/dz_m
                            b[global_idx] -= q_fets/area/dz_m
                            
                        elif connecting_location == "gearbox":
                            for element_idx in gearbox_elements:
                                A[global_idx, element_idx] -= (1/(R1+R_contact)+1/(R2+R_contact))/area/len(gearbox_elements)/dz_m
                            for element_idx in housing_elements:
                                A[global_idx, element_idx] += 1/(R2+R_contact)/area/len(housing_elements)/dz_m
                            A[global_idx, T2_idx] += 1/(R1+R_contact)/area/dz_m
                    
                    elif actuator_type == "PITCHYAW":  # PITCHYAW type
                        if connecting_location == "housing":
                            # Loop through elements in boundary region to create coupling between all elements
                            for element_idx in housing_elements:
                                A[global_idx, element_idx] -= (1/R2+1/R3+1/R4+1/R5)/area/len(housing_elements)/dz_m
                            for element_idx in gearbox_elements:
                                A[global_idx, element_idx] += 1/R2/area/len(gearbox_elements)/dz_m
                            for element_idx in motor_elements:
                                A[global_idx, element_idx] += 1/R5/area/len(motor_elements)/dz_m
                            A[global_idx, T4_idx] += 1/R4/area/dz_m
                            A[global_idx, T2_idx] += 1/R3/area/dz_m
                            b[global_idx] -= q_fets/area/dz_m
                            
                        elif connecting_location == "motor":
                            for element_idx in motor_elements:
                                A[global_idx, element_idx] -= 1/R5/area/len(motor_elements)/dz_m
                            for element_idx in housing_elements:
                                A[global_idx, element_idx] += 1/R5/area/len(housing_elements)/dz_m
                            
                        elif connecting_location == "gearbox":
                            for element_idx in gearbox_elements:
                                A[global_idx, element_idx] -= (1/R1+1/R2)/area/len(gearbox_elements)/dz_m
                            for element_idx in housing_elements:
                                A[global_idx, element_idx] += 1/R2/area/len(housing_elements)/dz_m
                            A[global_idx, T2_idx] += 1/R1/area/dz_m
            
        # Get convection parameters from region data
        region_data = info['region_data']
        htc = region_data.get('heat_transfer_coefficient', 7)  # W/m²K (default if not specified)
        T_inf = region_data.get('ambient_temperature', 21)  # °C
        
        if region_data.get('id') == 'wrist' or region_data.get('id') == 'forearm_lower':
            htc = 40
                
        # Process all surface, edge, and corner elements that don't have other BCs
        for element_type in ['surface', 'edge', 'corner']:
            for global_idx, i, j, k in info['element_indices'][element_type]:
                if global_idx not in elements_with_bc:
                    # Only apply convection to elements on top (k=Nz-1) and bottom (k=0) surfaces
                    if k == 0 or k == Nz-1:
                        # Apply convection boundary condition
                        A[global_idx, global_idx] -= htc / dz_m
                        b[global_idx] -= htc * T_inf / dz_m

    return A, b, actuator_elements 