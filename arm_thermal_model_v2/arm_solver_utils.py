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
    update_matrix_with_boundary_conditions,
    preconditioner,
    ElementCoordinates,
    create_layer_matrix,
    calculate_system_size
)
from boundary_conditions import ElementBoundary, BoundaryCondition
from visualization import plot_layer_surface, plot_boundary_conditions
from typing import List, Tuple, Dict

def read_geometry(file_path: str = "GEO.json") -> Tuple[dict, ThermalParameters, List[dict], List[dict], float]:
    """
    Read and parse the geometry file.
    """
    with open(file_path, "r") as f:
        geo_data = json5.load(f)
    params = get_parameters(geo_data)
    regions = geo_data["regions"]
    actuators = geo_data["actuators"]
    metal_conductivity = geo_data["environment"]["metal_conductivity"]
    print(f"\nUsing metal conductivity: {metal_conductivity} W/mK for all regions")
    return geo_data, params, regions, actuators, metal_conductivity

def initialize_global_matrix(total_elements: int, region_info: Dict, actuator_info: Dict, metal_conductivity: float) -> Tuple[scipy.sparse.lil_matrix, np.ndarray]:
    """
    Initialize the global system matrix and vector.
    """
    print("\nInitializing global matrix and vector...")
    total_actuator_unknowns = 0
    for info in actuator_info.values():
        total_actuator_unknowns += 2
    total_size = total_elements + total_actuator_unknowns
    A = scipy.sparse.lil_matrix((total_size, total_size))
    b = np.zeros(total_size)
    for region_id, info in region_info.items():
        print(f"\nBuilding matrix block for region: {region_id}")
        coords = info['coords']
        start_idx = info['start_idx']
        num_elements = info['num_elements']
        region_matrix = create_layer_matrix(coords.Nx, coords.Ny, coords.Nz,
                                          coords.dx*1e-3, coords.dy*1e-3, coords.dz*1e-3,
                                          metal_conductivity)
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
    return A, b

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
    geo_data, params, regions, actuators, metal_conductivity = read_geometry()
    region_info, actuator_info, total_elements, actuator_start_idx = calculate_system_size(
        regions, actuators, params
    )
    visualize_regions(regions, region_info, params)
    A, b = initialize_global_matrix(total_elements, region_info, actuator_info, metal_conductivity)
    A = update_matrix_with_geometries(A, region_info, actuator_info, metal_conductivity)
    A, b, actuator_elements = update_matrix_with_boundary_conditions(A, b, region_info, actuator_info, metal_conductivity)
    return dict(
        geo_data=geo_data,
        params=params,
        regions=regions,
        actuators=actuators,
        metal_conductivity=metal_conductivity,
        region_info=region_info,
        actuator_info=actuator_info,
        total_elements=total_elements,
        actuator_start_idx=actuator_start_idx,
        A=A,
        b=b,
        actuator_elements=actuator_elements
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
            print(f"Average Gearbox Structure Temperature: {avg_gearbox_temp:.2f}°C")
        if avg_motor_temp is not None:
            print(f"Average Motor Structure Temperature: {avg_motor_temp:.2f}°C")
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