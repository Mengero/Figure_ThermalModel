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
    
    Args:
        file_path: Path to the geometry file
        
    Returns:
        Tuple containing:
        - geo_data: Raw geometry data
        - params: Parsed thermal parameters
        - regions: List of regions
        - actuators: List of actuators
        - metal_conductivity: Metal thermal conductivity
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
    
    Args:
        total_elements: Total number of elements in system
        region_info: Dictionary with region information
        actuator_info: Dictionary with actuator information
        metal_conductivity: Metal thermal conductivity
        
    Returns:
        Tuple containing:
        - A: Global system matrix
        - b: Global system vector
    """
    print("\nInitializing global matrix and vector...")
    
    # Calculate total system size including actuator unknowns
    total_actuator_unknowns = 0
    for info in actuator_info.values():
        # Both ROLL and PITCHYAW types have 2 unknowns
        total_actuator_unknowns += 2
    
    total_size = total_elements + total_actuator_unknowns
    
    # Initialize global sparse matrix and vector with correct size
    A = scipy.sparse.lil_matrix((total_size, total_size))
    b = np.zeros(total_size)
    
    # Fill matrix block by block for each region
    for region_id, info in region_info.items():
        print(f"\nBuilding matrix block for region: {region_id}")
        coords = info['coords']
        start_idx = info['start_idx']
        num_elements = info['num_elements']
        
        # Create region matrix block using metal conductivity
        region_matrix = create_layer_matrix(coords.Nx, coords.Ny, coords.Nz,
                                          coords.dx*1e-3, coords.dy*1e-3, coords.dz*1e-3,
                                          metal_conductivity)
        
        # Insert region matrix block into global matrix
        end_idx = start_idx + num_elements
        A[start_idx:end_idx, start_idx:end_idx] = region_matrix
        print(f"Added block from index {start_idx} to {end_idx}")
    
    # Initialize actuator blocks in the matrix
    current_actuator_idx = total_elements  # Start after all elements
    for actuator_id, info in actuator_info.items():
        print(f"\nInitializing matrix block for actuator: {actuator_id}")
        
        # All actuator types have 2 unknowns
        num_unknowns = 2
        
        # Update actuator info with start index and number of unknowns
        info['start_idx'] = current_actuator_idx
        info['num_unknowns'] = num_unknowns
        
        # Create actuator block
        end_idx = current_actuator_idx + num_unknowns
        actuator_block = np.zeros((num_unknowns, num_unknowns))
        A[current_actuator_idx:end_idx, current_actuator_idx:end_idx] = actuator_block
        print(f"Added {info['type']} actuator block from index {current_actuator_idx} to {end_idx} (size: {num_unknowns}x{num_unknowns})")
        
        # Update current index for next actuator
        current_actuator_idx = end_idx
    
    return A, b

def visualize_regions(regions: List[dict], region_info: Dict, params: ThermalParameters) -> None:
    """
    Create visualization plots for all regions.
    
    Args:
        regions: List of regions from geometry file
        region_info: Dictionary with region information
        params: Thermal parameters
    """
    num_regions = len(regions)
    
    # Calculate subplot grid dimensions
    num_cols = 3  # We want 3 columns
    num_rows = (num_regions + num_cols - 1) // num_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6*num_rows))
    axes = axes.flatten()  # Flatten axes array for easier indexing
    
    # Plot each region
    for idx, region in enumerate(regions):
        region_id = region["id"]
        print(f"\nProcessing region: {region_id}")
        
        # Get stored coordinate system and adiabatic pairs for this region
        info = region_info[region_id]
        coords = info['coords']
        adiabatic_pairs = info.get('adiabatic_pairs', [])
        
        # Create boundary condition system
        boundary = ElementBoundary(coords, None, region_id, params)
        
        # Get boundary conditions
        boundary_conditions = boundary.label_boundary_conditions()
        
        # Plot boundary conditions for z=0 surface
        ax = axes[idx]
        plot_boundary_conditions(coords, boundary_conditions, f"{region_id} (z=0)", 0, ax, adiabatic_pairs)
        
        # Add region-specific information
        width = region["width"]
        height = region["height"]
        thickness = region["thickness"]
        ax.set_title(f"{region_id}\nSize: {width}x{height}x{thickness} mm", fontsize=12, pad=10)
        
        # Print region statistics
        # print(f"Region dimensions: {width}x{height}x{thickness} mm")
        # print("Boundary conditions:")
        # for bc in region.get("boundary_conditions", []):
        #     print(f"  - Type: {bc['type']}")
        # if adiabatic_pairs:
        #     print(f"  - Adiabatic pairs: {len(adiabatic_pairs)}")
    
    # Remove any unused subplots
    for idx in range(num_regions, len(axes)):
        fig.delaxes(axes[idx])
    
    # Set overall figure title
    fig.suptitle("Boundary Conditions for All Regions (z=0 surface)", fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def main():
    """Main function to solve the thermal model."""
    # Step 1: Read geometry and parameters
    geo_data, params, regions, actuators, metal_conductivity = read_geometry()
    
    # Step 2: Calculate system size and store region/actuator information
    region_info, actuator_info, total_elements, actuator_start_idx = calculate_system_size(
        regions, actuators, params
    )
    
    # Step 3: Visualize the regions and boundary conditions
    visualize_regions(regions, region_info, params)
    
    # Step 4: Initialize the global matrix and vector
    A, b = initialize_global_matrix(total_elements, region_info, actuator_info, metal_conductivity)
    print("\nBase matrix initialization complete.")
    
    # Step 5: Update matrix with geometry data
    print("\nUpdating matrix with geometry data...")
    A = update_matrix_with_geometries(A, region_info, actuator_info, metal_conductivity)
    print("Matrix updated with geometry data.")
    
    # Step 6: Update matrix with boundary conditions
    print("\nUpdating matrix with boundary conditions...")
    A, b, actuator_elements = update_matrix_with_boundary_conditions(A, b, region_info, actuator_info, metal_conductivity)
    print("Matrix updated with boundary conditions.")
    
    # Step 7: Convert to CSR format for solver
    A_final = scipy.sparse.csr_matrix(A)
    
    print("\nSolving system...")
    # Initial guess - all temperatures at 30°C
    u0 = np.ones_like(b) * 30.0
    
    # Set relative tolerance
    rel_tol = 1e-4
    
    # Store residual norms for plotting
    iteration = 0
    
    # Callback function to monitor convergence
    def callback(pr_norm):
        nonlocal iteration
        iteration += 1
        print(f"Iteration {iteration}: residual norm = {pr_norm:.4e}", end="\r")
    
    print("\nPreparing solver...")
    
    # Convert to CSR format for efficient solving
    A = A.tocsr()
    
    # Try direct sparse solver first
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
        
        # Create ILU preconditioner
        try:
            print("Computing ILU preconditioner...")
            # Convert to CSC format for ILU
            A_csc = A.tocsc()
            ILU = spilu(A_csc, drop_tol=1e-4, fill_factor=20)
            M_x = lambda x: ILU.solve(x)
            M = LinearOperator(A.shape, M_x)
            print("ILU preconditioner ready")
            
        except Exception as e:
            print(f"ILU preconditioner failed: {e}")
            print("Falling back to diagonal preconditioner...")
            M = preconditioner(A)
        
        # Try different iterative solvers in sequence
        solvers = [
            (bicgstab, "BiCGSTAB"),
            (lgmres, "LGMRES"),
            (gmres, "GMRES")
        ]
        
        u = None
        exitCode = None
        
        for solver, name in solvers:
            try:
                print(f"\nTrying {name} solver...")
                start_time = time.time()
                if solver == gmres:
                    u, exitCode = solver(A, b, M=M, x0=u0, atol=rel_tol, 
                                      callback=callback, callback_type='pr_norm')
                else:
                    u, exitCode = solver(A, b, M=M, x0=u0, atol=rel_tol, callback=callback)
                solve_time = time.time() - start_time
                
                if exitCode == 0:
                    print(f"\n{name} solve succeeded!")
                    break
                else:
                    print(f"\n{name} did not converge, trying next solver...")
                    
            except Exception as e:
                print(f"{name} solver failed: {e}")
                continue
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep final plot window open
    
    # Calculate and print residual norm
    residual = A_final @ u - b
    residual_norm = np.linalg.norm(residual)
    print(f"\nFinal residual norm: {residual_norm:.2e}")
    if exitCode == 0:
        print("\nSolution converged successfully!")
    else:
        print(f"\nWarning: Solution did not converge, exit code: {exitCode}")
    
    # Print temperatures for each region and actuator
    print("\nTemperature Results:")
    print("-" * 50)
    
    # Process each region
    for region_id, info in region_info.items():
        start_idx = info['start_idx']
        num_elements = info['num_elements']
        region_temps = u[start_idx:start_idx + num_elements]
        
        print(f"\nRegion: {region_id}")
        print(f"Average Temperature: {np.mean(region_temps):.2f}°C")
        print(f"Min Temperature: {np.min(region_temps):.2f}°C")
        print(f"Max Temperature: {np.max(region_temps):.2f}°C")
    
    # Process each actuator
    for actuator_id, info in actuator_info.items():
        start_idx = info['start_idx']
        num_unknowns = info['num_unknowns']
        actuator_temps = u[start_idx:start_idx + num_unknowns]
        
        # Calculate average temperatures for connecting structures
        housing_elements = actuator_elements[actuator_id]['housing_elements']
        gearbox_elements = actuator_elements[actuator_id]['gearbox_elements']
        motor_elements = actuator_elements[actuator_id]['motor_elements']
        
        # Calculate average temperatures if elements exist
        avg_housing_temp = np.mean([u[idx] for idx in housing_elements]) if housing_elements else None
        avg_gearbox_temp = np.mean([u[idx] for idx in gearbox_elements]) if gearbox_elements else None
        avg_motor_temp = np.mean([u[idx] for idx in motor_elements]) if motor_elements else None
        
        print(f"\nActuator: {actuator_id} ({info['type']})")
        print(f"Gearbox Temperature (T2): {actuator_temps[0]:.2f}°C")
        print(f"Motor Temperature (T4): {actuator_temps[1]:.2f}°C")
        
        # Print connecting structure temperatures
        if avg_housing_temp is not None:
            print(f"Average Housing Structure Temperature: {avg_housing_temp:.2f}°C")
        if avg_gearbox_temp is not None:
            print(f"Average Gearbox Structure Temperature: {avg_gearbox_temp:.2f}°C")
        if avg_motor_temp is not None:
            print(f"Average Motor Structure Temperature: {avg_motor_temp:.2f}°C")
    
    # Save results to file
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
        
        f.write("\nActuator Temperatures:\n")
        f.write("-" * 50 + "\n")
        for actuator_id, info in actuator_info.items():
            start_idx = info['start_idx']
            num_unknowns = info['num_unknowns']
            actuator_temps = u[start_idx:start_idx + num_unknowns]
            
            # Calculate average temperatures for connecting structures
            housing_elements = actuator_elements[actuator_id]['housing_elements']
            gearbox_elements = actuator_elements[actuator_id]['gearbox_elements']
            motor_elements = actuator_elements[actuator_id]['motor_elements']
            
            # Calculate average temperatures if elements exist
            avg_housing_temp = np.mean([u[idx] for idx in housing_elements]) if housing_elements else None
            avg_gearbox_temp = np.mean([u[idx] for idx in gearbox_elements]) if gearbox_elements else None
            avg_motor_temp = np.mean([u[idx] for idx in motor_elements]) if motor_elements else None
            
            f.write(f"\nActuator: {actuator_id} ({info['type']})\n")
            f.write(f"Gearbox Temperature (T2): {actuator_temps[0]:.2f}°C\n")
            f.write(f"Motor Temperature (T4): {actuator_temps[1]:.2f}°C\n")
            
            # Write connecting structure temperatures
            if avg_housing_temp is not None:
                f.write(f"Average Housing Structure Temperature: {avg_housing_temp:.2f}°C\n")
            if avg_gearbox_temp is not None:
                f.write(f"Average Gearbox Structure Temperature: {avg_gearbox_temp:.2f}°C\n")
            if avg_motor_temp is not None:
                f.write(f"Average Motor Structure Temperature: {avg_motor_temp:.2f}°C\n")
    
    print("\nResults have been saved to 'arm_thermal_results.txt'")
    
    # Plot temperature distributions for each region
    # print("\nGenerating temperature distribution plots...")
    # for region_id, info in region_info.items():
    #     coords = info['coords']
    #     start_idx = info['start_idx']
        
    #     # Create figure with two subplots for bottom and top surfaces
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    #     # Plot bottom surface (k=0)
    #     bottom_temps = u[start_idx:start_idx + coords.Nx*coords.Ny].reshape(coords.Ny, coords.Nx)
    #     im1 = ax1.imshow(bottom_temps, cmap='jet', interpolation='nearest', origin='lower')
    #     ax1.set_title(f'{region_id} Bottom Surface Temperature (°C)')
    #     plt.colorbar(im1, ax=ax1)
        
    #     # Plot top surface (k=Nz-1)
    #     top_start = start_idx + coords.Nx*coords.Ny*(coords.Nz-1)
    #     top_temps = u[top_start:top_start + coords.Nx*coords.Ny].reshape(coords.Ny, coords.Nx)
    #     im2 = ax2.imshow(top_temps, cmap='jet', interpolation='nearest', origin='lower')
    #     ax2.set_title(f'{region_id} Top Surface Temperature (°C)')
    #     plt.colorbar(im2, ax=ax2)
        
    #     plt.suptitle(f'Temperature Distribution for {region_id}')
    #     plt.tight_layout()
    #     plt.show()

if __name__ == "__main__":
    main() 