import numpy as np
import matplotlib.pyplot as plt
from thermal_utils import read_geo_file
from thermal_parameters import get_parameters, ThermalParameters
from thermal_analysis import create_layer_coordinates, LayerMapping, update_matrix_with_geometries, update_matrix_with_boundary_conditions, preconditioner, ElementCoordinates, create_layer_matrix
from boundary_conditions import ElementBoundary, BoundaryCondition
from typing import List, Tuple
import json5
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.linalg import norm  # Import the norm function from scipy.linalg
import time

def analyze_layer(boundary, coords, layer_name):
    """Analyze and print heat source information for a layer."""
    print(f"\nHeat Source Elements in {layer_name} Layer:")
    print("-" * 50)
    
    # Get all heat source elements
    heat_source_elements = boundary.get_heat_source_elements()
    print(f"Total heat source elements: {len(heat_source_elements)}")
    
    # Get heat source elements by type
    convective_elements = boundary.get_heat_source_elements("CONVECTIVE")
    const_qflux_elements = boundary.get_heat_source_elements("CONST_QFLUX")
    print(f"Convective heat source elements: {len(convective_elements)}")
    print(f"Constant heat flux elements: {len(const_qflux_elements)}")
    
    # Get heat source elements by ID
    heat_source_ids = boundary.get_heat_source_ids()
    for source_id in heat_source_ids:
        elements = boundary.get_heat_source_elements_by_id(source_id)
        print(f"Heat source '{source_id}': {len(elements)} elements")
    
    # Print boundary condition breakdown
    total_elements = coords.Nx * coords.Ny * coords.Nz
    inner_elements = [idx for idx, bc in boundary.boundary_conditions.items() 
                     if bc == BoundaryCondition.INNER]
    mapped_elements = boundary.get_mapped_elements()
    adiabatic_elements = boundary.get_adiabatic_elements()
    
    print("\nBoundary Condition Summary:")
    print(f"Total elements: {total_elements}")
    print(f"Inner elements: {len(inner_elements)}")
    print(f"Mapped elements: {len(mapped_elements)}")
    print(f"Adiabatic elements: {len(adiabatic_elements)}")
    print(f"Convective elements: {len(convective_elements)}")
    print(f"Constant heat flux elements: {len(const_qflux_elements)}")
    
    # Check for overlapping regions
    check_overlapping_regions(boundary, layer_name)

def check_overlapping_regions(boundary, layer_name):
    """Check for overlapping regions between mapped, convective, and constant Q flux elements."""
    mapped_elements = set(boundary.get_mapped_elements())
    convective_elements = set(boundary.get_heat_source_elements("CONVECTIVE"))
    const_qflux_elements = set(boundary.get_heat_source_elements("CONST_QFLUX"))
    
    # Check mapped and convective overlap
    mapped_convective_overlap = mapped_elements & convective_elements
    if mapped_convective_overlap:
        print(f"\nWARNING: Found {len(mapped_convective_overlap)} elements in {layer_name} that are both mapped and convective")
        print("  Mapped boundary condition takes priority over convective")
    
    # Check mapped and constant Q flux overlap
    mapped_const_qflux_overlap = mapped_elements & const_qflux_elements
    if mapped_const_qflux_overlap:
        print(f"\nWARNING: Found {len(mapped_const_qflux_overlap)} elements in {layer_name} that are both mapped and constant Q flux")
        print("  Mapped boundary condition takes priority over constant Q flux")
    
    # Check convective and constant Q flux overlap
    convective_const_qflux_overlap = convective_elements & const_qflux_elements
    if convective_const_qflux_overlap:
        print(f"\nWARNING: Found {len(convective_const_qflux_overlap)} elements in {layer_name} that are both convective and constant Q flux")
        print("  Constant Q flux takes priority over convective")

def plot_layer_surface(coords, title, ax=None, mapped_elements_list=None, colors=None):
    """
    Plot the top surface (z=z_max) of a layer.
    
    Args:
        coords: ElementCoordinates object
        title: Plot title
        ax: Matplotlib axis (optional)
        mapped_elements_list: List of lists of (i, j) tuples for mapped elements
        colors: List of colors for each mapped elements list
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    print(f"\nPlotting layer surface for {title}")
    print(f"Grid dimensions: Nx={coords.Nx}, Ny={coords.Ny}, Nz={coords.Nz}")
    
    # Get the top surface coordinates
    x_coords = []
    y_coords = []
    for i in range(coords.Nx):
        for j in range(coords.Ny):
            x, y, _ = coords.get_coordinates_from_3d(i, j, coords.Nz-1)
            x_coords.append(x)
            y_coords.append(y)
    
    print(f"Number of points: {len(x_coords)}")
    print(f"X range: {min(x_coords):.2f} to {max(x_coords):.2f}")
    print(f"Y range: {min(y_coords):.2f} to {max(y_coords):.2f}")
    
    # Create mesh grid
    X = np.array(x_coords).reshape(coords.Ny, coords.Nx)
    Y = np.array(y_coords).reshape(coords.Ny, coords.Nx)
    
    # Plot vertical grid lines
    for i in range(coords.Nx):
        ax.plot(X[:, i], Y[:, i], 'k-', alpha=0.2, linewidth=0.5)
    
    # Plot horizontal grid lines
    for j in range(coords.Ny):
        ax.plot(X[j, :], Y[j, :], 'k-', alpha=0.2, linewidth=0.5)
    
    # Plot element centers
    for i in range(coords.Nx):
        for j in range(coords.Ny):
            x, y, _ = coords.get_coordinates_from_3d(i, j, coords.Nz-1)
            ax.plot(x, y, 'k.', markersize=1, alpha=0.5)  # Small dots at element centers
    
    # Default colors if not provided
    if colors is None:
        colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # Highlight mapped elements if provided
    if mapped_elements_list:
        for idx, mapped_elements in enumerate(mapped_elements_list):
            if mapped_elements:
                print(f"Number of mapped elements (type {idx}): {len(mapped_elements)}")
                for i, j in mapped_elements:
                    # Get element center coordinates
                    x_center, y_center, _ = coords.get_coordinates_from_3d(i, j, coords.Nz-1)
                    # Get element size
                    dx = coords.dx
                    dy = coords.dy
                    # Create rectangle for mapped element
                    rect = plt.Rectangle((x_center - dx/2, y_center - dy/2), dx, dy,
                                       facecolor=colors[idx], alpha=0.3, edgecolor=colors[idx], linewidth=1)
                    ax.add_patch(rect)
                    # Add a dot at the center of mapped elements
                    ax.plot(x_center, y_center, f'{colors[idx][0]}.', markersize=3)
    
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('x (mm)', fontsize=10)
    ax.set_ylabel('y (mm)', fontsize=10)
    ax.set_aspect('equal')
    
    # Set equal limits for x and y
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    max_range = max(x_max - x_min, y_max - y_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    
    return ax

def plot_mapping(coords1, coords2, mapping, title, mapped_elements_list1=None, mapped_elements_list2=None):
    """
    Plot the mapping between two layers' top surfaces in separate subplots.
    
    Args:
        coords1: First layer's ElementCoordinates
        coords2: Second layer's ElementCoordinates
        mapping: LayerMapping object
        title: Plot title
        mapped_elements_list1: List of lists of (i, j) tuples for mapped elements in first layer
        mapped_elements_list2: List of lists of (i, j) tuples for mapped elements in second layer
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    print("\nPlotting mapping visualization")
    print(f"Number of mappings: {len(mapping.region_mappings)}")
    
    # Define colors for different mapping types
    colors = ['red', 'blue', 'green']
    
    # Plot first layer
    plot_layer_surface(coords1, "First Layer", ax1, mapped_elements_list1, colors)
    
    # Plot second layer
    plot_layer_surface(coords2, "Second Layer", ax2, mapped_elements_list2, colors)
    
    # Plot mapping lines
    mapping_lines_count = 0
    for mapping_id, mapping_data in mapping.region_mappings.items():
        if "_source_" in mapping_id:
            source_key = mapping_id
            target_key = mapping_id.replace("_source_", "_target_")
            if target_key in mapping.region_mappings:
                print(f"\nProcessing mapping: {mapping_id}")
                print(f"Number of points in source: {len(mapping.region_mappings[source_key])}")
                print(f"Number of points in target: {len(mapping.region_mappings[target_key])}")
                
                for source_idx, target_idx in zip(mapping.region_mappings[source_key], 
                                                mapping.region_mappings[target_key]):
                    # Get coordinates of source element
                    i1, j1, k1 = coords1.get_3d_indices(source_idx)
                    if k1 != coords1.Nz - 1:  # Only plot for top surface
                        continue
                    x1, y1, _ = coords1.get_coordinates_from_3d(i1, j1, k1)
                    
                    # Get coordinates of target element
                    i2, j2, k2 = coords2.get_3d_indices(target_idx)
                    if k2 != coords2.Nz - 1:  # Only plot for top surface
                        continue
                    x2, y2, _ = coords2.get_coordinates_from_3d(i2, j2, k2)
                    
                    # Plot mapping line connecting the two points
                    ax1.plot([x1, x2], [y1, y2], 'g-', alpha=0.3)  # Green lines for mapping
                    ax2.plot([x1, x2], [y1, y2], 'g-', alpha=0.3)  # Green lines for mapping
                    mapping_lines_count += 1
    
    print(f"Number of mapping lines plotted: {mapping_lines_count}")
    
    # Set overall figure title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, (ax1, ax2)

def get_mapped_elements(coords, mapping, mapping_type="metal_to_plastic_source"):
    """
    Get the indices of mapped elements for a layer.
    
    Args:
        coords: ElementCoordinates object
        mapping: LayerMapping object
        mapping_type: Type of mapping:
            - "metal_to_plastic_source": Source elements in metal layer for metal-to-plastic mapping
            - "metal_to_plastic_target": Target elements in plastic layer for metal-to-plastic mapping
            - "metal_to_metal_source": Source elements in metal layer for metal-to-metal mapping
            - "metal_to_metal_target": Target elements in metal layer for metal-to-metal mapping
        
    Returns:
        List of (i, j) tuples for mapped elements
    """
    mapped_elements = set()
    
    for mapping_id, mapping_data in mapping.region_mappings.items():
        if mapping_type == "metal_to_plastic_source":
            if "_source_" in mapping_id and "metal_layer" in mapping_id:
                for idx in mapping_data:
                    i, j, k = coords.get_3d_indices(idx)
                    if k == coords.Nz - 1:  # Only consider top surface
                        mapped_elements.add((i, j))
        elif mapping_type == "metal_to_plastic_target":
            if "_target_" in mapping_id and "plastic_layer" in mapping_id:
                for idx in mapping_data:
                    i, j, k = coords.get_3d_indices(idx)
                    if k == coords.Nz - 1:  # Only consider top surface
                        mapped_elements.add((i, j))
        elif mapping_type == "metal_to_metal_source":
            if "_source_" in mapping_id and "metal_layer" in mapping_id:
                for idx in mapping_data:
                    i, j, k = coords.get_3d_indices(idx)
                    if k == coords.Nz - 1:  # Only consider top surface
                        mapped_elements.add((i, j))
        elif mapping_type == "metal_to_metal_target":
            if "_target_" in mapping_id and "metal_layer" in mapping_id:
                for idx in mapping_data:
                    i, j, k = coords.get_3d_indices(idx)
                    if k == coords.Nz - 1:  # Only consider top surface
                        mapped_elements.add((i, j))
    
    return list(mapped_elements)

def plot_boundary_conditions(coords, boundary_conditions, title, surface_level, ax=None, figsize=(12, 8)):
    """
    Plot the boundary conditions for a specific surface of a layer.
    
    Args:
        coords: ElementCoordinates object
        boundary_conditions: Dictionary mapping global indices to boundary condition types
        title: Plot title
        surface_level: Z-index of the surface to plot (0 for bottom, Nz-1 for top)
        ax: Matplotlib axis (optional)
        figsize: Tuple of (width, height) in inches for the figure size
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    print(f"\nPlotting boundary conditions for {title} at z-level {surface_level}")
    print(f"Grid dimensions: Nx={coords.Nx}, Ny={coords.Ny}, Nz={coords.Nz}")
    
    # Get the specified surface coordinates
    x_coords = []
    y_coords = []
    for i in range(coords.Nx):
        for j in range(coords.Ny):
            x, y, _ = coords.get_coordinates_from_3d(i, j, surface_level)
            x_coords.append(x)
            y_coords.append(y)
    
    # Create mesh grid
    X = np.array(x_coords).reshape(coords.Nx, coords.Ny)
    Y = np.array(y_coords).reshape(coords.Nx, coords.Ny)
    
    # Plot grid lines
    for i in range(coords.Ny):
        ax.plot(X[:, i], Y[:, i], 'k-', alpha=0.2, linewidth=0.5)
    for j in range(coords.Nx):
        ax.plot(X[j, :], Y[j, :], 'k-', alpha=0.2, linewidth=0.5)
    
    # Define colors for different boundary conditions
    bc_colors = {
        BoundaryCondition.INNER: 'gray',
        BoundaryCondition.ADIABATIC: 'blue',
        BoundaryCondition.MAPPED: 'red',
        BoundaryCondition.CONVECTIVE: 'green',
        BoundaryCondition.CONST_Qflux: 'orange',
        BoundaryCondition.CONVECTIVE_AIRGAP: 'turquoise',
        "MAPPED_CONST_QFLUX": 'purple',  # Special color for combined MAPPED and CONST_Qflux
        "CONST_QFLUX_CONVECTIVE": 'pink'  # Special color for combined CONST_Qflux and CONVECTIVE
    }
    
    # Plot element centers with boundary condition colors
    for i in range(coords.Nx):
        for j in range(coords.Ny):
            x, y, _ = coords.get_coordinates_from_3d(i, j, surface_level)
            global_idx = coords.get_global_index(coords.Nx, coords.Ny, i, j, surface_level)
            
            # Get the combined boundary condition
            if isinstance(boundary_conditions, dict):
                bc = boundary_conditions.get(global_idx, [BoundaryCondition.INNER])
                if len(bc) > 1:
                    # Handle combined boundary conditions
                    if BoundaryCondition.MAPPED in bc and BoundaryCondition.CONST_Qflux in bc:
                        color = bc_colors["MAPPED_CONST_QFLUX"]
                    elif BoundaryCondition.CONST_Qflux in bc and BoundaryCondition.CONVECTIVE in bc:
                        color = bc_colors["CONST_QFLUX_CONVECTIVE"]
                    elif BoundaryCondition.MAPPED in bc:
                        color = bc_colors[BoundaryCondition.MAPPED]  # MAPPED takes priority
                    elif BoundaryCondition.CONVECTIVE_AIRGAP in bc:
                        color = bc_colors[BoundaryCondition.CONVECTIVE_AIRGAP]
                    elif BoundaryCondition.CONVECTIVE in bc:
                        color = bc_colors[BoundaryCondition.CONVECTIVE]
                    else:
                        color = bc_colors[bc[0]]
                else:
                    color = bc_colors[bc[0]]
            else:
                color = bc_colors[boundary_conditions.get(global_idx, BoundaryCondition.INNER)]
            
            # Plot element center
            ax.plot(x, y, '.', color=color, markersize=3)
            
            # Add rectangle for surface elements
            if color != bc_colors[BoundaryCondition.INNER]:
                dx = coords.dx
                dy = coords.dy
                rect = plt.Rectangle((x - dx/2, y - dy/2), dx, dy,
                                   facecolor=color, alpha=0.3, edgecolor=color, linewidth=1)
                ax.add_patch(rect)
    
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('x (mm)', fontsize=12)
    ax.set_ylabel('y (mm)', fontsize=12)
    ax.set_aspect('equal')
    
    # Set equal limits for x and y
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    max_range = max(x_max - x_min, y_max - y_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    
    # Add legend with larger font size
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, label=bc)
        for bc, color in bc_colors.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Adjust layout to prevent title and legend overlap
    plt.tight_layout()
    
    return ax

def print_boundary_statistics(boundary_conditions: dict, layer_name: str):
    """Print statistics about boundary conditions for a layer"""
    total_elements = len(boundary_conditions)
    inner_count = sum(1 for bc in boundary_conditions.values() if bc == BoundaryCondition.INNER)
    adiabatic_count = sum(1 for bc in boundary_conditions.values() if bc == BoundaryCondition.ADIABATIC)
    mapped_count = sum(1 for bc in boundary_conditions.values() if bc == BoundaryCondition.MAPPED)
    
    print(f"\nBoundary Condition Statistics for {layer_name}:")
    print(f"Total elements: {total_elements}")
    print(f"Inner elements: {inner_count} ({inner_count/total_elements*100:.1f}%)")
    print(f"Adiabatic elements: {adiabatic_count} ({adiabatic_count/total_elements*100:.1f}%)")
    print(f"Mapped elements: {mapped_count} ({mapped_count/total_elements*100:.1f}%)")

def main():
    # Read geometry data
    with open("GEO.json", "r") as f:
        geo_data = json5.load(f)
    
    # Get parameters
    params = get_parameters(geo_data)
    
    # Create coordinate system for metal layer only
    metal_coords = ElementCoordinates(params, "metal_layer")
    
    # Create mappings
    mapping = LayerMapping(params)
    
    # Create boundary condition system for metal layer
    metal_boundary = ElementBoundary(metal_coords, mapping, "metal_layer", params)
    
    # Label boundary conditions
    metal_boundary_conditions = metal_boundary.label_boundary_conditions()

    # Analyze and print information for metal layer
    analyze_layer(metal_boundary, metal_coords, "Metal")
                
    # Create figure with subplots (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    
    # Plot boundary conditions for z=0 (bottom surface)
    plot_boundary_conditions(metal_coords, metal_boundary_conditions, "Metal Layer Boundary Conditions (z=0)", 0, axes[0], figsize=(12, 8))
    
    # Plot boundary conditions for z=Nz-1 (top surface)
    plot_boundary_conditions(metal_coords, metal_boundary_conditions, f"Metal Layer Boundary Conditions (z={metal_coords.Nz-1})", metal_coords.Nz - 1, axes[1], figsize=(12, 8))
    
    # Set overall figure title
    fig.suptitle("Boundary Condition Visualization (Bottom and Top Surfaces)", fontsize=16, y=0.95)
    
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
    
    # Show the plots
    plt.show()
    
    # Initialize matrices using functions from thermal_analysis.py
    print("\nInitializing matrices...")
    
    # Create matrix for metal layer only
    A = create_layer_matrix(metal_coords.Nx, metal_coords.Ny, metal_coords.Nz, 
                          metal_coords.dx*1e-3, metal_coords.dy*1e-3, metal_coords.dz*1e-3,
                          params.get_region_thermal_conductivity("metal_layer"))
    A = A.toarray()  # Convert to dense array
        
    # Initialize vector for metal layer
    total_elements = metal_coords.Nx * metal_coords.Ny * metal_coords.Nz
    b = np.zeros(total_elements)
    
    print("\nUpdating matrix with geometry data...")
    A_updated_GEO = update_matrix_with_geometries(A, metal_coords, params)
    
    print("\nUpdating matrix with boundary conditions...")
    A_updated_BC, b_updated_BC = update_matrix_with_boundary_conditions(A_updated_GEO, b, metal_coords, params, metal_boundary, mapping)
    
    # Convert to sparse matrix for solver
    A_final = scipy.sparse.csr_matrix(A_updated_BC)
    
    print("\nSolving system...")
    # Initial guess - all zeros
    u0 = np.ones_like(b_updated_BC)*30
    
    # Set relative tolerance
    rel_tol = 1e-6
    
    # Callback function to monitor convergence
    def callback(pr_norm):
        print(f"Current residual norm: {pr_norm:.4e}", end="\r")
    
    # Create preconditioner
    M = preconditioner(A_updated_BC)
    
    # Solve system using GMRES
    start_time = time.time()
    u, exitCode = gmres(A_updated_BC, b_updated_BC, M=M, x0=u0, atol=rel_tol, callback=callback, callback_type='pr_norm')
    solve_time = time.time() - start_time
    print(f"\nSolve time: {solve_time:.2f} seconds")
    # Calculate and print residual norm
    residual = A_final @ u - b_updated_BC
    residual_norm = np.linalg.norm(residual)
    print(f"\nFinal residual norm: {residual_norm:.2e}")
    if exitCode == 0:
        print("\nSolution converged successfully!")
    else:
        print(f"\nWarning: Solution did not converge, exit code: {exitCode}")
        
    # Print temperature results for heat source regions
    print("\nHeat Source Region Temperatures:")
    print("-" * 50)
    
    # Create a list to store all output lines
    output_lines = []
    
    # Process metal layer heat sources
    for source_id, elements in metal_boundary.heat_source_elements.items():
        if not elements:
            continue
            
        temps = u[list(elements)]
        output = f"\nMetal Layer - Heat Source '{source_id}':\n"
        output += f"Average Temperature: {np.mean(temps):.2f} C\n"
        output += f"Min Temperature: {np.min(temps):.2f} C\n"
        output += f"Max Temperature: {np.max(temps):.2f} C"
        print(output)
        output_lines.append(output)
    
    # Save results to file
    with open('thermal_results.txt', 'w') as f:
        f.write("Thermal Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Solve time: {solve_time:.2f} seconds\n")
        f.write(f"Final residual norm: {residual_norm:.2e}\n")
        if exitCode == 0:
            f.write("Solution converged successfully!\n")
        else:
            f.write(f"Warning: Solution did not converge, exit code: {exitCode}\n")
        f.write("\nHeat Source Region Temperatures:\n")
        f.write("-" * 50 + "\n")
        for line in output_lines:
            f.write(line + "\n")
    
    print("\nResults have been saved to 'thermal_results.txt'")
                            
    # Plot temperature distributions on top and bottom surfaces
    print("\nGenerating temperature distribution plots...")
    
    # Get dimensions
    metal_nx, metal_ny = metal_coords.Nx, metal_coords.Ny
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot metal layer bottom surface (k=0)
    metal_bottom = u[:metal_nx*metal_ny].reshape(metal_ny, metal_nx)
    im1 = ax1.imshow(metal_bottom, cmap='jet', interpolation='nearest', origin='lower')
    ax1.set_title('Metal Layer Bottom Surface Temperature (°C)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot metal layer top surface (k=Nz-1)
    metal_top_start = metal_nx*metal_ny*(metal_coords.Nz-1)
    metal_top = u[metal_top_start:metal_top_start + metal_nx*metal_ny].reshape(metal_ny, metal_nx)
    im2 = ax2.imshow(metal_top, cmap='jet', interpolation='nearest', origin='lower')
    ax2.set_title('Metal Layer Top Surface Temperature (°C)')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main() 