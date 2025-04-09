import numpy as np
import matplotlib.pyplot as plt
from thermal_utils import read_geo_file
from thermal_parameters import get_parameters, ThermalParameters
from thermal_analysis import create_layer_coordinates, LayerMapping
from boundary_conditions import ElementBoundary, BoundaryCondition
from typing import List, Tuple
import json5

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

def plot_boundary_conditions(coords, boundary_conditions, title, surface_level, ax=None):
    """
    Plot the boundary conditions for a specific surface of a layer.
    
    Args:
        coords: ElementCoordinates object
        boundary_conditions: Dictionary mapping global indices to boundary condition types
        title: Plot title
        surface_level: Z-index of the surface to plot (0 for bottom, Nz-1 for top)
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
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
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, label=bc)
        for bc, color in bc_colors.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
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
    
    # Create coordinate systems
    metal_coords, plastic_coords = create_layer_coordinates(params)
    
    # Create mappings
    mapping = LayerMapping(params)
    
    # Create boundary condition systems
    metal_boundary = ElementBoundary(metal_coords, mapping, "metal_layer", params)
    plastic_boundary = ElementBoundary(plastic_coords, mapping, "plastic_layer", params)
    
    # Label boundary conditions
    metal_boundary_conditions = metal_boundary.label_boundary_conditions()
    plastic_boundary_conditions = plastic_boundary.label_boundary_conditions()

    # Analyze and print information for both layers
    analyze_layer(metal_boundary, metal_coords, "Metal")
    analyze_layer(plastic_boundary, plastic_coords, "Plastic")
                
    # Create figure with subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Plot boundary conditions for z=0 (bottom surface)
    plot_boundary_conditions(metal_coords, metal_boundary_conditions, "Metal Layer Boundary Conditions (z=0)", 0, axes[0, 0])
    plot_boundary_conditions(plastic_coords, plastic_boundary_conditions, "Plastic Layer Boundary Conditions (z=0)", 0, axes[0, 1])
    
    # Plot boundary conditions for z=Nz-1 (top surface)
    plot_boundary_conditions(metal_coords, metal_boundary_conditions, f"Metal Layer Boundary Conditions (z={metal_coords.Nz-1})", metal_coords.Nz - 1, axes[1, 0])
    plot_boundary_conditions(plastic_coords, plastic_boundary_conditions, f"Plastic Layer Boundary Conditions (z={plastic_coords.Nz-1})", plastic_coords.Nz - 1, axes[1, 1])
    
    # Set overall figure title
    fig.suptitle("Boundary Condition Visualization (Bottom and Top Surfaces)", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main() 