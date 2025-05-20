import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from thermal_parameters import ThermalParameters
from thermal_analysis import ElementCoordinates
from boundary_conditions import ElementBoundary

def plot_layer_surface(coords, title, ax=None):
    """
    Plot the top surface (z=z_max) of a layer.
    
    Args:
        coords: ElementCoordinates object
        title: Plot title
        ax: Matplotlib axis (optional)
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
            # Center x coordinates around x=0
            x = coords.get_coordinates_from_3d(i, j, coords.Nz-1)[0] - coords.Lx/2
            y = coords.get_coordinates_from_3d(i, j, coords.Nz-1)[1]
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
            # Center x coordinates around x=0
            x = coords.get_coordinates_from_3d(i, j, coords.Nz-1)[0] - coords.Lx/2
            y = coords.get_coordinates_from_3d(i, j, coords.Nz-1)[1]
            ax.plot(x, y, 'k.', markersize=1, alpha=0.5)  # Small dots at element centers
    
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

def plot_boundary_conditions(coords, boundary_conditions, title, surface_level, ax=None, adiabatic_pairs=None):
    """
    Plot the boundary conditions for a specific surface of a layer.
    
    Args:
        coords: ElementCoordinates object
        boundary_conditions: Dictionary mapping global indices to boundary condition types
        title: Plot title
        surface_level: Z-index of the surface to plot (0 for bottom, Nz-1 for top)
        ax: Matplotlib axis (optional)
        adiabatic_pairs: List of pre-calculated adiabatic pairs from region_info (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get the specified surface coordinates
    x_coords = []
    y_coords = []
    element_coords = {}  # Store coordinates for each element
    for j in range(coords.Ny):
        for i in range(coords.Nx):
            x = coords.get_coordinates_from_3d(i, j, surface_level)[0]
            y = coords.get_coordinates_from_3d(i, j, surface_level)[1]
            x_coords.append(x)
            y_coords.append(y)
            global_idx = coords.get_global_index(coords.Nx, coords.Ny, i, j, surface_level)
            element_coords[global_idx] = (x, y, i, j)
    
    # Create mesh grid
    X = np.array(x_coords).reshape(coords.Ny, coords.Nx)
    Y = np.array(y_coords).reshape(coords.Ny, coords.Nx)
    
    # Plot grid lines
    for i in range(coords.Ny):
        ax.plot(X[i, :], Y[i, :], 'k-', alpha=0.2, linewidth=0.5)
    for j in range(coords.Nx):
        ax.plot(X[:, j], Y[:, j], 'k-', alpha=0.2, linewidth=0.5)
    
    # Define colors for different boundary conditions
    bc_colors = {
        "INNER": 'gray',
        "ADIABATIC": 'blue',
        "CONST_Q": 'orange',
        "ACTUATOR_CONNECTED": 'red',
        "PLASTIC_COVERED": 'green',
        "CONVECTIVE": 'cyan',
        "MAPPED": 'purple'
    }
    
    # Plot elements
    for global_idx, (x, y, i, j) in element_coords.items():
        # Get the boundary condition
        bc_type = boundary_conditions.get(global_idx, "INNER")
        
        # Determine color
        if "CONST_Q" in bc_type:
            color = bc_colors["CONST_Q"]
        elif "ACTUATOR_CONNECTED" in bc_type:
            color = bc_colors["ACTUATOR_CONNECTED"]
        elif "MAPPED" in bc_type:
            color = bc_colors["MAPPED"]
        elif "ADIABATIC" in bc_type:
            color = bc_colors["ADIABATIC"]
        elif "PLASTIC_COVERED" in bc_type:
            color = bc_colors["PLASTIC_COVERED"]
        elif "CONVECTIVE" in bc_type:
            color = bc_colors["CONVECTIVE"]
        else:
            color = bc_colors["INNER"]
        
        # Plot element center
        ax.plot(x, y, '.', color=color, markersize=3)
        
        # Add rectangle for surface elements
        dx = coords.dx
        dy = coords.dy
        rect = plt.Rectangle((x - dx/2, y - dy/2), dx, dy,
                           facecolor=color, alpha=0.3, edgecolor=color, linewidth=1)
        ax.add_patch(rect)
    
    # Plot adiabatic pairs if provided
    if adiabatic_pairs:
        for pair in adiabatic_pairs:
            elem1 = pair['element1']
            elem2 = pair['element2']
            
            # Only plot if both elements are on the current surface level
            if elem1['k'] == surface_level and elem2['k'] == surface_level:
                x1 = elem1['x']
                y1 = elem1['y']
                x2 = elem2['x']
                y2 = elem2['y']
                
                # Draw a vertical line at the adiabatic boundary
                line_x = pair['line_x']
                y_min = min(y1, y2) - dy/2
                y_max = max(y1, y2) + dy/2
                ax.plot([line_x, line_x], [y_min, y_max], 'r-', linewidth=2, alpha=0.7)
                
                # Add small circles at the element centers
                ax.plot([x1, x2], [y1, y2], 'ro', markersize=4)
    
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
    ax.set_ylim(y_min, y_max)  # Keep y limits as is
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, label=bc_type)
        for bc_type, color in bc_colors.items()
    ]
    # Add adiabatic line to legend
    legend_elements.append(plt.Line2D([0], [0], color='r', linewidth=2, 
                                    label='Adiabatic Line'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    return ax

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
        print(f"Region dimensions: {width}x{height}x{thickness} mm")
        print("Boundary conditions:")
        for bc in region.get("boundary_conditions", []):
            print(f"  - Type: {bc['type']}")
    
    # Remove any unused subplots
    for idx in range(num_regions, len(axes)):
        fig.delaxes(axes[idx])
    
    # Set overall figure title
    fig.suptitle("Boundary Conditions for All Regions (z=0 surface)", fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('boundary_conditions.png', dpi=300, bbox_inches='tight')
    plt.close() 