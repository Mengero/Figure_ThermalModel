#!/usr/bin/env python3
"""
Calculate the total number of elements in the thermal model and provide a breakdown by region.
This script uses the ThermalParameters class to access the model parameters and calculate element counts.
"""

import os
import sys
from thermal_utils import read_geo_file
from thermal_parameters import get_parameters, ThermalParameters
from typing import Tuple

def calculate_region_elements(params: ThermalParameters, region_id: str) -> Tuple[int, int, int, int, float, float, float, float]:
    """
    Calculate the number of point elements and their spacing for a region.
    Point spacing is kept consistent across layers, while number of points can vary.
    
    Args:
        params: ThermalParameters object containing geometry data
        region_id: ID of the region to calculate elements for
        
    Returns:
        Tuple containing:
        - Total number of point elements
        - Number of points in x direction
        - Number of points in y direction
        - Number of points in z direction
        - Point spacing in x direction (mm)
        - Point spacing in y direction (mm)
        - Point spacing in z direction (mm)
        - Point volume (mm³) - for compatibility, set to 1.0
    """
    # Get region data
    metal_region = params.get_region_by_id("metal_layer")
    plastic_region = params.get_region_by_id("plastic_layer")
    
    # Get mesh settings
    mesh_settings = params.get_mesh_settings()
    Nx = mesh_settings.get("Nx", 20)
    Ny = mesh_settings.get("Ny", 20)
    Nz = mesh_settings.get("Nz", 1)
    
    # Get region dimensions
    Lx, Ly, Lz = params.get_region_dimensions(region_id)
    
    # For metal layer, use the specified Nx, Ny
    if region_id == "metal_layer":
        # Calculate point spacing (distance between points)
        dx = Lx / (Nx - 1) if Nx > 1 else Lx
        dy = Ly / (Ny - 1) if Ny > 1 else Ly
        dz = Lz / (Nz - 1) if Nz > 1 else Lz
    else:
        # For plastic layer, use the same point spacing as metal layer
        metal_Lx, metal_Ly, _ = params.get_region_dimensions("metal_layer")
        dx = metal_Lx / (Nx - 1) if Nx > 1 else metal_Lx  # Use same spacing as metal layer
        dy = metal_Ly / (Ny - 1) if Ny > 1 else metal_Ly  # Use same spacing as metal layer
        dz = Lz / (Nz - 1) if Nz > 1 else Lz
        
        # Calculate number of points for plastic layer to match the spacing
        Nx = int(round(Lx / dx)) + 1
        Ny = int(round(Ly / dy)) + 1
    
    # Calculate total number of point elements
    num_elements = Nx * Ny * Nz
    
    # Point volume is 1.0 since we're dealing with points
    point_volume = 1.0
    
    return num_elements, Nx, Ny, Nz, dx, dy, dz, point_volume

def calculate_total_elements(params: ThermalParameters) -> dict:
    """
    Calculate the total number of elements in the model and provide a breakdown by region.
    
    Args:
        params: ThermalParameters object
        
    Returns:
        Dictionary with element counts and details
    """
    results = {
        "total_elements": 0,
        "regions": {},
        "element_sizes": {},
        "mesh_dimensions": {},
        "memory_estimate": 0.0
    }
    
    # Get global mesh settings
    global_nx, global_ny, global_nz = params.get_mesh_dimensions()
    results["global_mesh"] = {
        "Nx": global_nx,
        "Ny": global_ny,
        "Nz": global_nz
    }
    
    # Get all region IDs
    region_ids = params.get_region_ids()
    
    # Calculate elements for each region
    for region_id in region_ids:
        num_elements, nx, ny, nz, dx, dy, dz, element_volume = calculate_region_elements(params, region_id)
        
        # Get region dimensions
        Lx, Ly, Lz = params.get_region_dimensions(region_id)
        
        # Store results
        results["regions"][region_id] = {
            "num_elements": num_elements,
            "element_size": (dx, dy, dz),
            "element_volume": element_volume,
            "region_volume": Lx * Ly * Lz,
            "region_dimensions": (Lx, Ly, Lz)
        }
        
        # Store mesh dimensions
        results["mesh_dimensions"][region_id] = {
            "Nx": nx,
            "Ny": ny,
            "Nz": nz
        }
        
        # Add to total
        results["total_elements"] += num_elements
        
        # Store element sizes
        results["element_sizes"][region_id] = {
            "dx": dx,
            "dy": dy,
            "dz": dz
        }
    
    # Estimate memory usage (assuming each element requires ~100 bytes of memory)
    # This is a rough estimate including temperature, material properties, connections, etc.
    memory_per_element = 100  # bytes
    results["memory_estimate"] = results["total_elements"] * memory_per_element / (1024 * 1024)  # MB
    
    return results

def print_element_report(results: dict):
    """
    Print a report of the element calculation results.
    
    Args:
        results: Dictionary with element counts and details
    """
    print("\n=== THERMAL MODEL ELEMENT REPORT ===")
    print(f"\nTotal Elements: {results['total_elements']:,}")
    print(f"Estimated Memory Usage: {results['memory_estimate']:.2f} MB")
    
    # Print global mesh settings
    global_mesh = results["global_mesh"]
    print(f"\nGlobal Mesh Settings:")
    print(f"  Nx: {global_mesh['Nx']}")
    print(f"  Ny: {global_mesh['Ny']}")
    print(f"  Nz: {global_mesh['Nz']}")
    
    print("\nElements by Region:")
    region_details = []
    for region_id, details in results["regions"].items():
        dx, dy, dz = details["element_size"]
        num_elements = details["num_elements"]
        percent = (num_elements / results["total_elements"]) * 100 if results["total_elements"] > 0 else 0
        
        # Get mesh dimensions for this region
        mesh_dims = results["mesh_dimensions"][region_id]
        
        # Get layer dimensions
        Lx, Ly, Lz = details["region_dimensions"]
        
        region_details.append({
            "id": region_id,
            "elements": num_elements,
            "percent": percent,
            "nx": mesh_dims["Nx"],
            "ny": mesh_dims["Ny"],
            "nz": mesh_dims["Nz"],
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "volume": details["region_volume"]
        })
        
    # Sort by number of elements (descending)
    region_details.sort(key=lambda x: x["elements"], reverse=True)
    
    for region in region_details:
        print(f"  {region['id']}:")
        print(f"    Elements: {region['elements']:,} ({region['percent']:.1f}%)")
        print(f"    Layer Dimensions: Lx={region['Lx']:.2f} mm, Ly={region['Ly']:.2f} mm, Lz={region['Lz']:.2f} mm")
        print(f"    Mesh Dimensions: Nx={region['nx']}, Ny={region['ny']}, Nz={region['nz']}")
        print(f"    Element Size: {region['dx']:.2f} × {region['dy']:.2f} × {region['dz']:.2f} mm")
        print(f"    Region Volume: {region['volume']:,.2f} mm³")
    
    print("\nElement Size Details:")
    for region_id, sizes in results["element_sizes"].items():
        mesh_dims = results["mesh_dimensions"][region_id]
        region_dims = results["regions"][region_id]["region_dimensions"]
        print(f"  {region_id}:")
        print(f"    Layer Dimensions: Lx={region_dims[0]:.2f} mm, Ly={region_dims[1]:.2f} mm, Lz={region_dims[2]:.2f} mm")
        print(f"    Mesh Dimensions: Nx={mesh_dims['Nx']}, Ny={mesh_dims['Ny']}, Nz={mesh_dims['Nz']}")
        print(f"    Element size: dx={sizes['dx']:.2f} mm, dy={sizes['dy']:.2f} mm, dz={sizes['dz']:.2f} mm")
    
    print("\n=== END OF REPORT ===")

def main():
    """Main function to calculate element counts."""
    # Get the path to the GEO.json file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    geo_file_path = os.path.join(script_dir, "GEO.json")
    
    # Read the geometry data
    try:
        geo_data = read_geo_file(geo_file_path)
    except Exception as e:
        print(f"Error reading geometry file: {e}")
        return 1
    
    # Create a ThermalParameters object
    params = get_parameters(geo_data)
    
    # Calculate element counts
    results = calculate_total_elements(params)
    
    # Print the report
    print_element_report(results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 