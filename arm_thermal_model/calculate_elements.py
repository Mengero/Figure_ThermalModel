#!/usr/bin/env python3
"""
Calculate the total number of elements in the thermal model and provide a breakdown by region.
This script uses the ThermalParameters class to access the model parameters and calculate element counts.
"""

import os
import sys
import numpy as np
from thermal_utils import read_geo_file
from thermal_parameters import get_parameters, ThermalParameters
from typing import Tuple

def calculate_region_elements(params: ThermalParameters, region_id: str):
    """
    Calculate the number of elements and spacing for a region.
    
    Args:
        params: ThermalParameters object
        region_id: ID of the region
        
    Returns:
        Tuple containing:
        - total_elements: Total number of elements
        - Nx: Number of elements in x direction
        - Ny: Number of elements in y direction
        - Nz: Number of elements in z direction
        - dx: Nominal element size in x direction
        - dy: Nominal element size in y direction
        - dz: Nominal element size in z direction
        - element_volume: Volume of each element
    """
    # Get region dimensions
    Lx, Ly, Lz = params.get_region_dimensions(region_id)
    
    # Get nominal mesh settings
    mesh_settings = params.get_mesh_settings()
    nominal_dx = mesh_settings['dx']
    nominal_dy = mesh_settings['dy']
    nominal_dz = mesh_settings['dz']
    
    # Calculate number of elements needed (minimum 2 elements in each direction)
    Nx = max(2, int(np.ceil(Lx / nominal_dx)) + 1)
    Ny = max(2, int(np.ceil(Ly / nominal_dy)) + 1)
    Nz = max(2, int(np.ceil(Lz / nominal_dz)) + 1)
    
    # Calculate total number of elements
    total_elements = Nx * Ny * Nz
    
    # Calculate element volume (using nominal sizes for now)
    element_volume = nominal_dx * nominal_dy * nominal_dz
    
    return total_elements, Nx, Ny, Nz, nominal_dx, nominal_dy, nominal_dz, element_volume

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