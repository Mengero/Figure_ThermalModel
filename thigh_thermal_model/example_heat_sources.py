"""
Example script to identify heat source elements in both metal and plastic layers.
"""

import os
import json
from thermal_analysis import ElementCoordinates, LayerMapping, create_layer_coordinates
from boundary_conditions import ElementBoundary, BoundaryCondition
from thermal_parameters import ThermalParameters

def read_geo_file(filepath):
    """Read the GEO.json file and parse it."""
    with open(filepath, 'r') as f:
        # Parse JSON with comments (assuming // style comments)
        content = f.read()
        # Remove single-line comments
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            comment_pos = line.find('//')
            if comment_pos >= 0:
                line = line[:comment_pos]
            cleaned_lines.append(line)
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Parse the cleaned JSON
        return json.loads(cleaned_content)

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

def main():
    # Get the path to the GEO.json file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    geo_file_path = os.path.join(script_dir, "GEO.json")
    
    # Read the geometry data
    geo_data = read_geo_file(geo_file_path)
    
    # Create a ThermalParameters object
    params = ThermalParameters(geo_data)
    
    # Create coordinate systems for metal and plastic layers
    metal_coords, plastic_coords = create_layer_coordinates(params)
    
    # Create a LayerMapping object
    layer_mapping = LayerMapping(params)
    
    # Create ElementBoundary objects for metal and plastic layers
    metal_boundary = ElementBoundary(metal_coords, layer_mapping, "metal_layer", params)
    plastic_boundary = ElementBoundary(plastic_coords, layer_mapping, "plastic_layer", params)
    
    # Label boundary conditions for both layers
    metal_boundary.label_boundary_conditions()
    plastic_boundary.label_boundary_conditions()
    
    # Analyze and print information for both layers
    analyze_layer(metal_boundary, metal_coords, "Metal")
    analyze_layer(plastic_boundary, plastic_coords, "Plastic")

if __name__ == "__main__":
    main() 