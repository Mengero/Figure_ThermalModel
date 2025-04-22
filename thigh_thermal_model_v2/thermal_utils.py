"""
Utility functions for thermal model processing.
"""

import json
import re
import os
from typing import Dict, Any

def read_geo_file(file_path: str) -> Dict[str, Any]:
    """
    Reads the GEO.json file, handling comments which are not valid in standard JSON.
    
    Args:
        file_path: Path to the GEO.json file
        
    Returns:
        Parsed geometry data
    """
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove comments (both // and /* */ style)
    content = re.sub(r'//.*?\n', '\n', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    # Remove trailing commas which are not valid in JSON
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*]', ']', content)
    
    # Parse the cleaned JSON
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        # Try a more lenient approach
        try:
            # Replace JSON with Python literal and evaluate
            content = content.replace('null', 'None')
            content = content.replace('true', 'True')
            content = content.replace('false', 'False')
            import ast
            data = ast.literal_eval(content)
            return data
        except Exception as e2:
            print(f"Failed to parse JSON even with lenient parsing: {e2}")
            raise e

def get_file_path(file_name: str, script_dir: str = None) -> str:
    """
    Get the absolute path to a file relative to the script directory.
    
    Args:
        file_name: Name of the file
        script_dir: Directory of the script (defaults to the directory of the current file)
        
    Returns:
        Absolute path to the file
    """
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(script_dir, file_name)

def print_section_header(title: str, width: int = 60):
    """
    Print a section header with the given title.
    
    Args:
        title: Title of the section
        width: Width of the header in characters
    """
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width)

def print_subsection_header(title: str, width: int = 60):
    """
    Print a subsection header with the given title.
    
    Args:
        title: Title of the subsection
        width: Width of the header in characters
    """
    print("\n" + "-" * width)
    print(f" {title} ".center(width))
    print("-" * width)

def format_value_with_unit(value: float, unit: str) -> str:
    """
    Format a value with its unit.
    
    Args:
        value: Value to format
        unit: Unit of the value
        
    Returns:
        Formatted string
    """
    if abs(value) >= 1000:
        return f"{value:,.2f} {unit}"
    elif abs(value) >= 100:
        return f"{value:.1f} {unit}"
    elif abs(value) >= 10:
        return f"{value:.2f} {unit}"
    elif abs(value) >= 1:
        return f"{value:.3f} {unit}"
    elif abs(value) >= 0.01:
        return f"{value:.4f} {unit}"
    else:
        return f"{value:.6e} {unit}"

def create_compatible_meshes(geo_data):
    """
    Creates compatible meshes for all regions based on mesh settings in the GEO file.
    Uses Nx and Ny from metal sheet to derive element sizes, then applies to other regions.
    
    Args:
        geo_data (dict): Geometry data from the GEO.json file
        
    Returns:
        dict: Mesh parameters for all regions
    """
    mesh_settings = geo_data.get("mesh_settings", {"Nx": 20, "Ny": 20, "Nz": 1})
    regions = geo_data.get("regions", [])
    
    # Initialize mesh parameters dictionary
    mesh_params = {}
    
    # Find the metal layer (assumed to be the first region or the one with layer_number=1)
    metal_region = next((r for r in regions if r["id"] == "metal_layer"), regions[0])
    
    # Calculate element sizes based on metal region
    metal_nx = mesh_settings.get("Nx", 20)
    metal_ny = mesh_settings.get("Ny", 20)
    metal_dx = metal_region["dimensions"]["Lx"] / metal_nx
    metal_dy = metal_region["dimensions"]["Ly"] / metal_ny
    
    print(f"Metal region element size: dx={metal_dx:.2f} mm, dy={metal_dy:.2f} mm")
    
    # Store metal region mesh parameters
    mesh_params[metal_region["id"]] = {
        "nx": metal_nx,
        "ny": metal_ny,
        "dx": metal_dx,
        "dy": metal_dy,
        "nz": mesh_settings.get("Nz", 1),
        "dz": metal_region["dimensions"]["Lz"] / mesh_settings.get("Nz", 1)
    }
    
    # Calculate mesh parameters for other regions using the same element sizes
    for region in regions:
        if region["id"] == metal_region["id"]:
            continue
        
        # Calculate number of elements based on metal element size
        region_nx = max(1, int(region["dimensions"]["Lx"] / metal_dx))
        region_ny = max(1, int(region["dimensions"]["Ly"] / metal_dy))
        
        # Calculate actual element sizes for exact fit
        region_dx = region["dimensions"]["Lx"] / region_nx
        region_dy = region["dimensions"]["Ly"] / region_ny
        
        # Store region mesh parameters
        mesh_params[region["id"]] = {
            "nx": region_nx,
            "ny": region_ny,
            "dx": region_dx,
            "dy": region_dy,
            "nz": mesh_settings.get("Nz", 1),
            "dz": region["dimensions"]["Lz"] / mesh_settings.get("Nz", 1)
        }
        
        print(f"Region {region['id']} element size: dx={region_dx:.2f} mm, dy={region_dy:.2f} mm")
    
    return mesh_params

def create_region_elements(region, mesh_params):
    """
    Creates elements for a region based on mesh parameters.
    
    Args:
        region (dict): Region data from the GEO.json file
        mesh_params (dict): Mesh parameters for the region
        
    Returns:
        list: List of elements for the region
    """
    region_id = region["id"]
    params = mesh_params[region_id]
    
    nx = params["nx"]
    ny = params["ny"]
    nz = params["nz"]
    dx = params["dx"]
    dy = params["dy"]
    dz = params["dz"]
    
    region_centroid = region["centroid"]
    region_dims = region["dimensions"]
    
    # Calculate starting points
    start_x = region_centroid["x"] - region_dims["Lx"] / 2
    start_y = region_centroid["y"] - region_dims["Ly"] / 2
    start_z = region_centroid["z"] - region_dims["Lz"] / 2
    
    elements = []
    
    # Create elements
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Calculate element centroid
                elem_x = start_x + (i + 0.5) * dx
                elem_y = start_y + (j + 0.5) * dy
                elem_z = start_z + (k + 0.5) * dz
                
                # Create element
                element = {
                    "id": f"{region_id}_elem_{i}_{j}_{k}",
                    "centroid": {"x": elem_x, "y": elem_y, "z": elem_z},
                    "dimensions": {"Lx": dx, "Ly": dy, "Lz": dz},
                    "grid_indices": {"i": i, "j": j, "k": k},
                    "region_id": region_id,
                    "layer_number": region.get("layer_number", 0),
                    "type": "REGION",
                    "material_properties": region["material_properties"]
                }
                
                elements.append(element)
    
    return elements

def process_heat_sources(region, elements, mesh_params):
    """
    Processes heat sources for a region and applies them to elements.
    
    Args:
        region (dict): Region data from the GEO.json file
        elements (list): List of elements for the region
        mesh_params (dict): Mesh parameters for the region
        
    Returns:
        list: Updated list of elements with heat source properties
    """
    if "heat_sources" not in region:
        return elements
    
    region_id = region["id"]
    params = mesh_params[region_id]
    
    # Create spatial lookup for elements
    element_lookup = {}
    for element in elements:
        i, j, k = element["grid_indices"]["i"], element["grid_indices"]["j"], element["grid_indices"]["k"]
        element_lookup[(i, j, k)] = element
    
    # Process each heat source
    for heat_source in region["heat_sources"]:
        heat_source_id = heat_source["id"]
        source_type = heat_source["type"]
        centroid = heat_source["centroid"]
        dimensions = heat_source["dimensions"]
        
        # Calculate bounds of the heat source
        x_min = centroid["x"] - dimensions["Lx"] / 2
        x_max = centroid["x"] + dimensions["Lx"] / 2
        y_min = centroid["y"] - dimensions["Ly"] / 2
        y_max = centroid["y"] + dimensions["Ly"] / 2
        
        # Find region bounds
        region_centroid = region["centroid"]
        region_dims = region["dimensions"]
        region_x_min = region_centroid["x"] - region_dims["Lx"] / 2
        region_y_min = region_centroid["y"] - region_dims["Ly"] / 2
        
        # Calculate grid indices for heat source bounds
        i_min = max(0, int((x_min - region_x_min) / params["dx"]))
        i_max = min(params["nx"] - 1, int((x_max - region_x_min) / params["dx"]))
        j_min = max(0, int((y_min - region_y_min) / params["dy"]))
        j_max = min(params["ny"] - 1, int((y_max - region_y_min) / params["dy"]))
        
        # Total area of the heat source
        heat_source_area = dimensions["Lx"] * dimensions["Ly"]
        
        # Apply heat source to overlapping elements
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                # Check all z layers (usually just one for surface heat sources)
                for k in range(params["nz"]):
                    if (i, j, k) in element_lookup:
                        element = element_lookup[(i, j, k)]
                        
                        # Calculate element bounds
                        elem_x = element["centroid"]["x"]
                        elem_y = element["centroid"]["y"]
                        elem_dx = element["dimensions"]["Lx"]
                        elem_dy = element["dimensions"]["Ly"]
                        
                        elem_x_min = elem_x - elem_dx / 2
                        elem_x_max = elem_x + elem_dx / 2
                        elem_y_min = elem_y - elem_dy / 2
                        elem_y_max = elem_y + elem_dy / 2
                        
                        # Calculate overlap area
                        overlap_x_min = max(elem_x_min, x_min)
                        overlap_x_max = min(elem_x_max, x_max)
                        overlap_y_min = max(elem_y_min, y_min)
                        overlap_y_max = min(elem_y_max, y_max)
                        
                        # Skip if no overlap
                        if overlap_x_min >= overlap_x_max or overlap_y_min >= overlap_y_max:
                            continue
                        
                        overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
                        area_ratio = overlap_area / heat_source_area
                        
                        # Update element with heat source properties
                        element["heat_source_id"] = heat_source_id
                        element["heat_source_type"] = source_type
                        
                        # Set type-specific properties
                        if source_type == "CONVECTIVE":
                            element["heat_transfer_coefficient"] = heat_source.get("heat_transfer_coefficient", 0.0)
                            element["ambient_temperature"] = heat_source.get("ambient_temperature", 25.0)
                        elif source_type == "CONST_Qflux":
                            # Scale power by area ratio
                            element["power"] = heat_source.get("power", 0.0) * area_ratio
                        elif source_type == "CONST_T":
                            element["temperature"] = heat_source.get("temperature", 25.0)
                            element["has_fixed_temperature"] = True
                        elif source_type == "CONDUCTIVE":
                            element["thermal_conductivity"] = heat_source.get("thermal_conductivity", 0.0)
                            element["distance"] = heat_source.get("distance", 0.0)
                            if "L" in heat_source:
                                element["distance"] = heat_source["L"]
    
    return elements

def create_element_mappings(geo_data, mesh_params, region_elements):
    """
    Creates mappings between elements based on the mappings in the GEO file.
    
    Args:
        geo_data (dict): Geometry data from the GEO.json file
        mesh_params (dict): Mesh parameters for all regions
        region_elements (dict): Dictionary mapping region IDs to lists of elements
        
    Returns:
        list: List of element mappings
    """
    mappings = geo_data.get("mappings", [])
    
    # Flatten elements list for easier lookup
    all_elements = []
    for elements in region_elements.values():
        all_elements.extend(elements)
    
    # Create spatial lookup for elements
    element_by_position = {}
    element_by_grid = {}
    
    for element in all_elements:
        region_id = element["region_id"]
        i, j, k = element["grid_indices"]["i"], element["grid_indices"]["j"], element["grid_indices"]["k"]
        
        # Round coordinates for lookup
        x = round(element["centroid"]["x"], 3)
        y = round(element["centroid"]["y"], 3)
        z = round(element["centroid"]["z"], 3)
        
        pos_key = (x, y, z)
        grid_key = (region_id, i, j, k)
        
        element_by_position[pos_key] = element
        element_by_grid[grid_key] = element
    
    # Create element mappings list
    element_mappings = []
    
    # Process each mapping
    for mapping in mappings:
        mapping_id = mapping["id"]
        mapping_type = mapping["type"]
        
        source_spec = mapping["source"]
        target_spec = mapping["target"]
        
        source_region_id = source_spec["region_id"]
        target_region_id = target_spec["region_id"]
        
        # Handle mappings with specific centroids
        if "centroid" in source_spec and "centroid" in target_spec:
            source_centroid = source_spec["centroid"]
            target_centroid = target_spec["centroid"]
            source_dimensions = source_spec["dimensions"]
            target_dimensions = target_spec["dimensions"]
            
            # Find region bounds
            source_region = next((r for r in geo_data["regions"] if r["id"] == source_region_id), None)
            target_region = next((r for r in geo_data["regions"] if r["id"] == target_region_id), None)
            
            if not source_region or not target_region:
                print(f"Warning: Region not found for mapping {mapping_id}")
                continue
            
            source_params = mesh_params[source_region_id]
            target_params = mesh_params[target_region_id]
            
            # Calculate normalized positions in regions
            source_region_centroid = source_region["centroid"]
            source_region_dims = source_region["dimensions"]
            source_rel_x = (source_centroid["x"] - (source_region_centroid["x"] - source_region_dims["Lx"]/2)) / source_region_dims["Lx"]
            source_rel_y = (source_centroid["y"] - (source_region_centroid["y"] - source_region_dims["Ly"]/2)) / source_region_dims["Ly"]
            
            target_region_centroid = target_region["centroid"]
            target_region_dims = target_region["dimensions"]
            target_rel_x = (target_centroid["x"] - (target_region_centroid["x"] - target_region_dims["Lx"]/2)) / target_region_dims["Lx"]
            target_rel_y = (target_centroid["y"] - (target_region_centroid["y"] - target_region_dims["Ly"]/2)) / target_region_dims["Ly"]
            
            # Calculate grid indices for source and target centroids
            source_i = int(source_rel_x * source_params["nx"])
            source_j = int(source_rel_y * source_params["ny"])
            target_i = int(target_rel_x * target_params["nx"])
            target_j = int(target_rel_y * target_params["ny"])
            
            # Calculate element bounds for source and target
            source_half_width_elements = int(source_dimensions["Lx"] / (2 * source_params["dx"]))
            source_half_height_elements = int(source_dimensions["Ly"] / (2 * source_params["dy"]))
            target_half_width_elements = int(target_dimensions["Lx"] / (2 * target_params["dx"]))
            target_half_height_elements = int(target_dimensions["Ly"] / (2 * target_params["dy"]))
            
            # Create mappings between source and target elements
            for si in range(source_i - source_half_width_elements, source_i + source_half_width_elements + 1):
                for sj in range(source_j - source_half_height_elements, source_j + source_half_height_elements + 1):
                    # Skip if outside source grid
                    if not (0 <= si < source_params["nx"] and 0 <= sj < source_params["ny"]):
                        continue
                    
                    # Calculate normalized position within source area
                    s_rel_x = (si - (source_i - source_half_width_elements)) / (2 * source_half_width_elements)
                    s_rel_y = (sj - (source_j - source_half_height_elements)) / (2 * source_half_height_elements)
                    
                    # Calculate corresponding target indices
                    ti = int(target_i - target_half_width_elements + s_rel_x * (2 * target_half_width_elements))
                    tj = int(target_j - target_half_height_elements + s_rel_y * (2 * target_half_height_elements))
                    
                    # Skip if outside target grid
                    if not (0 <= ti < target_params["nx"] and 0 <= tj < target_params["ny"]):
                        continue
                    
                    # Find source and target elements
                    source_element = element_by_grid.get((source_region_id, si, sj, 0))
                    target_element = element_by_grid.get((target_region_id, ti, tj, 0))
                    
                    if source_element and target_element:
                        # Create mapping
                        element_mapping = {
                            "id": mapping_id,
                            "type": mapping_type,
                            "source_element": source_element,
                            "target_element": target_element
                        }
                        
                        # Add type-specific properties
                        if mapping_type == "CONDUCTIVE":
                            element_mapping["thermal_conductivity"] = mapping.get("thermal_conductivity", 1.0)
                            element_mapping["distance"] = mapping.get("L", mapping.get("distance", 0.001))
                        elif mapping_type == "INTERFACE":
                            element_mapping["thermal_conductivity"] = mapping.get("thermal_conductivity", 0.1)
                        elif mapping_type == "DIRECT_CONTACT":
                            element_mapping["contact_resistance"] = mapping.get("contact_resistance", 0.0)
                        
                        element_mappings.append(element_mapping)
        
        # Handle mappings without specific centroids (map entire regions)
        else:
            source_elements = region_elements.get(source_region_id, [])
            target_elements = region_elements.get(target_region_id, [])
            
            # Map all source elements to corresponding target elements
            for source_element in source_elements:
                # Find corresponding target element
                si = source_element["grid_indices"]["i"]
                sj = source_element["grid_indices"]["j"]
                sk = source_element["grid_indices"]["k"]
                
                # Calculate normalized position in source region
                source_region = next((r for r in geo_data["regions"] if r["id"] == source_region_id), None)
                source_params = mesh_params[source_region_id]
                source_rel_x = si / source_params["nx"]
                source_rel_y = sj / source_params["ny"]
                
                # Calculate corresponding target indices
                target_params = mesh_params[target_region_id]
                ti = int(source_rel_x * target_params["nx"])
                tj = int(source_rel_y * target_params["ny"])
                
                # Find target element
                target_element = element_by_grid.get((target_region_id, ti, tj, 0))
                
                if target_element:
                    # Create mapping
                    element_mapping = {
                        "id": mapping_id,
                        "type": mapping_type,
                        "source_element": source_element,
                        "target_element": target_element
                    }
                    
                    # Add type-specific properties
                    if mapping_type == "CONDUCTIVE":
                        element_mapping["thermal_conductivity"] = mapping.get("thermal_conductivity", 1.0)
                        element_mapping["distance"] = mapping.get("L", mapping.get("distance", 0.001))
                    elif mapping_type == "INTERFACE":
                        element_mapping["thermal_conductivity"] = mapping.get("thermal_conductivity", 0.1)
                    elif mapping_type == "DIRECT_CONTACT":
                        element_mapping["contact_resistance"] = mapping.get("contact_resistance", 0.0)
                    
                    element_mappings.append(element_mapping)
    
    return element_mappings

def create_thermal_model(geo_file_path):
    """
    Creates a thermal model from a GEO.json file.
    
    Args:
        geo_file_path (str): Path to the GEO.json file
        
    Returns:
        tuple: (geo_data, mesh_params, region_elements, element_mappings)
    """
    # Read the GEO file
    geo_data = read_geo_file(geo_file_path)
    
    # Create compatible meshes
    mesh_params = create_compatible_meshes(geo_data)
    
    # Create elements for each region
    region_elements = {}
    for region in geo_data["regions"]:
        region_id = region["id"]
        elements = create_region_elements(region, mesh_params)
        elements = process_heat_sources(region, elements, mesh_params)
        region_elements[region_id] = elements
    
    # Create element mappings
    element_mappings = create_element_mappings(geo_data, mesh_params, region_elements)
    
    return geo_data, mesh_params, region_elements, element_mappings

def print_model_summary(geo_data, mesh_params, region_elements, element_mappings):
    """
    Prints a summary of the thermal model.
    
    Args:
        geo_data (dict): Geometry data from the GEO.json file
        mesh_params (dict): Mesh parameters for all regions
        region_elements (dict): Dictionary mapping region IDs to lists of elements
        element_mappings (list): List of element mappings
    """
    print("\n===== THERMAL MODEL SUMMARY =====")
    
    # Environment
    env = geo_data.get("environment", {})
    print(f"\nEnvironment:")
    print(f"  Ambient Temperature: {env.get('ambient_temperature', 25.0)} °C")
    print(f"  Heat Transfer Coefficient: {env.get('heat_transfer_coefficient', 10.0)} W/m²K")
    
    # Regions
    print(f"\nRegions:")
    for region_id, elements in region_elements.items():
        params = mesh_params[region_id]
        print(f"  {region_id}:")
        print(f"    Elements: {len(elements)}")
        print(f"    Mesh: {params['nx']}x{params['ny']}x{params['nz']}")
        print(f"    Element Size: {params['dx']:.2f}x{params['dy']:.2f}x{params['dz']:.2f} mm")
    
    # Mappings
    print(f"\nMappings:")
    mapping_counts = {}
    for mapping in element_mappings:
        mapping_id = mapping["id"]
        if mapping_id not in mapping_counts:
            mapping_counts[mapping_id] = 0
        mapping_counts[mapping_id] += 1
    
    for mapping_id, count in mapping_counts.items():
        print(f"  {mapping_id}: {count} element pairs")
    
    # Summary
    total_elements = sum(len(elements) for elements in region_elements.values())
    print(f"\nTotal Elements: {total_elements}")
    print(f"Total Mappings: {len(element_mappings)}")
    print("===============================\n")

if __name__ == "__main__":
    # Example usage
    geo_file_path = "thigh_thermal_model/GEO.json"
    geo_data, mesh_params, region_elements, element_mappings = create_thermal_model(geo_file_path)
    print_model_summary(geo_data, mesh_params, region_elements, element_mappings) 