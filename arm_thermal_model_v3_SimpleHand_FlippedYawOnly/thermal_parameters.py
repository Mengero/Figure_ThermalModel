"""
Thermal parameters module for accessing the thermal model parameters.
This module provides structured access to all the parameters defined in the GEO.json file.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# Type aliases for clarity
RegionData = Dict[str, Any]
ElementData = Dict[str, Any]

class ThermalParameters:
    """Class for accessing thermal parameters from the parsed JSON data."""
    
    def __init__(self, geo_data: Dict[str, Any]):
        """
        Initialize with geometry data from JSON file.
        
        Args:
            geo_data: Dictionary containing geometry data
        """
        self.geo_data = geo_data
        self.regions_by_id = {r["id"]: r for r in self.get_regions()}
        
        # Print region and boundary condition information
        print("\nRegion and Boundary Condition Information:")
        print("=" * 50)
        for region_id, region in self.regions_by_id.items():
            print(f"\nRegion: {region_id}")
            print(f"  Centroid: x={region['centroid']['x']:.3f}, y={region['centroid']['y']:.3f}, z={region['centroid'].get('z', 0):.3f}")
            print(f"  Size: {region['width']}x{region['height']}x{region['thickness']} mm")
            print(f"  Region Limits:")
            x_min = region['centroid']['x'] - region['width']/2
            x_max = region['centroid']['x'] + region['width']/2
            y_min = region['centroid']['y'] - region['height']/2
            y_max = region['centroid']['y'] + region['height']/2
            print(f"    x: {x_min:.3f} to {x_max:.3f} mm")
            print(f"    y: {y_min:.3f} to {y_max:.3f} mm")
            print(f"    z: 0 to {region['thickness']} mm")
            
            if "boundary_conditions" in region:
                print("\n  Boundary Conditions:")
                for bc in region["boundary_conditions"]:
                    print(f"\n    Type: {bc['type']}")
                    
                    if bc['type'] == 'ADIABATIC':
                        # Handle ADIABATIC boundary conditions with source_1
                        source = bc['source_1']
                        print(f"    Source 1:")
                        print(f"      Centroid: x={source['centroid']['x']:.3f}, y={source['centroid']['y']:.3f}, z={source['centroid'].get('z', 0):.3f}")
                        print(f"      Size: {source.get('width', 0)}x{source.get('height', 0)}x{source.get('thickness', 0)} mm")
                        x_min = source['centroid']['x'] - source.get('width', 0)/2
                        x_max = source['centroid']['x'] + source.get('width', 0)/2
                        y_min = source['centroid']['y'] - source.get('height', 0)/2
                        y_max = source['centroid']['y'] + source.get('height', 0)/2
                        z_min = source['centroid'].get('z', 0) - source.get('thickness', 0)/2
                        z_max = source['centroid'].get('z', 0) + source.get('thickness', 0)/2
                        print(f"      Limits:")
                        print(f"        x: {x_min:.3f} to {x_max:.3f} mm")
                        print(f"        y: {y_min:.3f} to {y_max:.3f} mm")
                        print(f"        z: {z_min:.3f} to {z_max:.3f} mm")
                    else:
                        # Handle standard boundary conditions with direct centroid
                        if 'centroid' in bc:
                            print(f"    Centroid: x={bc['centroid']['x']:.3f}, y={bc['centroid']['y']:.3f}, z={bc['centroid'].get('z', 0):.3f}")
                            print(f"    Size: {bc.get('width', 0)}x{bc.get('height', 0)}x{bc.get('thickness', 0)} mm")
                            x_min = bc['centroid']['x'] - bc.get('width', 0)/2
                            x_max = bc['centroid']['x'] + bc.get('width', 0)/2
                            y_min = bc['centroid']['y'] - bc.get('height', 0)/2
                            y_max = bc['centroid']['y'] + bc.get('height', 0)/2
                            z_min = bc['centroid'].get('z', 0) - bc.get('thickness', 0)/2
                            z_max = bc['centroid'].get('z', 0) + bc.get('thickness', 0)/2
                            print(f"    Limits:")
                            print(f"      x: {x_min:.3f} to {x_max:.3f} mm")
                            print(f"      y: {y_min:.3f} to {y_max:.3f} mm")
                            print(f"      z: {z_min:.3f} to {z_max:.3f} mm")
                            
                            # Print additional parameters based on boundary condition type
                            if bc['type'] == 'PLASTIC_COVERED':
                                print(f"    Plastic Thickness: {bc.get('plastic_thickness', 0)} mm")
                            elif bc['type'] == 'CONST_Q':
                                print(f"    Heat Flux: {bc.get('q', 0)} W")
                            elif bc['type'] == 'ACTUATOR_CONNECTED':
                                print(f"    Actuator ID: {bc.get('actuator_id', '')}")
                                print(f"    Connection Location: {bc.get('connecting_location', '')}")
        
        # Store mappings by ID for easy lookup
        self.mappings_by_id = {}
        for region in geo_data.get("regions", []):
            for bc in region.get("boundary_conditions", []):
                if bc["type"] == "MAPPED":
                    mapping_id = bc.get("id", f"{region['id']}_mapping_{len(self.mappings_by_id)}")
                    self.mappings_by_id[mapping_id] = {
                        "source": {
                            "region_id": region["id"],
                            "centroid": bc["centroid"],
                            "dimensions": {
                                "Lx": bc["width"],
                                "Ly": bc["height"],
                                "Lz": bc["thickness"]
                            }
                        }
                    }
        
        # Store heat sources for easy lookup
        self.heat_sources = []
        for region in geo_data.get("regions", []):
            for bc in region.get("boundary_conditions", []):
                if bc["type"] in ["CONST_Q", "CONVECTIVE", "CONVECTIVE_AIRGAP"]:
                    source_id = bc.get("id", f"{region['id']}_{bc['type']}_{len(self.heat_sources)}")
                    self.heat_sources.append({
                        "id": source_id,
                        "type": bc["type"],
                        "region_id": region["id"],
                        **bc  # Include all other parameters from the boundary condition
                    })
    
    # Environment parameters
    def get_ambient_temperature(self) -> float:
        """Get the ambient temperature in degrees C."""
        return self.geo_data.get("environment", {}).get("ambient_temperature", 25.0)
    
    def get_heat_transfer_coefficient(self) -> float:
        """Get the heat transfer coefficient in W/m²K."""
        return self.geo_data.get("environment", {}).get("heat_transfer_coefficient", 10.0)
    
    def get_initial_temperature(self) -> float:
        """Get the initial temperature in degrees C."""
        return self.geo_data.get("initial_conditions", {}).get("temperature", 25.0)
    
    # Region parameters
    def get_region_ids(self) -> List[str]:
        """Get a list of all region IDs."""
        return list(self.regions_by_id.keys())
    
    def get_region_by_id(self, region_id: str) -> Dict[str, Any]:
        """Get a specific region by its ID."""
        return self.regions_by_id.get(region_id)
    
    def get_region_dimensions(self, region_id: str) -> Tuple[float, float, float]:
        """
        Get the dimensions (Lx, Ly, Lz) of a region in mm.
        
        Args:
            region_id: ID of the region to get dimensions for
            
        Returns:
            Tuple of (Lx, Ly, Lz) in mm
        """
        region = self.get_region_by_id(region_id)
        if not region:
            raise ValueError(f"Region '{region_id}' not found in geometry data")
        
        # Get dimensions from width, height, and thickness
        Lx = region["width"]  # width in x direction
        Ly = region["height"]  # height in y direction
        Lz = region["thickness"]  # thickness in z direction
        
        return Lx, Ly, Lz
    
    def get_region_position(self, region_id: str) -> Tuple[float, float, float]:
        """
        Get the position (x, y, z) of a region's origin in mm.
        The origin is at the center of the bottom line:
        - x=0 at the center of the width
        - y=0 at the bottom of the height
        - z=0 at the bottom of the thickness
        
        Args:
            region_id: ID of the region to get position for
            
        Returns:
            Tuple of (x, y, z) coordinates in mm
        """
        region = self.get_region_by_id(region_id)
        if not region:
            raise ValueError(f"Region '{region_id}' not found in geometry data")
        
        # Get region centroid and dimensions
        centroid = region["centroid"]
        width = region["width"]
        height = region["height"]
        thickness = region["thickness"]
        
        # Calculate origin position:
        # x: centroid.x is at center, so no adjustment needed
        # y: move up from centroid.y by half the height to get to bottom
        # z: move up from centroid.z by half the thickness to get to bottom
        x = centroid["x"]  # Already at center
        y = centroid["y"] - height/2  # Move to bottom
        z = centroid["z"] - thickness/2  # Move to bottom
        
        return (x, y, z)
    
    def get_region_bounds(self, region_id: str) -> Dict[str, float]:
        """
        Get the bounds of a region in mm.
        Based on the origin at the center of the bottom line:
        - x bounds are symmetric around x=0
        - y bounds start from y=0
        - z bounds start from z=0
        
        Args:
            region_id: ID of the region to get bounds for
            
        Returns:
            Dictionary with x_min, x_max, y_min, y_max, z_min, z_max values
        """
        region = self.get_region_by_id(region_id)
        if not region:
            raise ValueError(f"Region '{region_id}' not found in geometry data")
        
        # Get region dimensions
        width = region["width"]
        height = region["height"]
        thickness = region["thickness"]
        
        # Get origin position
        x, y, z = self.get_region_position(region_id)
        
        return {
            'x_min': x - width/2,  # Half width to left of center
            'x_max': x + width/2,  # Half width to right of center
            'y_min': y,            # Starting from bottom (y=0)
            'y_max': y + height,   # Full height up
            'z_min': z,            # Starting from bottom (z=0)
            'z_max': z + thickness # Full thickness up
        }
    
    # Mesh parameters
    def get_mesh_settings(self, region_id: str = None) -> Dict[str, float]:
        """
        Get the mesh settings. If region_id is provided, get region-specific settings.
        If region doesn't have specific settings, fall back to global settings.
        
        Args:
            region_id: Optional ID of the region to get mesh settings for
            
        Returns:
            Dictionary with dx, dy, dz settings in mm
        """
        # Default mesh settings
        default_settings = {"dx": 10, "dy": 10, "dz": 0.5}
        
        if region_id is not None:
            # Get region data
            region = self.get_region_by_id(region_id)
            if region and "mesh_settings" in region:
                # Return region-specific settings
                return region["mesh_settings"]
        
        # Fall back to global settings if no region-specific settings found
        return self.geo_data.get("mesh_settings", default_settings)
    
    def get_mesh_dimensions(self) -> Tuple[float, float, float]:
        """Get the mesh element sizes (dx, dy, dz) in mm."""
        settings = self.get_mesh_settings()
        return (settings.get("dx", 10), settings.get("dy", 10), settings.get("dz", 0.5))
    
    def get_environment(self) -> Dict[str, Any]:
        """Get environment parameters."""
        return self.geo_data.get("environment", {})
    
    def get_initial_conditions(self) -> Dict[str, Any]:
        """Get initial conditions."""
        return self.geo_data.get("initial_conditions", {})
    
    def get_regions(self) -> List[Dict[str, Any]]:
        """Get all regions."""
        return self.geo_data.get("regions", [])

def get_parameters(geo_data: Dict[str, Any]) -> ThermalParameters:
    """
    Create a ThermalParameters object from parsed geometry data.
    
    Args:
        geo_data: Parsed geometry data from GEO.json
    
    Returns:
        ThermalParameters object for accessing the thermal parameters
    """
    return ThermalParameters(geo_data)


# Example usage
if __name__ == "__main__":
    import os
    from thermal_utils import read_geo_file
    
    # Get the path to the GEO.json file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    geo_file_path = os.path.join(script_dir, "GEO.json")
    
    # Read the geometry data
    geo_data = read_geo_file(geo_file_path)
    
    # Create a ThermalParameters object
    params = get_parameters(geo_data)
    
    # Print some parameters
    print(f"Ambient temperature: {params.get_ambient_temperature()} °C")
    print(f"Initial temperature: {params.get_initial_temperature()} °C")
    
    # Print region information
    region_ids = params.get_region_ids()
    print(f"\nRegions: {', '.join(region_ids)}")
    
    for region_id in region_ids:
        print(f"\nRegion: {region_id}")
        print(f"  Dimensions: {params.get_region_dimensions(region_id)} mm")
        bounds = params.get_region_bounds(region_id)
        print(f"  Bounds:")
        print(f"    x: {bounds['x_min']:.3f} to {bounds['x_max']:.3f} mm")
        print(f"    y: {bounds['y_min']:.3f} to {bounds['y_max']:.3f} mm")
        print(f"    z: {bounds['z_min']:.3f} to {bounds['z_max']:.3f} mm") 