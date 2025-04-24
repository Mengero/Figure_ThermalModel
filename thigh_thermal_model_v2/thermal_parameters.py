"""
Thermal parameters module for accessing the thermal model parameters.
This module provides structured access to all the parameters defined in the GEO.json file.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# Type aliases for clarity
RegionData = Dict[str, Any]
MappingData = Dict[str, Any]
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
        self.mappings_by_id = {m["id"]: m for m in self.get_mappings()}
        self.metal_mappings = {}  # Dictionary of lists for metal layer
        self.plastic_mappings = {}  # Dictionary of lists for plastic layer
        self.convective_mappings = {}  # Dictionary of lists for convective layer
        self.heat_sources = self._collect_all_heat_sources()
    
    def _collect_all_heat_sources(self) -> List[Dict[str, Any]]:
        """
        Collect all heat sources from all regions.
        
        Returns:
            List of dictionaries containing heat source information with region_id added
        """
        all_sources = []
        for region_id, region in self.regions_by_id.items():
            if "heat_sources" in region:
                for source in region["heat_sources"]:
                    # Make a copy of the source and add the region_id
                    source_with_region = source.copy()
                    source_with_region["region_id"] = region_id
                    all_sources.append(source_with_region)
        return all_sources
    
    def get_all_heat_sources(self) -> List[Dict[str, Any]]:
        """
        Get all heat sources from all regions.
        
        Returns:
            List of dictionaries containing heat source information
        """
        return self.heat_sources
    
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
        """Get the dimensions (Lx, Ly, Lz) of a region in mm."""
        region = self.get_region_by_id(region_id)
        if not region:
            return (0.0, 0.0, 0.0)
        
        dims = region["dimensions"]
        return (dims["Lx"], dims["Ly"], dims["Lz"])
    
    def get_region_position(self, region_id: str) -> Tuple[float, float, float]:
        """Get the position (x, y, z) of a region's origin in mm."""
        region = self.get_region_by_id(region_id)
        if not region:
            return (0.0, 0.0, 0.0)
        
        position = region.get("position", {})
        return (position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0))
    
    def get_region_bounds(self, region_id: str) -> Dict[str, float]:
        """Get the bounds of a region in mm."""
        x, y, z = self.get_region_position(region_id)
        Lx, Ly, Lz = self.get_region_dimensions(region_id)
        
        return {
            'x_min': x,
            'x_max': x + Lx,
            'y_min': y,
            'y_max': y + Ly,
            'z_min': z,
            'z_max': z + Lz
        }
    
    def get_region_material_properties(self, region_id: str) -> Dict[str, float]:
        """Get the material properties of a region."""
        region = self.get_region_by_id(region_id)
        if not region:
            return {}
        
        return region.get("material_properties", {})
    
    def get_region_thermal_conductivity(self, region_id: str) -> float:
        """Get the thermal conductivity of a region in W/mK."""
        props = self.get_region_material_properties(region_id)
        return props.get("thermal_conductivity", 0.0)
    
    def get_region_density(self, region_id: str) -> float:
        """Get the density of a region in kg/m³."""
        props = self.get_region_material_properties(region_id)
        return props.get("density", 0.0)
    
    def get_region_specific_heat(self, region_id: str) -> float:
        """Get the specific heat of a region in J/kgK."""
        props = self.get_region_material_properties(region_id)
        return props.get("specific_heat", 0.0)
    
    def get_region_thermal_diffusivity(self, region_id: str) -> float:
        """
        Calculate the thermal diffusivity of a region in m²/s.
        
        Thermal diffusivity = thermal_conductivity / (density * specific_heat)
        """
        k = self.get_region_thermal_conductivity(region_id)
        rho = self.get_region_density(region_id)
        cp = self.get_region_specific_heat(region_id)
        
        if rho <= 0 or cp <= 0:
            return 0.0
        
        # Convert from mm²/s to m²/s (divide by 1e6)
        return k / (rho * cp) / 1e6
    
    # Heat source parameters
    def get_region_heat_sources(self, region_id: str) -> List[Dict[str, Any]]:
        """Get all heat sources for a region."""
        region = self.get_region_by_id(region_id)
        if not region:
            return []
        
        return region.get("heat_sources", [])
    
    def get_heat_source_by_id(self, region_id: str, source_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific heat source by ID from a region."""
        sources = self.get_region_heat_sources(region_id)
        for source in sources:
            if source.get("id") == source_id:
                return source
        return None
    
    def get_heat_source_power(self, region_id: str, source_id: str) -> float:
        """Get the power of a heat source in W."""
        source = self.get_heat_source_by_id(region_id, source_id)
        if not source:
            return 0.0
        
        return source.get("power", 0.0)
    
    # Mapping parameters
    def get_mapping_ids(self) -> List[str]:
        """Get a list of all mapping IDs."""
        return list(self.mappings_by_id.keys())
    
    def get_mapping(self, mapping_id: str) -> Optional[MappingData]:
        """Get a mapping by ID."""
        return self.mappings_by_id.get(mapping_id)
    
    def get_mappings_by_type(self, mapping_type: str) -> List[MappingData]:
        """Get all mappings of a specific type."""
        return [mapping for mapping in self.geo_data.get("mappings", []) 
                if mapping.get("type") == mapping_type]
    
    def get_mappings_for_region(self, region_id: str, as_source: bool = True) -> List[MappingData]:
        """
        Get all mappings where the specified region is either source or target.
        
        Args:
            region_id: ID of the region to look for
            as_source: If True, look for mappings where region is the source,
                      otherwise look for mappings where region is the target
        
        Returns:
            List of mapping data where the region is source or target
        """
        mappings = []
        for mapping in self.geo_data.get("mappings", []):
            if as_source and mapping.get("source", {}).get("region_id") == region_id:
                mappings.append(mapping)
            elif not as_source and mapping.get("target", {}).get("region_id") == region_id:
                mappings.append(mapping)
        return mappings
    
    def get_mapping_thermal_conductivity(self, mapping_id: str) -> float:
        """Get the thermal conductivity of a mapping in W/mK."""
        mapping = self.get_mapping(mapping_id)
        if not mapping:
            return 0.0
        
        return mapping.get("thermal_conductivity", 0.0)
    
    def get_mapping_distance(self, mapping_id: str) -> float:
        """Get the distance (L) of a mapping in mm."""
        mapping = self.get_mapping(mapping_id)
        if not mapping:
            return 0.0
        
        # Try to get L first, then fall back to distance
        return mapping.get("L", mapping.get("distance", 0.0))
    
    # Mesh parameters
    def get_mesh_settings(self) -> Dict[str, int]:
        """Get the mesh settings."""
        return self.geo_data.get("mesh_settings", {"Nx": 20, "Ny": 20, "Nz": 1})
    
    def get_mesh_dimensions(self) -> Tuple[int, int, int]:
        """Get the mesh dimensions (Nx, Ny, Nz)."""
        settings = self.get_mesh_settings()
        return (settings.get("Nx", 20), settings.get("Ny", 20), settings.get("Nz", 1))
    
    # Advanced calculations
    def calculate_element_size(self, region_id: str, mesh_dims: Optional[Tuple[int, int, int]] = None) -> Tuple[float, float, float]:
        """
        Calculate the element size for a region based on mesh settings.
        
        Args:
            region_id: ID of the region
            mesh_dims: Optional mesh dimensions to use instead of the global settings
        
        Returns:
            Tuple of (dx, dy, dz) in mm
        """
        if mesh_dims is None:
            mesh_dims = self.get_mesh_dimensions()
        
        Nx, Ny, Nz = mesh_dims
        Lx, Ly, Lz = self.get_region_dimensions(region_id)
        
        dx = Lx / Nx if Nx > 0 else 0.0
        dy = Ly / Ny if Ny > 0 else 0.0
        dz = Lz / Nz if Nz > 0 else 0.0
        
        return (dx, dy, dz)
    
    def calculate_element_volume(self, region_id: str, mesh_dims: Optional[Tuple[int, int, int]] = None) -> float:
        """
        Calculate the volume of an element for a region based on mesh settings.
        
        Args:
            region_id: ID of the region
            mesh_dims: Optional mesh dimensions to use instead of the global settings
        
        Returns:
            Element volume in mm³
        """
        dx, dy, dz = self.calculate_element_size(region_id, mesh_dims)
        return dx * dy * dz
    
    def calculate_timestep_limit(self, region_id: str, safety_factor: float = 0.5) -> float:
        """
        Calculate the maximum timestep for stability in seconds.
        
        For explicit thermal solvers, the timestep is limited by the diffusion stability criterion:
        dt <= safety_factor * min(dx², dy², dz²) / (2 * alpha * dim)
        
        where alpha is the thermal diffusivity and dim is the number of dimensions (1, 2, or 3)
        
        Args:
            region_id: ID of the region
            safety_factor: Safety factor to apply to the maximum timestep (0-1)
        
        Returns:
            Maximum stable timestep in seconds
        """
        dx, dy, dz = self.calculate_element_size(region_id)
        alpha = self.get_region_thermal_diffusivity(region_id)
        
        if alpha <= 0:
            return float('inf')
        
        # Consider only non-zero dimensions for the stability calculation
        dims = sum(1 for d in (dx, dy, dz) if d > 0)
        if dims == 0:
            return float('inf')
        
        # Calculate minimum grid spacing squared
        min_d_squared = min(d*d for d in (dx, dy, dz) if d > 0)
        
        # Convert from mm² to m² (divide by 1e6)
        min_d_squared_m = min_d_squared / 1e6
        
        # Calculate maximum timestep
        dt_max = safety_factor * min_d_squared_m / (2 * alpha * dims)
        
        return dt_max
    
    def calculate_characteristic_time(self, region_id: str) -> float:
        """
        Calculate the characteristic time for heat diffusion across the region in seconds.
        
        The characteristic time is approximately L² / alpha, where L is the 
        smallest dimension and alpha is the thermal diffusivity.
        
        Args:
            region_id: ID of the region
        
        Returns:
            Characteristic time in seconds
        """
        Lx, Ly, Lz = self.get_region_dimensions(region_id)
        alpha = self.get_region_thermal_diffusivity(region_id)
        
        if alpha <= 0:
            return float('inf')
        
        # Use the smallest non-zero dimension
        dims = [d for d in (Lx, Ly, Lz) if d > 0]
        if not dims:
            return float('inf')
        
        min_L = min(dims)
        
        # Convert from mm² to m² (divide by 1e6)
        min_L_squared_m = (min_L * min_L) / 1e6
        
        # Calculate characteristic time
        tau = min_L_squared_m / alpha
        
        return tau

    def get_environment(self) -> Dict[str, Any]:
        """Get environment parameters."""
        return self.geo_data.get("environment", {})
    
    def get_initial_conditions(self) -> Dict[str, Any]:
        """Get initial conditions."""
        return self.geo_data.get("initial_conditions", {})
    
    def get_regions(self) -> List[Dict[str, Any]]:
        """Get all regions."""
        return self.geo_data.get("regions", [])
    
    def get_surfaces(self) -> List[Dict[str, Any]]:
        """Get all surfaces."""
        return self.geo_data.get("surfaces", [])
    
    def get_mappings(self) -> List[Dict[str, Any]]:
        """Get all mappings."""
        return self.geo_data.get("mappings", [])
    
    def get_surface_by_id(self, surface_id: str) -> Dict[str, Any]:
        """Get a specific surface by its ID."""
        for surface in self.get_surfaces():
            if surface["id"] == surface_id:
                return surface
        raise ValueError(f"Surface with ID {surface_id} not found")
    
    def get_mapping_by_ids(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Get a specific mapping by source and target region IDs."""
        for mapping in self.get_mappings():
            if (mapping["source"]["region_id"] == source_id and 
                mapping["target"]["region_id"] == target_id):
                return mapping
        raise ValueError(f"Mapping from {source_id} to {target_id} not found")
    
    def get_mapping_bounds(self, mapping_id: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get the bounds of both source and target regions in a mapping."""
        mapping = self.mappings_by_id.get(mapping_id)
        if not mapping:
            raise ValueError(f"Mapping {mapping_id} not found")
        
        source = mapping["source"]
        target = mapping["target"]
        
        source_bounds = {
            'x_min': source["x"],
            'x_max': source["x"] + source["dimensions"]["Lx"],
            'y_min': source["y"],
            'y_max': source["y"] + source["dimensions"]["Ly"],
            'z_min': source["z"],
            'z_max': source["z"] + source["dimensions"]["Lz"]
        }
        
        target_bounds = {
            'x_min': target["x"],
            'x_max': target["x"] + target["dimensions"]["Lx"],
            'y_min': target["y"],
            'y_max': target["y"] + target["dimensions"]["Ly"],
            'z_min': target["z"],
            'z_max': target["z"] + target["dimensions"]["Lz"]
        }
        
        return source_bounds, target_bounds

    def get_convective_mappings(self) -> Dict[str, Any]:
        """
        Get all convective mappings.
        
        Returns:
            Dictionary of convective mappings with their associated data
        """
        return self.convective_mappings
    
    def identify_heat_source_elements(self, coords) -> Dict[str, List[int]]:
        """
        Find elements that belong to heat source regions for a specific coordinate system.
        
        Args:
            coords: ElementCoordinates object for the layer
            
        Returns:
            Dictionary mapping heat source IDs to lists of global indices
        """
        heat_source_elements = {}
        
        # Get heat sources for this region
        heat_sources = []
        region_id = coords.region_id
        region = self.get_region_by_id(region_id)
        
        if region and "heat_sources" in region:
            for source in region["heat_sources"]:
                source_copy = source.copy()
                source_copy["region_id"] = region_id
                heat_sources.append(source_copy)
        
        if not heat_sources:
            print(f"No heat sources found for region {region_id}")
            return heat_source_elements
        
        print(f"\nFound {len(heat_sources)} heat sources in region {region_id}")
        
        # Small tolerance for floating-point comparisons
        tolerance = 1e-6
        
        # Track which elements have been assigned to heat sources
        assigned_elements = set()
        
        # Process each heat source
        for source in heat_sources:
            source_id = source.get('id', f"source_{len(heat_source_elements)}")
            source_type = source.get('type', 'UNKNOWN')
            
            print(f"Processing heat source: {source_id} (type: {source_type}) in region {region_id}")
            
            # Get heat source position and dimensions
            centroid = source.get('centroid', {})
            dimensions = source.get('dimensions', {})
            
            # Calculate heat source bounds using centroid and dimensions
            x_min = centroid.get('x', 0) - dimensions.get('Lx', 0) / 2
            x_max = centroid.get('x', 0) + dimensions.get('Lx', 0) / 2
            y_min = centroid.get('y', 0) - dimensions.get('Ly', 0) / 2
            y_max = centroid.get('y', 0) + dimensions.get('Ly', 0) / 2
            z_min = centroid.get('z', 0) - dimensions.get('Lz', 0) / 2
            z_max = centroid.get('z', 0) + dimensions.get('Lz', 0) / 2
            
            # Calculate element sizes
            dx = coords.Lx / coords.Nx
            dy = coords.Ly / coords.Ny
            
            print(f"  Heat source bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}], z=[{z_min:.2f}, {z_max:.2f}]")
            print(f"  Element sizes: dx={dx:.2f}, dy={dy:.2f}")
            
            # Initialize list for this heat source
            heat_source_elements[source_id] = []
            
            # Find all elements that fall within the heat source bounds
            for k in range(coords.Nz):
                for j in range(coords.Ny):
                    for i in range(coords.Nx):
                        # Get coordinates of this element
                        x, y, z = coords.get_coordinates_from_3d(i, j, k)
                        
                        # Check if element is within heat source bounds
                        # Expand bounds by half element size in x and y directions
                        if (x_min - dx/2 - tolerance <= x <= x_max + dx/2 + tolerance and
                            y_min - dy/2 - tolerance <= y <= y_max + dy/2 + tolerance and
                            z_min - tolerance <= z <= z_max + tolerance):
                            
                            # For convective and const_qflux elements, they must be on the correct surface
                            if source_type == "CONVECTIVE" or source_type == "CONST_Qflux" or source_type == "CONVECTIVE_AIRGAP":
                                # Check if this is a surface element (top or bottom surface)
                                if not (k == 0 or k == coords.Nz - 1):
                                    continue
                                
                                # For z level in centroid, make sure we're on the right surface
                                target_z = centroid.get('z', 0)
                                if abs(z - target_z) > tolerance:
                                    continue
                            
                            # Add global index to list
                            global_idx = coords.get_global_index(coords.Nx, coords.Ny, i, j, k)
                            
                            # Check if element was already assigned to another heat source
                            if global_idx in assigned_elements:
                                # Get the previous heat source type
                                prev_sources = [src_id for src_id, elems in heat_source_elements.items() if global_idx in elems]
                                prev_source = self.get_heat_source_by_id(region_id, prev_sources[0])
                                prev_type = prev_source.get('type', 'UNKNOWN')
                                
                                # Only warn if not reassigning from convective to constant Q flux
                                if not ((prev_type == "CONVECTIVE" and source_type == "CONST_Qflux") or (prev_type == "CONVECTIVE_AIRGAP" and source_type == "CONST_Qflux")):
                                    print(f"WARNING: Element {global_idx} is assigned to multiple heat sources!")
                                    print(f"  Previously assigned to: {prev_sources}")
                                    print(f"  Now being assigned to: {source_id}")
                            else:
                                assigned_elements.add(global_idx)
                            heat_source_elements[source_id].append(global_idx)
            
            print(f"  Found {len(heat_source_elements[source_id])} elements in heat source {source_id}")
        
        return heat_source_elements
    
    def get_heat_source_by_type(self, source_type: str) -> Dict[str, Any]:
        """
        Get heat source parameters for a specific type.
        
        Args:
            source_type: Type of heat source (e.g., "CONVECTIVE", "CONST_Qflux", "CONVECTIVE_AIRGAP")
            
        Returns:
            Dictionary containing heat source parameters
        """
        if source_type.upper() == "CONVECTIVE":
            return {
                "h": self.h_conv,
                "T_inf": self.T_inf,
                "elements": self.heat_source_elements["CONVECTIVE"]
            }
        elif source_type.upper() == "CONST_QFLUX":
            return {
                "q_flux": self.q_flux,
                "elements": self.heat_source_elements["CONST_Qflux"]
            }
        elif source_type.upper() == "CONVECTIVE_AIRGAP":
            return {
                "h": self.h_conv_airgap,
                "T_inf": self.T_inf_airgap,
                "elements": self.heat_source_elements["CONVECTIVE_AIRGAP"]
            }
        else:
            raise ValueError(f"Unknown heat source type: {source_type}")
    
    def get_all_heat_sources_by_type(self, source_type: str) -> List[Dict[str, Any]]:
        """
        Get all heat sources of a specific type across all regions.
        
        Args:
            source_type: Type of heat source (e.g., "CONVECTIVE", "CONST_Qflux")
            
        Returns:
            List of heat sources matching the type
        """
        all_sources = []
        
        for region_id in self.get_region_ids():
            sources = self.get_heat_source_by_type(source_type)
            for source in sources:
                source_copy = source.copy()
                source_copy["region_id"] = region_id
                all_sources.append(source_copy)
        
        return all_sources

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
        print(f"  Thermal conductivity: {params.get_region_thermal_conductivity(region_id)} W/mK")
        print(f"  Density: {params.get_region_density(region_id)} kg/m³")
        print(f"  Specific heat: {params.get_region_specific_heat(region_id)} J/kgK")
        print(f"  Thermal diffusivity: {params.get_region_thermal_diffusivity(region_id):.2e} m²/s")
        
        # Calculate timestep and characteristic time
        dt_max = params.calculate_timestep_limit(region_id)
        tau = params.calculate_characteristic_time(region_id)
        
        print(f"  Max stable timestep: {dt_max:.2e} s")
        print(f"  Characteristic time: {tau:.2f} s")
        
        # Print heat sources
        heat_sources = params.get_region_heat_sources(region_id)
        if heat_sources:
            print(f"  Heat sources: {len(heat_sources)}")
            for source in heat_sources[:2]:  # Show first 2 heat sources
                source_id = source.get("id", "unknown")
                source_type = source.get("type", "unknown")
                print(f"    {source_id} ({source_type})")
                
                if source_type == "CONST_Qflux":
                    power = params.get_heat_source_power(region_id, source_id)
                    print(f"      Power: {power} W")
            
            if len(heat_sources) > 2:
                print(f"    ... and {len(heat_sources) - 2} more heat sources")
    
    # Print mapping information
    mapping_ids = params.get_mapping_ids()
    print(f"\nMappings: {len(mapping_ids)}")
    
    # Group mappings by type
    mapping_types = set(params.get_mapping(mid).get("type", "") for mid in mapping_ids)
    
    for mapping_type in mapping_types:
        mappings = params.get_mappings_by_type(mapping_type)
        print(f"\n  {mapping_type} mappings: {len(mappings)}")
        
        for mapping in mappings[:2]:  # Show first 2 mappings of each type
            mapping_id = mapping.get("id", "unknown")
            source_region = mapping.get("source", {}).get("region_id", "unknown")
            target_region = mapping.get("target", {}).get("region_id", "unknown")
            
            print(f"    {mapping_id}: {source_region} → {target_region}")
            
            if mapping_type == "CONDUCTIVE":
                tc = params.get_mapping_thermal_conductivity(mapping_id)
                dist = params.get_mapping_distance(mapping_id)
                print(f"      Thermal conductivity: {tc} W/mK")
                print(f"      Distance: {dist} mm")
        
        if len(mappings) > 2:
            print(f"    ... and {len(mappings) - 2} more {mapping_type} mappings") 