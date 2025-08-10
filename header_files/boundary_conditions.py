from enum import Enum
from typing import Dict, List, Tuple, Set
import numpy as np
from thermal_analysis import ElementCoordinates, LayerMapping
from thermal_parameters import ThermalParameters

class BoundaryCondition(str, Enum):
    """Enumeration of possible boundary conditions"""
    INNER = "INNER"
    ADIABATIC = "ADIABATIC"
    CONST_Q = "CONST_Q"
    CONST_T = "CONST_T"
    ACTUATOR_CONNECTED = "ACTUATOR_CONNECTED"
    PLASTIC_COVERED = "PLASTIC_COVERED"
    CONVECTIVE = "CONVECTIVE"
    USERDEF_CONVECTION = "USERDEF_CONVECTION"
    MAPPED = "MAPPED"
    NODE_CONNECTED = "NODE_CONNECTED"
    USERDEF_CONDUCTION = "USERDEF_CONDUCTION"

# Define boundary condition priorities (higher number = higher priority)
BC_PRIORITIES = {
    BoundaryCondition.INNER: 0,
    BoundaryCondition.ADIABATIC: 1,
    BoundaryCondition.PLASTIC_COVERED: 2,
    BoundaryCondition.CONVECTIVE: 3,
    BoundaryCondition.USERDEF_CONVECTION: 3,  # Same priority as CONVECTIVE
    BoundaryCondition.MAPPED: 4,
    BoundaryCondition.NODE_CONNECTED: 4,  # Same priority as MAPPED
    BoundaryCondition.USERDEF_CONDUCTION: 4,  # User-defined conduction
    BoundaryCondition.CONST_Q: 5,
    BoundaryCondition.ACTUATOR_CONNECTED: 5,
    BoundaryCondition.CONST_T: 6  # Highest priority - Dirichlet boundary condition
}

class ElementBoundary:
    def __init__(self, coords: ElementCoordinates, mapping: LayerMapping, layer_id: str, params: ThermalParameters = None):
        """
        Initialize boundary condition labeling system.
        
        Args:
            coords: ElementCoordinates object for the layer
            mapping: LayerMapping object containing mapping information
            layer_id: Identifier string for this layer (e.g., "shoulder", "humerus_upper", etc.)
            params: ThermalParameters object (optional, for heat source identification)
        """
        self.coords = coords
        self.mapping = mapping
        self.layer_id = layer_id
        self.boundary_conditions = {}  # Dictionary to store boundary conditions for each element
        self.params = params
        
        # Get region data from params
        self.region_data = None
        if self.params is not None:
            for region in self.params.geo_data["regions"]:
                if region["id"] == layer_id:
                    self.region_data = region
                    break
    
    def _is_surface_element(self, i: int, j: int, k: int) -> bool:
        """Check if an element is on the surface of the layer"""
        return (k == 0 or k == self.coords.Nz - 1 or
                i == 0 or i == self.coords.Nx - 1 or
                j == 0 or j == self.coords.Ny - 1)
    
    def _get_element_position(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """
        Get the physical position (x, y, z) of an element.
        Returns:
        - x: centered coordinates (-width/2 to +width/2)
        - y: from bottom (0 to height)
        - z: from bottom (0 to thickness)
        """
        # Get raw coordinates
        x, y, z = self.coords.get_coordinates_from_3d(i, j, k)
        
        # Print debug info for shoulder region
        # if self.layer_id == "shoulder":
        #     print(f"\nElement position for ({i}, {j}, {k}):")
        #     print(f"Raw coordinates: ({x:.3f}, {y:.3f}, {z:.3f})")
        #     print(f"Region dimensions: Lx={self.coords.Lx:.3f}, Ly={self.coords.Ly:.3f}, Lz={self.coords.Lz:.3f}")
        #     print(f"Element sizes: dx={self.coords.dx:.3f}, dy={self.coords.dy:.3f}, dz={self.coords.dz:.3f}")
        
        return (x, y, z)  # x is already centered in get_coordinates_from_3d
    
    def _is_in_boundary_region(self, x: float, y: float, z: float, bc_data: dict) -> bool:
        """
        Check if a point (x, y, z) is within a boundary condition region.
        x is already in centered coordinates (-width/2 to +width/2)
        y and z start from 0
        """
        if not self.region_data:
            return False
            
        # Handle ADIABATIC boundary conditions with source regions
        bc_data_type = bc_data.get("type", "UNKNOWN")
        if bc_data_type == "ADIABATIC":
            # Accept both legacy rectangle (source_1) and new line-based format
            source = bc_data.get("source_1")
            if source:
                # Get boundary condition region dimensions
                bc_x = source["centroid"]["x"]
                bc_y = source["centroid"]["y"]
                bc_z = source.get("centroid", {}).get("z", 0)
                bc_width = source["width"]
            else:
                # For line-based adiabatic, use the midpoint for validation footprint
                p1 = bc_data.get("point1", {"x":0, "y":0})
                p2 = bc_data.get("point2", {"x":0, "y":0})
                bc_x = (p1["x"] + p2["x"]) / 2
                bc_y = (p1["y"] + p2["y"]) / 2
                bc_z = 0
                bc_width = 0
            # Height/thickness only exist for legacy source; for line, use zero extents
            if source:
                bc_height = source["height"]
                bc_thickness = source.get("thickness", 0)
            else:
                bc_height = 0
                bc_thickness = 0
            
            # Calculate region limits
            x_min = bc_x - bc_width/2
            x_max = bc_x + bc_width/2
            y_min = bc_y - bc_height/2
            y_max = bc_y + bc_height/2
            z_min = bc_z - bc_thickness/2
            z_max = bc_z + bc_thickness/2
            
            # Add a small tolerance for floating point comparison
            eps = 1e-6
            
            # Check if point is within the boundary region
            in_region = (x_min - eps <= x <= x_max + eps and 
                       y_min - eps <= y <= y_max + eps and 
                       z_min - eps <= z <= z_max + eps)
                       
            if in_region:
                return True
                
            return False
            
        # Handle standard boundary conditions (including MAPPED)
        # Get boundary condition region dimensions
        bc_x = bc_data["centroid"]["x"]  # Already relative to center
        bc_y = bc_data["centroid"]["y"]  # From bottom
        bc_z = bc_data.get("centroid", {}).get("z", 0)
        bc_width = bc_data["width"]
        bc_height = bc_data["height"]
        bc_thickness = bc_data.get("thickness", 0)  # Default to 0 if not specified
        
        # Calculate region limits
        x_min = bc_x - bc_width/2
        x_max = bc_x + bc_width/2
        y_min = bc_y - bc_height/2
        y_max = bc_y + bc_height/2
        z_min = bc_z - bc_thickness/2
        z_max = bc_z + bc_thickness/2
        
        eps = 1e-6
        # Add buffer zone for CONST_Q and ACTUATOR_CONNECTED boundary conditions
        if bc_data_type in ["CONST_Q", "ACTUATOR_CONNECTED"]:
            # Add dx/2, dy/2, dz/2 buffer to the region bounds
            x_min -= self.coords.dx/2 - eps
            x_max += self.coords.dx/2 + eps
            y_min -= self.coords.dy/2 - eps
            y_max += self.coords.dy/2 + eps
            z_min -= self.coords.dz/2 - eps
            z_max += self.coords.dz/2 + eps
        else:
            # Add a small tolerance for floating point comparison for other boundary conditions
            x_min -= eps
            x_max += eps
            y_min -= eps
            y_max += eps
            z_min -= eps
            z_max += eps
        
        # Check if point is within the boundary region
        in_region = (x_min <= x <= x_max and 
                    y_min <= y <= y_max and 
                    z_min <= z <= z_max)
        
        return in_region
    
    def label_boundary_conditions(self) -> Dict[int, List[str]]:
        """Label boundary conditions for all elements in the layer"""
        if not self.region_data:
            print(f"Warning: No region data found for {self.layer_id}")
            return {}
        
        # Initialize boundary conditions
        for k in range(self.coords.Nz):
            for j in range(self.coords.Ny):
                for i in range(self.coords.Nx):
                    global_idx = self.coords.get_global_index(self.coords.Nx, self.coords.Ny, i, j, k)
                    
                    if not self._is_surface_element(i, j, k):
                        # Inner element
                        self.boundary_conditions[global_idx] = ["INNER"]
                        continue
                    
                    # Get element position
                    x, y, z = self._get_element_position(i, j, k)
                    
                    # Initialize boundary conditions list
                    element_bcs = []
                    mapped_bc = False
                    
                    # First pass: collect all boundary conditions for this element
                    for bc in self.region_data.get("boundary_conditions", []):
                        if self._is_in_boundary_region(x, y, z, bc):
                            bc_type = bc.get("type", "UNKNOWN")
                            if bc_type == "MAPPED":
                                mapped_bc = True
                            else:
                                element_bcs.append(bc_type)
                    
                    # Sort boundary conditions by priority (excluding MAPPED)
                    element_bcs.sort(key=lambda x: BC_PRIORITIES[x], reverse=True)
                    
                    # Check for overlapping high-priority conditions
                    high_priority_bcs = [bc for bc in element_bcs if BC_PRIORITIES[bc] > BC_PRIORITIES[BoundaryCondition.PLASTIC_COVERED]]
                    if len(high_priority_bcs) > 1:
                        print(f"Warning: Multiple high-priority boundary conditions {high_priority_bcs} found for element at ({i}, {j}, {k}) in {self.layer_id}")
                    
                    # Apply priority rules
                    final_bcs = []
                    has_high_priority = False
                    
                    for bc in element_bcs:
                        if BC_PRIORITIES[bc] > BC_PRIORITIES[BoundaryCondition.PLASTIC_COVERED]:
                            if not has_high_priority:
                                final_bcs.append(bc)
                                has_high_priority = True
                        elif bc == BoundaryCondition.PLASTIC_COVERED and not has_high_priority:
                            final_bcs.append(bc)
                    
                    # If no boundary conditions found, apply defaults:
                    # - z surfaces (k=0 or k=Nz-1) are convective
                    # - x and y surfaces (i=0, i=Nx-1, j=0, j=Ny-1) are adiabatic
                    if not final_bcs:
                        if k == 0 or k == self.coords.Nz - 1:
                            final_bcs = ["CONVECTIVE"]
                        else:
                            final_bcs = ["ADIABATIC"]
                    
                    # Add MAPPED if it exists (can coexist with any other boundary condition)
                    if mapped_bc:
                        final_bcs.append("MAPPED")
                    
                    # Store the final boundary conditions
                    self.boundary_conditions[global_idx] = final_bcs
        
        return self.boundary_conditions
    
    def get_boundary_condition(self, global_idx: int) -> List[str]:
        """Get the list of boundary conditions for a specific element"""
        return self.boundary_conditions.get(global_idx, ["INNER"])
    
    def get_elements_by_condition(self, condition_type: str) -> List[int]:
        """Get list of global indices for elements with a specific boundary condition"""
        return [idx for idx, bcs in self.boundary_conditions.items() 
                if condition_type in bcs]

    def _is_mapped_element(self, i: int, j: int, k: int) -> bool:
        """Check if an element is part of any mapping relevant to this layer"""
        # Assuming the get_global_index signature is (Nx, Ny, i, j, k) based on recent context
        global_idx = self.coords.get_global_index(self.coords.Nx, self.coords.Ny, i, j, k)
        
        # Iterate through all defined mappings (e.g., 'metal_layer_1_source_to_plastic_layer_1_target')
        for mapping_id, mapped_indices in self.mapping.region_mappings.items():
            # Check if this mapping involves the current layer
            if self.layer_id in mapping_id: 
                # Check if the current element's index is in the list for this mapping
                # Using a set for faster lookup if lists are large
                if global_idx in set(mapped_indices): 
                    return True  # Element is mapped in at least one relevant mapping
        return False # Element is not mapped in any relevant mapping
    
    def _is_heat_source_element(self, i: int, j: int, k: int) -> Tuple[bool, str, str]:
        """
        Check if an element is part of any heat source.
        
        Args:
            i, j, k: Element indices
            
        Returns:
            Tuple of (is_heat_source, heat_source_id, heat_source_type)
        """
        if self.params is None:
            return False, "", ""
        
        # Get global index for this element
        global_idx = self.coords.get_global_index(self.coords.Nx, self.coords.Ny, i, j, k)
        
        # Check if heat sources have been identified already
        if not self.heat_source_elements:
            # Identify heat source elements if not done yet
            self.heat_source_elements = self.params.identify_heat_source_elements(self.coords)
        
        # First check for CONST_Qflux heat sources
        for source_id, element_indices in self.heat_source_elements.items():
            if global_idx in element_indices:
                # Get the heat source type
                source_type = "UNKNOWN"
                for source in self.params.get_region_heat_sources(self.layer_id):
                    if source.get("id") == source_id:
                        source_type = source.get("type", "UNKNOWN")
                        if source_type == "CONST_Qflux":
                            return True, source_id, source_type
                        break
        
        # Then check for CONVECTIVE and CONVECTIVE_AIRGAP heat sources
        for source_id, element_indices in self.heat_source_elements.items():
            if global_idx in element_indices:
                # Get the heat source type
                source_type = "UNKNOWN"
                for source in self.params.get_region_heat_sources(self.layer_id):
                    if source.get("id") == source_id:
                        source_type = source.get("type", "UNKNOWN")
                        if source_type in ["CONVECTIVE", "CONVECTIVE_AIRGAP"]:
                            return True, source_id, source_type
                        break
        
        return False, "", ""
    
    def get_mapped_elements(self) -> List[int]:
        """Get list of global indices for all mapped elements"""
        return [idx for idx, bc in self.boundary_conditions.items() 
                if BoundaryCondition.MAPPED in bc]
    
    def get_surface_elements(self) -> List[int]:
        """Get list of global indices for all surface elements"""
        return [idx for idx, bc in self.boundary_conditions.items() 
                if BoundaryCondition.INNER not in bc]
    
    def get_adiabatic_elements(self) -> List[int]:
        """Get list of global indices for all adiabatic elements"""
        return [idx for idx, bc in self.boundary_conditions.items() 
                if BoundaryCondition.ADIABATIC in bc]
    
    def get_heat_source_elements(self, source_type: str = None) -> List[int]:
        """
        Get list of global indices for all heat source elements.
        
        Args:
            source_type: Optional filter for specific heat source type (e.g., "CONVECTIVE", "CONST_Qflux")
            
        Returns:
            List of global indices for heat source elements
        """
        if source_type is None:
            # Return all heat source elements regardless of type
            return [idx for idx, bc in self.boundary_conditions.items() 
                    if BoundaryCondition.CONVECTIVE in bc or 
                       BoundaryCondition.CONST_Qflux in bc or 
                       BoundaryCondition.CONVECTIVE_AIRGAP in bc]
        elif source_type.upper() == "CONST_QFLUX":
            # Return only constant heat flux elements
            return [idx for idx, bc in self.boundary_conditions.items() 
                    if BoundaryCondition.CONST_Qflux in bc]
        elif source_type.upper() == "CONVECTIVE":
            # Return only convective heat source elements
            return [idx for idx, bc in self.boundary_conditions.items() 
                    if BoundaryCondition.CONVECTIVE in bc]
        elif source_type.upper() == "CONVECTIVE_AIRGAP":
            # Return only convective airgap heat source elements
            return [idx for idx, bc in self.boundary_conditions.items() 
                    if BoundaryCondition.CONVECTIVE_AIRGAP in bc]
        else:
            # Unknown heat source type
            return []
    
    def get_heat_source_elements_by_id(self, source_id: str) -> List[int]:
        """
        Get list of global indices for elements belonging to a specific heat source.
        
        Args:
            source_id: ID of the heat source (e.g., "hipx_mech", "knee_mech")
            
        Returns:
            List of global indices for elements in the specified heat source
        """
        if not self.heat_source_elements:
            # Identify heat source elements if not done yet
            if self.params is not None:
                self.heat_source_elements = self.params.identify_heat_source_elements(self.coords)
        
        # Return elements for the specified heat source ID
        return self.heat_source_elements.get(source_id, [])
    
    def get_heat_source_ids(self) -> List[str]:
        """
        Get a list of all heat source IDs for this layer.
        
        Returns:
            List of heat source IDs
        """
        if not self.heat_source_elements:
            # Identify heat source elements if not done yet
            if self.params is not None:
                self.heat_source_elements = self.params.identify_heat_source_elements(self.coords)
        
        return list(self.heat_source_elements.keys()) 