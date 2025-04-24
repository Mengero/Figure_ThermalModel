from typing import Dict, List, Tuple
import numpy as np
from thermal_analysis import ElementCoordinates, LayerMapping
from thermal_parameters import ThermalParameters

class BoundaryCondition:
    """Enum-like class for boundary condition types"""
    INNER = "inner"          # Inner element, not on any boundary
    ADIABATIC = "adiabatic"  # Surface element with no heat transfer
    MAPPED = "mapped"        # Surface element that is part of a mapping
    CONVECTIVE = "convective" # Surface element with convective heat transfer
    CONST_Qflux = "const_qflux" # Surface element with constant heat flux
    CONVECTIVE_AIRGAP = "convective_airgap" # Surface element with convective heat transfer through an air gap

class ElementBoundary:
    def __init__(self, coords: ElementCoordinates, mapping: LayerMapping, layer_id: str, params: ThermalParameters = None):
        """
        Initialize boundary condition labeling system.
        
        Args:
            coords: ElementCoordinates object for the layer
            mapping: LayerMapping object containing mapping information
            layer_id: Identifier string for this layer (e.g., "metal_layer", "plastic_layer")
            params: ThermalParameters object (optional, for heat source identification)
        """
        self.coords = coords
        self.mapping = mapping
        self.layer_id = layer_id
        self.boundary_conditions = {}  # Dictionary to store boundary conditions for each element
        self.params = params
        self.heat_source_elements = {}  # Dictionary to store heat source elements by ID
        self.combined_boundary_conditions = {}  # Dictionary to store combined boundary conditions
        
        # Identify heat sources during initialization if params is provided
        if self.params is not None:
            self.heat_source_elements = self.params.identify_heat_source_elements(self.coords)
            print(f"\nInitialized {layer_id} with {len(self.heat_source_elements)} heat sources")
            for source_id, elements in self.heat_source_elements.items():
                print(f"  Heat source '{source_id}': {len(elements)} elements")
        
    def _is_surface_element(self, i: int, j: int, k: int) -> bool:
        """Check if an element is on the surface of the layer"""
        return (k == 0 or k == self.coords.Nz - 1 or
                i == 0 or i == self.coords.Nx - 1 or
                j == 0 or j == self.coords.Ny - 1)
    
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
    
    def label_boundary_conditions(self) -> Dict[int, List[str]]:
        """
        Label boundary conditions for all elements in the layer.
        Elements can have multiple boundary conditions simultaneously.
        Priority rules:
        - MAPPED takes priority over CONVECTIVE and CONVECTIVE_AIRGAP (can't be both)
        - MAPPED and CONST_Qflux can be combined
        - CONST_Qflux and CONVECTIVE can be combined
        - CONST_Qflux and CONVECTIVE_AIRGAP can be combined
        
        Returns:
            Dictionary mapping global indices to lists of boundary condition types
        """
        # Track elements that have been assigned to check for overlaps
        assigned_elements = {}
        
        for k in range(self.coords.Nz):
            for j in range(self.coords.Ny):
                for i in range(self.coords.Nx):
                    global_idx = self.coords.get_global_index(self.coords.Nx, self.coords.Ny, i, j, k)
                    
                    if not self._is_surface_element(i, j, k):
                        # Inner element
                        self.boundary_conditions[global_idx] = [BoundaryCondition.INNER]
                        continue
                    
                    # Initialize boundary conditions list
                    boundary_conditions = []
                    
                    # Check for mapped elements first (highest priority)
                    if self._is_mapped_element(i, j, k):
                        boundary_conditions.append(BoundaryCondition.MAPPED)
                    
                    # Check for heat sources
                    is_heat_source, source_id, source_type = self._is_heat_source_element(i, j, k)
                    if is_heat_source:
                        if source_type == "CONST_Qflux":
                            boundary_conditions.append(BoundaryCondition.CONST_Qflux)
                        elif source_type == "CONVECTIVE" and BoundaryCondition.MAPPED not in boundary_conditions:
                            # Only add CONVECTIVE if not already MAPPED
                            boundary_conditions.append(BoundaryCondition.CONVECTIVE)
                        elif source_type == "CONVECTIVE_AIRGAP" and BoundaryCondition.MAPPED not in boundary_conditions:
                            # Only add CONVECTIVE_AIRGAP if not already MAPPED
                            boundary_conditions.append(BoundaryCondition.CONVECTIVE_AIRGAP)
                    
                    # If no special boundary conditions, mark as adiabatic
                    if not boundary_conditions:
                        boundary_conditions.append(BoundaryCondition.ADIABATIC)
                    
                    # Store the boundary conditions
                    self.boundary_conditions[global_idx] = boundary_conditions
                    assigned_elements[global_idx] = boundary_conditions
        
        return self.boundary_conditions

    def get_combined_boundary_condition(self, global_idx: int) -> str:
        """
        Get a combined boundary condition string for visualization.
        This is used for determining the color in plots.
        
        Returns:
            String representing the combined boundary condition
        """
        if global_idx not in self.boundary_conditions:
            return BoundaryCondition.INNER
        
        bcs = self.boundary_conditions[global_idx]
        
        # Special cases for combined boundary conditions
        if BoundaryCondition.MAPPED in bcs and BoundaryCondition.CONST_Qflux in bcs:
            return "MAPPED_CONST_QFLUX"
        elif BoundaryCondition.CONST_Qflux in bcs and BoundaryCondition.CONVECTIVE in bcs:
            return "CONST_QFLUX_CONVECTIVE"
        elif BoundaryCondition.MAPPED in bcs:
            return BoundaryCondition.MAPPED  # MAPPED takes priority
        elif len(bcs) == 1:
            return bcs[0]
        else:
            return "_".join(bcs)

    def get_boundary_condition(self, global_idx: int) -> List[str]:
        """Get the list of boundary conditions for a specific element"""
        return self.boundary_conditions.get(global_idx, [BoundaryCondition.INNER])
    
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