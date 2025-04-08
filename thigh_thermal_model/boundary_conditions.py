from typing import Dict, List, Tuple
import numpy as np
from thermal_analysis import ElementCoordinates, LayerMapping
from thermal_parameters import ThermalParameters

class BoundaryCondition:
    """Enum-like class for boundary condition types"""
    INNER = "inner"          # Inner element, not on any boundary
    ADIABATIC = "adiabatic"  # Surface element with no heat transfer
    MAPPED = "mapped"        # Surface element that is part of a mapping

class ElementBoundary:
    def __init__(self, coords: ElementCoordinates, mapping: LayerMapping, layer_id: str):
        """
        Initialize boundary condition labeling system.
        
        Args:
            coords: ElementCoordinates object for the layer
            mapping: LayerMapping object containing mapping information
            layer_id: Identifier string for this layer (e.g., "metal_layer", "plastic_layer")
        """
        self.coords = coords
        self.mapping = mapping
        self.layer_id = layer_id # Store the layer ID
        self.boundary_conditions = {}  # Dictionary to store boundary conditions for each element
        
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
    
    def label_boundary_conditions(self) -> Dict[int, str]:
        """
        Label boundary conditions for all elements in the layer.
        
        Returns:
            Dictionary mapping global indices to boundary condition types
        """
        for k in range(self.coords.Nz):
            for j in range(self.coords.Ny):
                for i in range(self.coords.Nx):
                    global_idx = self.coords.get_global_index(self.coords.Nx, self.coords.Ny, i, j, k)
                    
                    if not self._is_surface_element(i, j, k):
                        # Inner element
                        self.boundary_conditions[global_idx] = BoundaryCondition.INNER
                    elif self._is_mapped_element(i, j, k):
                        # Mapped surface element
                        self.boundary_conditions[global_idx] = BoundaryCondition.MAPPED
                    else:
                        # Non-mapped surface element (adiabatic by default)
                        self.boundary_conditions[global_idx] = BoundaryCondition.ADIABATIC
        
        return self.boundary_conditions
    
    def get_boundary_condition(self, global_idx: int) -> str:
        """Get the boundary condition for a specific element"""
        return self.boundary_conditions.get(global_idx, BoundaryCondition.INNER)
    
    def get_mapped_elements(self) -> List[int]:
        """Get list of global indices for all mapped elements"""
        return [idx for idx, bc in self.boundary_conditions.items() 
                if bc == BoundaryCondition.MAPPED]
    
    def get_surface_elements(self) -> List[int]:
        """Get list of global indices for all surface elements"""
        return [idx for idx, bc in self.boundary_conditions.items() 
                if bc != BoundaryCondition.INNER]
    
    def get_adiabatic_elements(self) -> List[int]:
        """Get list of global indices for all adiabatic elements"""
        return [idx for idx, bc in self.boundary_conditions.items() 
                if bc == BoundaryCondition.ADIABATIC] 