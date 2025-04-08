"""
Geometry module for the thermal model.
Handles all geometric calculations and domain setup.
"""

import numpy as np
from . import config

class GeometryModel:
    """Class to manage the geometric model of the heat sink."""
    
    def __init__(self):
        """Initialize the geometry model with dimensions from config."""
        # Calculate derived dimensions
        config.D3 = np.sqrt(config.D1**2 - (config.D2/2)**2)
        
        # Calculate pelvis dimensions based on area and aspect ratio
        ratio = config.D3 / config.D2
        config.L_PELVIS = np.sqrt(config.A_CONV / ratio)
        config.W_PELVIS = config.L_PELVIS / ratio
        
        # Actuator dimensions
        self.t_atr = 4 * 1e-3  # m
        self.r_atr = 44.65 * 1e-3  # m
        self.l_atr = np.pi * 2 * self.r_atr  # m
        self.w_atr = (53.5 - 42.7) * 1e-3  # m
        self.w_contact = (53.5 - 46.3) * 1e-3  # m
        
        # Calculate contact areas
        self.d_pelvis_spinex = np.sqrt(config.A_ATR_SPINEX)
        self.d_pelvis_hipy = np.sqrt(config.A_ATR_HIPY)
        
        # Set up domain limits
        self.setup_domain_limits()
        
        # Set up discretization
        self.setup_discretization()
        
    def setup_domain_limits(self):
        """Set up the limits of the computational domain."""
        # Pelvis domain
        self.x_lim_pelvis = np.array([0, config.W_PELVIS])
        self.y_lim_pelvis = np.array([0, config.L_PELVIS])
        self.z_lim_pelvis = np.array([0, config.T_PELVIS])
        
        # SPINEX location
        self.x_lim_spinex = np.array([
            config.W_PELVIS/2 - self.d_pelvis_spinex/2,
            config.W_PELVIS/2 + self.d_pelvis_spinex/2
        ])
        self.y_lim_spinex = np.array([0, self.d_pelvis_spinex])
        
        # Left HIPY location
        self.x_lim_hipy_l = np.array([
            config.W_PELVIS/2 - config.D2/2 - self.d_pelvis_hipy/2,
            config.W_PELVIS/2 - config.D2/2 + self.d_pelvis_hipy/2
        ])
        self.y_lim_hipy_l = np.array([
            config.L_PELVIS - self.d_pelvis_hipy,
            config.L_PELVIS
        ])
        
        # Right HIPY location
        self.x_lim_hipy_r = np.array([
            config.W_PELVIS/2 + config.D2/2 - self.d_pelvis_hipy/2,
            config.W_PELVIS/2 + config.D2/2 + self.d_pelvis_hipy/2
        ])
        self.y_lim_hipy_r = np.array([
            config.L_PELVIS - self.d_pelvis_hipy,
            config.L_PELVIS
        ])
        
        # Corner points
        self.corners = np.array([
            [0, 0, 0], [0, 0, config.T_PELVIS], 
            [0, config.L_PELVIS, 0], [0, config.L_PELVIS, config.T_PELVIS],
            [config.W_PELVIS, 0, 0], [config.W_PELVIS, 0, config.T_PELVIS], 
            [config.W_PELVIS, config.L_PELVIS, 0], [config.W_PELVIS, config.L_PELVIS, config.T_PELVIS]
        ])
        
    def setup_discretization(self):
        """Set up the discretization grid."""
        self.dx = config.W_PELVIS / config.NX
        self.dy = config.L_PELVIS / config.NY
        self.dz = config.T_PELVIS / config.NZ
        
        self.x_pts = np.linspace(0, config.W_PELVIS, config.NX)
        self.y_pts = np.linspace(0, config.L_PELVIS, config.NY)
        self.z_pts = np.linspace(0, config.T_PELVIS, config.NZ)
        
    def is_point_in_region(self, x, y, z):
        """
        Determine if a point is within a special region (SPINEX or HIPY).
        
        Parameters:
            x, y, z (float): Coordinates of the point
            
        Returns:
            int: 0 = regular region, 1 = HIPY region, 2 = SPINEX region
        """
        # Only check top surface
        if abs(z - config.T_PELVIS) < config.TOL:
            # Check if in HIPY regions
            if ((self.x_lim_hipy_l[0] <= x <= self.x_lim_hipy_l[1] and 
                 self.y_lim_hipy_l[0] <= y <= self.y_lim_hipy_l[1]) or
                (self.x_lim_hipy_r[0] <= x <= self.x_lim_hipy_r[1] and 
                 self.y_lim_hipy_r[0] <= y <= self.y_lim_hipy_r[1])):
                return 1
            
            # Check if in SPINEX region
            if (self.x_lim_spinex[0] <= x <= self.x_lim_spinex[1] and 
                self.y_lim_spinex[0] <= y <= self.y_lim_spinex[1]):
                return 2
                
        return 0
