"""
Configuration module for the thermal model.
Contains all physical constants, material properties, and simulation parameters.
"""

# Material properties
K_AL = 120  # Thermal conductivity of aluminum, W/(m·K)
H = 5       # Heat transfer coefficient, W/(m²·K)
T_INF = 30  # Ambient temperature, °C
T_WALL = 70 # Wall temperature, °C

# Geometry - Areas
A_CONV = 57602.8753 * 1e-6        # m², convectional surface area
A_ATR_SPINEX = 2120.4774 * 1e-6   # m², SPINEX contacting area
A_ATR_HIPY = 1791.2303 * 1e-6     # m², HIPY contacting area

# Heat source parameters
Q_ATR_SPINEX = 0.5  # Heat generation at SPINEX, W
Q_ATR_HIPY = 20     # Heat generation at HIPY, W

# Heat flux calculations
Q_FLUX_ATR_SPINEX = Q_ATR_SPINEX / A_ATR_SPINEX  # W/m²
Q_FLUX_ATR_HIPY = Q_ATR_HIPY / A_ATR_HIPY        # W/m²

# Geometry - Distances
D1 = 120.7812 * 1e-3  # m, relative distance between SpineX and HIPY
D2 = 100 * 1e-3       # m, relative distance between 2 HIPYs
D3 = None             # Will be calculated in geometry module

# Pelvis dimensions
L_PELVIS = None  # Will be calculated in geometry module
W_PELVIS = None  # Will be calculated in geometry module
T_PELVIS = 2 * 1e-3  # m, thickness

# Mesh parameters
NX = 20  # Number of cells in x-direction
NY = 20  # Number of cells in y-direction
NZ = 10  # Number of cells in z-direction

# Solver parameters
TOLERANCE = 1e-6  # Convergence tolerance
MAX_ITERATIONS = 10000  # Maximum number of iterations

# Numerical precision parameters
TOL = 1e-6       # Absolute tolerance for geometry calculations
REL_TOL = 1e-9   # Relative tolerance
