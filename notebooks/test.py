import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.linalg import norm  # Import the norm function from scipy.linalg
import time


k_AL=120; h=7.5; T_inf = 20; T_wall = 70


A_conv = 77306 * 1e-6 # m², convectional surface area
A_ATR_SPINEX = 2120.4774 * 1e-6 # m², SPINEX contacting area
A_ATR_HIPY = 1791.2303 * 1e-6 # m², HIPY contacting area

Q_ATR_SPINEX = 0.5 # W
Q_ATR_HYPY = 20 # W

# Heat sink calculation

d1 = 120.7812 * 1e-3 # m, relative distance between SpineX and HIPY
d2 = 100 * 1e-3 # m, relative distance between 2 HIPYs
d3 = np.sqrt(d1**2-(d2/2)**2)

W_pelvis = 299.0796e-3
L_pelvis = A_conv/W_pelvis
t_pelvis = 2*1e-3 # m

# SPINEX and HIPY dimensions and offsets
L_pelvis_SPINEX = 191.6729e-3
W_pelvis_SPINEX = 11.063e-3
d_pelvis_SPINEX = 7e-3

correction_factor = 1.5
L_pelvis_HIPY = 306.14e-3 / correction_factor
W_pelvis_HIPY = 5.85e-3 * correction_factor
d_pelvis_HIPY = 13.9105e-3

q_ATR_SPINX = Q_ATR_SPINEX/L_pelvis_SPINEX/W_pelvis_SPINEX
q_ATR_HYPY = Q_ATR_HYPY/W_pelvis_HIPY/L_pelvis_HIPY

x_lim_pelvis = np.array([0, W_pelvis])
y_lim_pelvis = np.array([0, L_pelvis])
z_lim_pelvis = np.array([0, t_pelvis])

points = np.array([[0,0,0], [0,0,t_pelvis], [0,L_pelvis,0], [0,L_pelvis,t_pelvis],
                   [W_pelvis,0,0], [W_pelvis,0,t_pelvis], [W_pelvis,L_pelvis,0], [W_pelvis,L_pelvis,t_pelvis]])

x_lim_SPINEX = np.array([W_pelvis/2 - L_pelvis_SPINEX/2 , W_pelvis/2 + L_pelvis_SPINEX/2])
y_lim_SPINEX = np.array([d_pelvis_SPINEX , d_pelvis_SPINEX + W_pelvis_SPINEX])

x_lim_HIPY_L = np.array([d_pelvis_HIPY, d_pelvis_HIPY + W_pelvis_HIPY])
y_lim_HIPY_L = np.array([L_pelvis - L_pelvis_HIPY, L_pelvis])

x_lim_HIPY_R = np.array([W_pelvis - W_pelvis_HIPY - d_pelvis_HIPY, W_pelvis - d_pelvis_HIPY])
y_lim_HIPY_R = np.array([L_pelvis - L_pelvis_HIPY, L_pelvis])

# Actuator Calculation
t_ATR = 4*1e-3 # m
r_ATR = 44.65*1e-3 # m
L_ATR = np.pi*2*r_ATR # m
W_ATR = (53.5 - 42.7)*1e-3 # m

W_contact = (53.5 - 46.3)*1e-3 # m

tol = 1e-8
rel_tol = 1e-8

def tridiag(n, sub_val=-1, main_val=2, super_val=-1):
    """
    Create a dense tridiagonal matrix of size n x n.

    Parameters:
        n (int): Size of the matrix (number of rows/columns).
        sub_val (float): Value for the sub-diagonal elements (default: -1).
        main_val (float): Value for the main diagonal elements (default: 2).
        super_val (float): Value for the super-diagonal elements (default: -1).

    Returns:
        A (ndarray): A tridiagonal matrix.
    """
    sub_diag = sub_val * np.ones(n-1)
    main_diag = main_val * np.ones(n)
    super_diag = super_val * np.ones(n-1)
    
    A = np.diag(main_diag, 0) + np.diag(sub_diag, -1) + np.diag(super_diag, 1)
    return A

def add_index(index, indices_array):
    """Check if 'index' is already in 'indices_array'. 
    If yes, raise an error; otherwise, append and return the new array."""
    if np.any(indices_array == index):
        print(f"Error: index {index} already used.")
        raise ValueError("Index used twice")
    return np.append(indices_array, index)

def is_point_in_region(x0, y0, z0): 
    x_lim_SPINEX = np.array([W_pelvis/2 - L_pelvis_SPINEX/2 , W_pelvis/2+L_pelvis_SPINEX/2])
    y_lim_SPINEX = np.array([d_pelvis_SPINEX , d_pelvis_SPINEX + W_pelvis_SPINEX])

    x_lim_HIPY_L = np.array([d_pelvis_HIPY, d_pelvis_HIPY + W_pelvis_HIPY])
    y_lim_HIPY_L = np.array([L_pelvis - L_pelvis_HIPY, L_pelvis])

    x_lim_HIPY_R = np.array([W_pelvis - W_pelvis_HIPY - d_pelvis_HIPY, W_pelvis - d_pelvis_HIPY])
    y_lim_HIPY_R = np.array([L_pelvis - L_pelvis_HIPY, L_pelvis]) 
    # Check if the point (x0, y0) lies within the bounds
    if abs(z0-t_pelvis) < tol:
        if  ((x_lim_HIPY_L[0] <= x0 <= x_lim_HIPY_L[1] and y_lim_HIPY_L[0] <= y0 <= y_lim_HIPY_L[1]) or
            (x_lim_HIPY_R[0] <= x0 <= x_lim_HIPY_R[1] and y_lim_HIPY_R[0] <= y0 <= y_lim_HIPY_R[1])):
            return 1
        if  (x_lim_SPINEX[0] <= x0 <= x_lim_SPINEX[1] and y_lim_SPINEX[0] <= y0 <= y_lim_SPINEX[1]):
            return 2
    else:
        return 0
    
# Preconditioner (simple diagonal preconditioning)
def preconditioner(A):
    M_inv = diags(1./A.diagonal())
    return LinearOperator(matvec=lambda x: M_inv @ x, shape=A.shape, dtype=A.dtype)


Nx = 40; Ny = 40; Nz = 6
dx = W_pelvis/Nx; dy = L_pelvis / Ny; dz = t_pelvis / Nz
Ax = tridiag(Nx)/dx**2; Ay = tridiag(Ny)/dy**2; Az = tridiag(Nz)/dz**2
Ix = np.identity(Nx); Iy = np.identity(Ny); Iz = np.identity(Nz)

A_ini = np.kron(np.kron(Az, Iy), Ix) + np.kron(np.kron(Iz, Ay), Ix) + np.kron(np.kron(Iz, Iy), Ax)
Ashape = np.size(A_ini[:,1])
b = np.zeros(Ashape)
A_final = A_ini.copy()

x_pts = np.linspace(0, W_pelvis, Nx)
y_pts = np.linspace(0, L_pelvis, Ny)
z_pts = np.linspace(0, t_pelvis, Nz)

# Initialize the array to store used indices (using integer type)
indices_used = np.array([], dtype=int)

for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            idx = i + j*Nx + k*Nx*Ny
            x_tmp = x_pts[i]; y_tmp = y_pts[j]; z_tmp = z_pts[k]
            xyz = np.array([x_tmp, y_tmp, z_tmp])
            distances = np.linalg.norm(points - xyz, axis=1)
            if np.min(distances) < tol: # points
                indices_used = add_index(idx, indices_used)
                if np.argmin(distances) == 0:
                    # (0,0,0)
                    A_final[idx, :] = 0
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_HYPY/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_SPINX/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2 - h/dz/k_AL
                        b[idx] = -T_inf*h/dz/k_AL
                    A_final[idx, idx + 1] = 1/dx**2
                    A_final[idx, idx + Nx] = 1/dy**2
                    A_final[idx, idx + Nx*Ny] = 1/dz**2
                    
                elif np.argmin(distances) == 1:
                    # (0,0,1)
                    A_final[idx, :] = 0
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_HYPY/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_SPINX/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2 - h/dz/k_AL
                        b[idx] = -T_inf*h/dz/k_AL
                    A_final[idx, idx + 1] = 1/dx**2
                    A_final[idx, idx + Nx] = 1/dy**2
                    A_final[idx, idx - Nx*Ny] = 1/dz**2
                elif np.argmin(distances) == 2:
                    # (0,1,0)
                    A_final[idx, :] = 0
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_HYPY/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_SPINX/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2 - h/dz/k_AL
                        b[idx] = -T_inf*h/dz/k_AL
                    A_final[idx, idx + 1] = 1/dx**2
                    A_final[idx, idx - Nx] = 1/dy**2
                    A_final[idx, idx + Nx*Ny] = 1/dz**2    
                elif np.argmin(distances) == 3:
                    # (0,1,1)
                    A_final[idx, :] = 0
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_HYPY/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_SPINX/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2 - h/dz/k_AL
                        b[idx] = -T_inf*h/dz/k_AL
                    A_final[idx, idx + 1] = 1/dx**2
                    A_final[idx, idx - Nx] = 1/dy**2
                    A_final[idx, idx - Nx*Ny] = 1/dz**2 
                elif np.argmin(distances) == 4:
                    # (1,0,0)
                    A_final[idx, :] = 0
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_HYPY/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_SPINX/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2 - h/dz/k_AL
                        b[idx] = -T_inf*h/dz/k_AL
                    A_final[idx, idx - 1] = 1/dx**2
                    A_final[idx, idx + Nx] = 1/dy**2
                    A_final[idx, idx + Nx*Ny] = 1/dz**2 
                elif np.argmin(distances) == 5:
                    # (1,0,1)
                    A_final[idx, :] = 0
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_HYPY/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_SPINX/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2 - h/dz/k_AL
                        b[idx] = -T_inf*h/dz/k_AL
                    A_final[idx, idx - 1] = 1/dx**2
                    A_final[idx, idx + Nx] = 1/dy**2
                    A_final[idx, idx - Nx*Ny] = 1/dz**2
                elif np.argmin(distances) == 6:
                    # (1,1,0)
                    A_final[idx, :] = 0
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_HYPY/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_SPINX/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2 - h/dz/k_AL
                        b[idx] = -T_inf*h/dz/k_AL
                    A_final[idx, idx - 1] = 1/dx**2
                    A_final[idx, idx - Nx] = 1/dy**2
                    A_final[idx, idx + Nx*Ny] = 1/dz**2
                elif np.argmin(distances) == 7:
                    # (1,1,1)
                    A_final[idx, :] = 0
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_HYPY/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_SPINX/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2 - h/dz/k_AL
                        b[idx] = -T_inf*h/dz/k_AL
                    A_final[idx, idx - 1] = 1/dx**2
                    A_final[idx, idx - Nx] = 1/dy**2
                    A_final[idx, idx - Nx*Ny] = 1/dz**2
            elif ((np.min(abs(x_lim_pelvis-x_tmp)) < tol and np.min(abs(y_lim_pelvis-y_tmp)) < tol) or 
                  (np.min(abs(x_lim_pelvis-x_tmp)) < tol and np.min(abs(z_lim_pelvis-z_tmp)) < tol) or
                  (np.min(abs(y_lim_pelvis-y_tmp)) < tol and np.min(abs(z_lim_pelvis-z_tmp)) < tol)):
                # edges
                indices_used = add_index(idx, indices_used)
                A_final[idx, :] = 0
                if (np.min(abs(x_lim_pelvis-x_tmp)) < tol and np.min(abs(y_lim_pelvis-y_tmp)) < tol):
                    # parallel to z-axis
                    A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 2/dz**2
                    A_final[idx, idx + Nx*Ny] = 1/dz**2
                    A_final[idx, idx - Nx*Ny] = 1/dz**2
                    if np.argmin(x_lim_pelvis - x_tmp) == 0:
                        A_final[idx, idx + 1] = 1/dx**2
                    else:
                        A_final[idx, idx - 1] = 1/dx**2
                    if np.argmin(y_lim_pelvis - y_tmp) == 0:
                        A_final[idx, idx + Nx] = 1/dy**2
                    else:
                        A_final[idx, idx - Nx] = 1/dy**2
                        
                elif (np.min(abs(x_lim_pelvis-x_tmp)) < tol and np.min(abs(z_lim_pelvis-z_tmp)) < tol):
                    # parallel to y-axis
                    A_final[idx, idx + Nx] = 1/dy**2
                    A_final[idx, idx - Nx] = 1/dy**2
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -1/dx**2 - 2/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_HYPY/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 2/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_SPINX/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        A_final[idx, idx] = -1/dx**2 - 2/dy**2 - 1/dz**2 - h/dz/k_AL
                        b[idx] = -T_inf*h/dz/k_AL
                        if np.argmin(abs(x_lim_pelvis - x_tmp)) == 0:
                            A_final[idx, idx + 1] = 1/dx**2
                        else:
                            A_final[idx, idx - 1] = 1/dx**2
                        if np.argmin(abs(z_lim_pelvis - z_tmp)) == 0:
                            A_final[idx, idx + Nx*Ny] = 1/dz**2
                        else:
                            A_final[idx, idx - Nx*Ny] = 1/dz**2
                        
                elif (np.min(abs(y_lim_pelvis-y_tmp)) < tol and np.min(abs(z_lim_pelvis-z_tmp)) < tol):
                    # parallel to x-axis
                        
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -2/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_HYPY/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -2/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_SPINX/k_AL/dz
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        A_final[idx, idx] = -2/dx**2 - 1/dy**2 - 1/dz**2 - h/dz/k_AL
                        b[idx] = -T_inf*h/dz/k_AL
                    A_final[idx, idx + 1] = 1/dx**2
                    A_final[idx, idx - 1] = 1/dx**2
                    if np.argmin(abs(y_lim_pelvis - y_tmp)) == 0:
                        A_final[idx, idx + Nx] = 1/dy**2
                    else:
                        A_final[idx, idx - Nx] = 1/dy**2
                    if np.argmin(abs(z_lim_pelvis - z_tmp)) == 0:
                        A_final[idx, idx + Nx*Ny] = 1/dz**2
                    else:
                        A_final[idx, idx - Nx*Ny] = 1/dz**2  

            elif (np.min(abs(x_lim_pelvis - x_tmp)) < tol or
                  np.min(abs(y_lim_pelvis - y_tmp)) < tol or
                  np.min(abs(z_lim_pelvis - z_tmp)) < tol):
                # surface
                indices_used = add_index(idx, indices_used)
                if np.min(abs(z_lim_pelvis - z_tmp)) < tol:
                    # +Z or -Z surface
                    A_final[idx, :] = 0
                    b[idx] = 0
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -2/dx**2 - 2/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_HYPY/k_AL/dz
                        A_final[idx, idx - Nx*Ny] = 1/dz**2
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -2/dx**2 - 2/dy**2 - 1/dz**2
                        b[idx] = -q_ATR_SPINX/k_AL/dz
                        A_final[idx, idx - Nx*Ny] = 1/dz**2
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        if np.argmin(abs(z_lim_pelvis - z_tmp)) == 0:
                            # adiabatic on Z=0 surface
                            A_final[idx, idx] = -2/dx**2 - 2/dy**2 - 1/dz**2
                            A_final[idx, idx + Nx*Ny] = 1/dz**2
                        else:
                            # convective on Z=t_pelvis surface
                            A_final[idx, idx] = -2/dx**2 - 2/dy**2 - 1/dz**2 - h/dz/k_AL
                            b[idx] = -T_inf*h/dz/k_AL
                            A_final[idx, idx - Nx*Ny] = 1/dz**2
                    A_final[idx, idx - 1] = 1/dx**2
                    A_final[idx, idx - Nx] = 1/dy**2
                    A_final[idx, idx + 1] = 1/dx**2
                    A_final[idx, idx + Nx] = 1/dy**2
                elif np.min(abs(x_lim_pelvis - x_tmp)) < tol:
                    # +X or -X surface
                    A_final[idx, :] = 0
                    A_final[idx, idx] = -1/dx**2 - 2/dy**2 - 2/dz**2
                    A_final[idx, idx - Nx] = 1/dy**2
                    A_final[idx, idx + Nx] = 1/dy**2
                    A_final[idx, idx + Nx*Ny] = 1/dz**2
                    A_final[idx, idx - Nx*Ny] = 1/dz**2
                    if np.argmin(abs(x_lim_pelvis - x_tmp)) == 0:
                        A_final[idx, idx + 1] = 1/dx**2
                    else:
                        A_final[idx, idx - 1] = 1/dx**2
                elif np.min(abs(y_lim_pelvis - y_tmp)) < tol:
                    # +y or -y surface
                    A_final[idx, :] = 0
                    A_final[idx, idx] = -2/dx**2 - 1/dy**2 - 2/dz**2
                    A_final[idx, idx + 1] = 1/dx**2
                    A_final[idx, idx - 1] = 1/dx**2
                    A_final[idx, idx + Nx*Ny] = 1/dz**2
                    A_final[idx, idx - Nx*Ny] = 1/dz**2
                    if np.argmin(abs(y_lim_pelvis - y_tmp)) == 0:
                        A_final[idx, idx + Nx] = 1/dy**2
                    else:
                        A_final[idx, idx - Nx] = 1/dy**2
                        
total_indices = (4*(Nx-2) + 4*(Ny-2) + 4*(Nz-2)) + ((Nx-2)*(Ny-2) + (Nx-2)*(Nz-2) + (Ny-2)*(Nz-2))*2 + 8
if len(indices_used) != total_indices:
    print(len(indices_used), Nz*Nx*Ny)
    print("some point left over")
    raise ValueError("missing indices")
else:
    print("BC set")
    
print('IC set')

print(np.allclose(A_final, A_final.T, atol=1e-12))

# Add timing and iteration counter
iteration_count = 0

# Callback function to print the current residual norm with iteration counter
def callback(residual_norm):
    global iteration_count
    iteration_count += 1
    print(f"Iteration {iteration_count:4d} | Residual: {residual_norm:.2e}", end='\r', flush=True)

print("\nStarting GMRES solver...")
start_time = time.time()

u0 = np.ones(Ashape) * T_inf
M = preconditioner(A_final)
u, exitCode = gmres(A_final, b, M=M, x0 = u0, atol=rel_tol, callback=callback, callback_type='pr_norm')

end_time = time.time()
calculation_time = end_time - start_time

# Clear the last line of the progress output
print(" " * 50, end='\r')

print("\n" + "="*50)
print("Calculation Summary:")
print("-"*50)
print(f"Total iterations: {iteration_count}")
print(f"Calculation time: {calculation_time:.2f} seconds")
print(f"Exit code: {exitCode}")
print("="*50 + "\n")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# Reshape u into a 3D array of shape (Nx, Ny, Nz)
u_3d = u.reshape((Nz, Ny, Nx))


# Calculate average temperatures for regions 1 (HIPY) and 2 (SPINEX)
region1_temps = []  # HIPY
region2_temps = []  # SPINEX

# Only check the top surface where z = t_pelvis
for i in range(Nx):
    for j in range(Ny):
        x_tmp = x_pts[i]
        y_tmp = y_pts[j]
        z_tmp = t_pelvis  # We only care about the top surface
        
        region = is_point_in_region(x_tmp, y_tmp, z_tmp)
        if region == 1:  # HIPY region
            region1_temps.append(u_3d[-1,j,i])
        elif region == 2:  # SPINEX region
            region2_temps.append(u_3d[-1,j,i])

avg_temp_HIPY = np.mean(region1_temps)
avg_temp_SPINEX = np.mean(region2_temps)

print("\nAverage Temperatures:")
print(f"HIPY (Region 1): {avg_temp_HIPY:.2f}°C")
print(f"SPINEX (Region 2): {avg_temp_SPINEX:.2f}°C")

# ---------------------------
# 1. Plot u on the y–z plane at x = Nx//2
# ---------------------------
x_index = Nx // 2
# This slice has shape (Ny, Nz) where:
# - the first dimension corresponds to y (horizontal)
# - the second dimension corresponds to z (vertical)
yz_slice = u_3d[:, :, x_index]

# Using default indexing ('xy') returns arrays of shape (Nz, Ny), so we transpose the slice:
Y, Z = np.meshgrid(y_pts, z_pts)
# Since yz_slice is (Ny, Nz), we transpose it so that the dimensions match:

plt.figure(figsize=(10,1))
cont1 = plt.contourf(Y, Z, yz_slice, levels=100, cmap='jet')
plt.colorbar(cont1)
plt.title(f"u on y–z plane at x = {x_index}")
plt.xlabel("y")
plt.ylabel("z")
plt.show()

# ---------------------------
# 2. Plot u on the x–z plane at y = Ny//2
# ---------------------------
y_index = Ny // 2
# This slice has shape (Nx, Nz) where:
# - the first dimension corresponds to x (horizontal)
# - the second dimension corresponds to z (vertical)
xz_slice = u_3d[:, y_index, :]
X, Z = np.meshgrid(x_pts, z_pts)
# Transpose the slice so that its first dimension matches X (horizontal)

plt.figure(figsize=(10,1))
cont2 = plt.contourf(X, Z, xz_slice, levels=100, cmap='jet')
plt.colorbar(cont2)
plt.title(f"u on x–z plane at y = {y_index}")
plt.xlabel("x")
plt.ylabel("z")
plt.show()


# Create a figure and a 3D axes
X, Y = np.meshgrid(x_pts, y_pts)

plt.figure()
cont2 = plt.contourf(X, Y, u_3d[-1,:,:], levels=20, cmap='jet')
plt.colorbar(cont2)
plt.title(f"u on x–z plane at y = {y_index}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Create a figure and a 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the surface; the facecolors are automatically mapped to the Z values.
surf = ax.plot_surface(X, Y, u_3d[-1,:,:], cmap='jet', edgecolor='none')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Temperature [C]')
ax.set_title('3D Surface Plot of Temperature Value at Top Surface')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.show()
