import numpy as np

k_AL=120; h=5; T_inf = 30; T_wall = 70


A_conv = 57602.8753 * 1e-6 # m², convectional surface area
A_ATR_SPINEX = 2120.4774 * 1e-6 # m², SPINEX contacting area
A_ATR_HIPY = 1791.2303 * 1e-6 # m², HIPY contacting area

Q_ATR_SPINEX = 0.5 # W
Q_ATR_HYPY = 20 # W

q_ATR_SPINEX = Q_ATR_SPINEX/A_ATR_SPINEX
q_ATR_HYPY = Q_ATR_HYPY/A_ATR_HIPY

# Heat sink calculation

d1 = 120.7812 * 1e-3 # m, relative distance between SpineX and HIPY
d2 = 100 * 1e-3 # m, relative distance between 2 HIPYs
d3 = np.sqrt(d1**2-(d2/2)**2)

ratio = d3/d2
L_pelvis = np.sqrt(A_conv/ratio)
W_pelvis = L_pelvis/ratio
t_pelvis = 2*1e-3 # m

d_pelvis_SPINEX = np.sqrt(A_ATR_SPINEX)
d_pelvis_HIPY = np.sqrt(A_ATR_HIPY)

x_lim_pelvis = np.array([0, W_pelvis])
y_lim_pelvis = np.array([0, L_pelvis])
z_lim_pelvis = np.array([0, t_pelvis])

points = np.array([[0,0,0], [0,0,t_pelvis], [0,L_pelvis,0], [0,L_pelvis,t_pelvis],
                   [W_pelvis,0,0], [W_pelvis,0,t_pelvis], [W_pelvis,L_pelvis,0], [W_pelvis,L_pelvis,t_pelvis]])

x_lim_SPINEX = np.array([W_pelvis/2 - d_pelvis_SPINEX/2 , W_pelvis/2+d_pelvis_SPINEX/2])
y_lim_SPINEX = np.array([0 , d_pelvis_SPINEX])

x_lim_HIPY_L = np.array([W_pelvis/2 - d2/2 - d_pelvis_HIPY/2, W_pelvis/2 - d2/2 + d_pelvis_HIPY/2])
y_lim_HIPY_L = np.array([L_pelvis - d_pelvis_HIPY, L_pelvis])

x_lim_HIPY_R = np.array([W_pelvis/2 + d2/2 - d_pelvis_HIPY/2, W_pelvis/2 + d2/2 + d_pelvis_HIPY/2])
y_lim_HIPY_R = np.array([L_pelvis - d_pelvis_HIPY, L_pelvis])

# Actuator Calculation
t_ATR = 4*1e-3 # m
r_ATR = 44.65*1e-3 # m
L_ATR = np.pi*2*r_ATR # m
W_ATR = (53.5 - 42.7)*1e-3 # m

W_contact = (53.5 - 46.3)*1e-3 # m

tol = 1e-6
rel_tol = 1e-9

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
    x_lim_SPINEX = np.array([W_pelvis/2 - d_pelvis_SPINEX/2 , W_pelvis/2+d_pelvis_SPINEX/2])
    y_lim_SPINEX = np.array([0 , d_pelvis_SPINEX])

    x_lim_HIPY_L = np.array([W_pelvis/2 - d2/2 - d_pelvis_HIPY/2, W_pelvis/2 - d2/2 + d_pelvis_HIPY/2])
    y_lim_HIPY_L = np.array([L_pelvis - d_pelvis_HIPY, L_pelvis])

    x_lim_HIPY_R = np.array([W_pelvis/2 + d2/2 - d_pelvis_HIPY/2, W_pelvis/2 + d2/2 + d_pelvis_HIPY/2])
    y_lim_HIPY_R = np.array([L_pelvis - d_pelvis_HIPY, L_pelvis])   
    # Check if the point (x0, y0) lies within the bounds
    if abs(z0-t_pelvis) < tol:
        if  ((x_lim_HIPY_L[0] <= x0 <= x_lim_HIPY_L[1] and y_lim_HIPY_L[0] <= y0 <= y_lim_HIPY_L[1]) or
            (x_lim_HIPY_R[0] <= x0 <= x_lim_HIPY_R[1] and y_lim_HIPY_R[0] <= y0 <= y_lim_HIPY_R[1])):
            return 1
        if  (x_lim_SPINEX[0] <= x0 <= x_lim_SPINEX[1] and y_lim_SPINEX[0] <= y0 <= y_lim_SPINEX[1]):
            return 2
    else:
        return 0


Nx = 20; Ny = 20; Nz = 10
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
                        b[idx] = q_ATR_HYPY
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = q_ATR_SPINEX
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
                        b[idx] = q_ATR_HYPY
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = q_ATR_SPINEX
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
                        b[idx] = q_ATR_HYPY
                    #     A_final[idx, idx] = 1
                    #     b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = q_ATR_SPINEX
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
                        b[idx] = q_ATR_HYPY
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = q_ATR_SPINEX
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
                        b[idx] = q_ATR_HYPY
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = q_ATR_SPINEX
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
                        b[idx] = q_ATR_HYPY
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = q_ATR_SPINEX
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
                        b[idx] = q_ATR_HYPY
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = q_ATR_SPINEX
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
                        b[idx] = q_ATR_HYPY
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = q_ATR_SPINEX
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
                    
                        
                    if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                        A_final[idx, idx] = -1/dx**2 - 2/dy**2 - 1/dz**2
                        b[idx] = q_ATR_HYPY
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -1/dx**2 - 2/dy**2 - 1/dz**2
                        b[idx] = q_ATR_SPINEX
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    else:
                        A_final[idx, idx] = -1/dx**2 - 2/dy**2 - 1/dz**2 - h/dz/k_AL
                        b[idx] = -T_inf*h/dz/k_AL
                    A_final[idx, idx + Nx] = 1/dy**2
                    A_final[idx, idx - Nx] = 1/dy**2
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
                        b[idx] = q_ATR_HYPY
                        # A_final[idx, idx] = 1
                        # b[idx] = T_wall
                    elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                        A_final[idx, idx] = -2/dx**2 - 1/dy**2 - 1/dz**2
                        b[idx] = q_ATR_SPINEX
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
                    A_final[idx, idx - 1] = 1/dx**2
                    A_final[idx, idx - Nx] = 1/dy**2
                    A_final[idx, idx + 1] = 1/dx**2
                    A_final[idx, idx + Nx] = 1/dy**2
                    if np.argmin(abs(z_lim_pelvis - z_tmp)) == 0:
                        A_final[idx, idx] = -2/dx**2 - 2/dy**2 - 1/dz**2
                        A_final[idx, idx + Nx*Ny] = 1/dz**2
                    else:
                        if is_point_in_region(x_tmp, y_tmp, z_tmp) == 1:
                            A_final[idx, idx] = -2/dx**2 - 2/dy**2 - 1/dz**2
                            b[idx] = q_ATR_HYPY
                            # A_final[idx, idx] = 1
                            # b[idx] = T_wall
                        elif is_point_in_region(x_tmp, y_tmp, z_tmp) == 2:
                            A_final[idx, idx] = -2/dx**2 - 2/dy**2 - 1/dz**2
                            b[idx] = q_ATR_SPINEX
                            # A_final[idx, idx] = 1
                            # b[idx] = T_wall
                        else:
                            A_final[idx, idx] = -2/dx**2 - 2/dy**2 - 1/dz**2 - h/dz/k_AL
                            b[idx] = -T_inf*h/dz/k_AL
                        A_final[idx, idx - Nx*Ny] = 1/dz**2
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

u = np.ones(Ashape)*T_inf
D = np.diag(A_final)
Di = 1.0 / D

# Jacobi iteration
tolerance = 1e-6
error = 1
errors = []
iter = 0
while error > tolerance:
    iter += 1
    u_new = u + Di * (b - A_final.dot(u))
    error = np.linalg.norm(u_new - u, np.inf)
    u = u_new
    print(error, iter)
        
print(error, iter)