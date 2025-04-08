import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.linalg import norm
import time
import matplotlib.pyplot as plt

# Import our matrix builder
from heat_solver.matrix_builder import MatrixBuilder
from heat_solver.geometry import PelvisGeometry

# Physical constants 
k_AL = 120
h = 7.5
T_inf = 20
T_wall = 70

# Heat sink calculation
A_conv = 77306 * 1e-6  # m², convectional surface area
A_ATR_SPINEX = 2120.4774 * 1e-6  # m², SPINEX contacting area
A_ATR_HIPY = 1791.2303 * 1e-6  # m², HIPY contacting area

Q_ATR_SPINEX = 0.5  # W
Q_ATR_HYPY = 20  # W

# Mesh parameters
Nx = 40
Ny = 40
Nz = 4

# Solver parameters
tol = 1e-8
rel_tol = 1e-8

# Preconditioner (simple diagonal preconditioning)
def preconditioner(A):
    M_inv = diags(1./A.diagonal())
    return LinearOperator(matvec=lambda x: M_inv @ x, shape=A.shape, dtype=A.dtype)

# Original helper functions from test.py
def tridiag(n, sub_val=-1, main_val=2, super_vale=-1):
    """Create a dense tridiagonal matrix of size n x n."""
    sub_diag = sub_val * np.ones(n-1)
    main_diag = main_val * np.ones(n)
    super_diag = super_val * np.ones(n-1)
    
    A = np.diag(main_diag, 0) + np.diag(sub_diag, -1) + np.diag(super_diag, 1)
    return A

def add_index(index, indices_array):
    """Check if 'index' is already in 'indices_array'."""
    if np.any(indices_array == index):
        print(f"Error: index {index} already used.")
        raise ValueError("Index used twice")
    return np.append(indices_array, index)

print("Initializing geometry...")
geometry = PelvisGeometry()

print("Building matrix using MatrixBuilder...")
matrix_builder = MatrixBuilder(geometry)
matrix_builder.build_base_matrix()

print("Applying boundary conditions...")
try:
    matrix_builder.apply_boundary_conditions()
    A_final = matrix_builder.A
    b = matrix_builder.b
    print("Boundary conditions applied successfully!")
except Exception as e:
    print(f"Error in MatrixBuilder: {e}")
    print("Falling back to original matrix building method...")
    
    # Original matrix building code from test.py
    dx = geometry.W_pelvis/Nx
    dy = geometry.L_pelvis/Ny
    dz = geometry.t_pelvis/Nz
    
    Ax = tridiag(Nx)/dx**2
    Ay = tridiag(Ny)/dy**2
    Az = tridiag(Nz)/dz**2
    
    Ix = np.identity(Nx)
    Iy = np.identity(Ny)
    Iz = np.identity(Nz)
    
    A_ini = np.kron(np.kron(Az, Iy), Ix) + np.kron(np.kron(Iz, Ay), Ix) + np.kron(np.kron(Iz, Iy), Ax)
    Ashape = np.size(A_ini[:,1])
    b = np.zeros(Ashape)
    A_final = A_ini.copy()
    
    # ... original boundary condition code would go here ...
    # For simplicity, we'll just use a placeholder matrix for testing
    print("WARNING: Using placeholder matrix - not applying boundary conditions")

# Check if the matrix is symmetric
is_symmetric = np.allclose(A_final, A_final.T, atol=1e-12)
print(f"Matrix is symmetric: {is_symmetric}")

# Print matrix info
print(f"Matrix shape: {A_final.shape}")
print(f"Number of non-zero elements: {np.count_nonzero(A_final)}")
print(f"Vector b shape: {b.shape}")

# Initialize the solver
print("\nStarting GMRES solver...")
iteration_count = 0

# Callback function to print the current residual norm with iteration counter
def callback(residual_norm):
    global iteration_count
    iteration_count += 1
    print(f"Iteration {iteration_count:4d} | Residual: {residual_norm:.2e}", end='\r', flush=True)

start_time = time.time()

# Initial guess
u0 = np.ones(A_final.shape[0]) * T_inf
M = preconditioner(A_final)

try:
    u, exitCode = gmres(A_final, b, M=M, x0=u0, atol=rel_tol, 
                     callback=callback, callback_type='pr_norm')
    end_time = time.time()
    
    print("\n" + "="*50)
    print("Calculation Summary:")
    print("-"*50)
    print(f"Total iterations: {iteration_count}")
    print(f"Calculation time: {end_time - start_time:.2f} seconds")
    print(f"Exit code: {exitCode}")
    print("="*50 + "\n")
    
    # Basic solution statistics
    print(f"Solution min: {np.min(u):.2f}")
    print(f"Solution max: {np.max(u):.2f}")
    print(f"Solution mean: {np.mean(u):.2f}")
    
    # Reshape for region analysis
    u_3d = u.reshape((Nz, Ny, Nx))
    
    # Calculate average temperatures for regions
    region1_temps = []  # SPINEX
    region2_temps = []  # HIPY_L
    region3_temps = []  # HIPY_R
    
    # Get mesh points
    x_pts, y_pts, z_pts = geometry.get_mesh_points()
    
    # Check top surface
    for i in range(Nx):
        for j in range(Ny):
            x_tmp = x_pts[i]
            y_tmp = y_pts[j]
            z_tmp = geometry.t_pelvis  # We only care about the top surface
            
            region = geometry.is_point_in_region(x_tmp, y_tmp, z_tmp)
            if region == geometry.REGION_SPINEX:
                region1_temps.append(u_3d[-1, j, i])
            elif region == geometry.REGION_HIPY_L:
                region2_temps.append(u_3d[-1, j, i])
            elif region == geometry.REGION_HIPY_R:
                region3_temps.append(u_3d[-1, j, i])
    
    print("\nAverage Temperatures:")
    print(f"SPINEX: {np.mean(region1_temps):.2f}°C") if region1_temps else print("No points in SPINEX region")
    print(f"HIPY_L: {np.mean(region2_temps):.2f}°C") if region2_temps else print("No points in HIPY_L region")
    print(f"HIPY_R: {np.mean(region3_temps):.2f}°C") if region3_temps else print("No points in HIPY_R region")
    
    # ==================== PLOTTING ====================
    
    # ---------------------------
    # 1. Plot u on the y–z plane at x = Nx//2
    # ---------------------------
    x_index = Nx // 2
    yz_slice = u_3d[:, :, x_index]
    Y, Z = np.meshgrid(y_pts, z_pts)
    
    plt.figure(figsize=(10,4))
    cont1 = plt.contourf(Y, Z, yz_slice, levels=100, cmap='jet')
    plt.colorbar(cont1)
    plt.title(f"Temperature on y–z plane at x = {x_pts[x_index]:.2e} m")
    plt.xlabel("y [m]")
    plt.ylabel("z [m]")
    plt.savefig('yz_slice.png')
    plt.show()
    
    # ---------------------------
    # 2. Plot u on the x–z plane at y = Ny//2
    # ---------------------------
    y_index = Ny // 2
    xz_slice = u_3d[:, y_index, :]
    X, Z = np.meshgrid(x_pts, z_pts)
    
    plt.figure(figsize=(10,4))
    cont2 = plt.contourf(X, Z, xz_slice, levels=100, cmap='jet')
    plt.colorbar(cont2)
    plt.title(f"Temperature on x–z plane at y = {y_pts[y_index]:.2e} m")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.savefig('xz_slice.png')
    plt.show()
    
    # ---------------------------
    # 3. Contour plot of top surface
    # ---------------------------
    X, Y = np.meshgrid(x_pts, y_pts)
    
    plt.figure(figsize=(10,8))
    cont3 = plt.contourf(X, Y, u_3d[-1,:,:], levels=20, cmap='jet')
    plt.colorbar(cont3)
    plt.title(f"Temperature on top surface (z = {geometry.t_pelvis:.2e} m)")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.savefig('top_surface_contour.png')
    plt.show()
    
    # ---------------------------
    # 4. 3D surface plot of top surface
    # ---------------------------
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, u_3d[-1,:,:], cmap='jet', edgecolor='none')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Temperature [°C]')
    ax.set_title('3D Surface Plot of Temperature at Top Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('top_surface_3d.png')
    plt.show()
    
except Exception as e:
    print(f"\nError during solving: {e}")
    import traceback
    traceback.print_exc()