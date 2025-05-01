import numpy as np
import numpy.linalg as la

# -- Step 1: Spreading Resistance (R_sp) --
# (This uses your existing calc_p, calc_C, and calc_R_sp approach.)

def calc_p(t_bar, Bi_sink, lam_n):
    """
    Compute the array of p-coefficients for the dimensionless problem.
    """
    p0 = -(t_bar + 1 / Bi_sink)
    pn = -(Bi_sink * np.cosh(lam_n[1:] * t_bar) + lam_n[1:] * np.sinh(lam_n[1:] * t_bar)) / \
         (Bi_sink * np.sinh(lam_n[1:] * t_bar) + lam_n[1:] * np.cosh(lam_n[1:] * t_bar))
    return np.hstack(([p0], pn))

def calc_C(N, a_bar, Bi_max, lam_n, p):
    """
    Solve for C-coefficients in the dimensionless series solution.
    """
    A_mat = np.zeros((N+1, N+1))
    b = np.zeros(N+1)

    # m=0 row
    A_mat[0, 0] = 1 / Bi_max - p[0] * a_bar
    for n in range(1, N+1):
        A_mat[0, n] = -np.sin(lam_n[n] * a_bar) / lam_n[n]
    b[0] = -a_bar

    # m=1..N rows
    for m in range(1, N+1):
        A_mat[m, 0] = -p[0] * np.sin(lam_n[m] * a_bar) / lam_n[m]
        for n in range(1, N+1):
            if m == n:
                A_mat[m, n] = (lam_n[m] * p[m] / (2 * Bi_max)
                               - a_bar / 2
                               - np.sin(2 * lam_n[m] * a_bar) / (4 * lam_n[m]))
            else:
                A_mat[m, n] = -(lam_n[n]*np.sin(lam_n[n]*a_bar)*np.cos(lam_n[m]*a_bar)
                                - lam_n[m]*np.cos(lam_n[n]*a_bar)*np.sin(lam_n[m]*a_bar)) \
                              / (lam_n[n]**2 - lam_n[m]**2)
        b[m] = -np.sin(lam_n[m] * a_bar) / lam_n[m]

    C = la.solve(A_mat, b)
    return C

def calc_R_sp(k, L, a_bar, lam_n, C, p):
    """
    Calculate spreading resistance R_sp using:
        k*L*R_sp = -[C0*a_bar + sum_{n=1 to N}( Cn*pn*sin(lam_n[n]*a_bar )]^-1 + p0
    """
    p0 = p[0]
    N = len(C) - 1

    # X = C0*a_bar + sum_{n=1..N}( C[n]*p[n]*sin(lam_n[n]*a_bar) )
    X = C[0]*a_bar
    for n in range(1, N+1):
        X += C[n]*p[n]*np.sin(lam_n[n]*a_bar)

    kL_Rsp = p0 - 1.0/X  # k*L*R_sp
    return kL_Rsp/(k*L)


# -- Step 2: 1-D Conduction Resistance (R_cond) --
def calc_R_cond_1D(thickness, k_cond, area):
    """
    1-D conduction resistance for a slab:
        R_cond = thickness / (k_cond * area)
    """
    return thickness / (k_cond * area)


# -- Step 3: Fin Heat Transfer Rate q_j (Adiabatic Tip) --
def calc_fin_heat_adiabatic_tip(h, k_fin, P, A_c, L_fin, T_base, T_inf):
    """
    For a straight fin with adiabatic tip:
        q_j = sqrt(h * P * k_fin * A_c) * (T_base - T_inf) * tanh(m * L_fin),
    where m = sqrt( h * P / (k_fin * A_c) ).

    P     : perimeter of the fin cross section [m]
    A_c   : cross-sectional area of the fin [m^2]
    L_fin : length of the fin [m]
    """
    m = np.sqrt(h * P / (k_fin * A_c))
    return np.sqrt(h * P * k_fin * A_c) * (T_base - T_inf) * np.tanh(m * L_fin)


# -- Step 4: Calculate HTC of air via a Nusselt correlation --
def calc_h_air(k_air, L, Ra_L, Pr):
    Nu_L = (0.825 + 0.387 * (Ra_L**(1/6)) / (1 + (0.492/Pr)**(9/16))**(8/27))**2
    h = Nu_L * k_air / L
    return h

def calc_rayleigh_number(g, beta, T_s, T_inf, L, alpha, nu):
    """
    Calculate the Rayleigh number for natural convection:
    
        Ra_L = g * beta * (T_s - T_inf) * L^3 / (alpha * nu)
    
    Parameters
    ----------
    g : float
        Gravitational acceleration [m/s^2]
    beta : float
        Thermal expansion coefficient [1/K]
    T_s : float
        Surface (or wall) temperature [C or K, but be consistent]
    T_inf : float
        Ambient temperature [C or K, same scale as T_s]
    L : float
        Characteristic length [m]
    alpha : float
        Thermal diffusivity [m^2/s]
    nu : float
        Kinematic viscosity [m^2/s]
    
    Returns
    -------
    float
        Rayleigh number (dimensionless)
    """
    return g * beta * (T_s - T_inf) * (L**3) / (alpha * nu)

def cal_theta(xi, eta, C, p, lam_n, N):
    """
    Calculate the dimensionless temperature difference (theta) at a given dimensionless location (xi, eta).
    
    xi = x/A,   eta = z/A
    """
    theta_value = C[0]*(eta + p[0])
    for n in range(1, N+1):
        theta_value += C[n] * np.cos(lam_n[n]*xi) * (np.cosh(lam_n[n]*eta) + p[n]*np.sinh(lam_n[n]*eta))
    return theta_value

# -- Step 5: Putting it all together in a main driver --
def main():
    """
    Demonstrate the algorithm steps:
    1) R_sp:   Spreading resistance
    2) R_cond: 1-D conduction resistance in sink
    3) q_j:    Heat from the fin (adiabatic tip)
    4) h:      Air-side HTC from correlation
    5) Q_base: Remaining heat dissipated via conduction path
    6) T_b:    Base temperature given Q, R_sp, R_cond, and conv. path
    """
    # -------------------------------------------------
    # (A) USER/GEOMETRY/PHYSICAL INPUTS
    # -------------------------------------------------
    # Example geometry for the dimensionless solution
    N       = 150        # of terms for the series
    Bi_max  = 1e6
    L_fin   = 1000/1e3          # length of the fin (standing rod) [m], 700 for Ruiqi's Design; 
    V_heatsink_measured = 117141.4314/1e9       # [m^3], 176002.0472 for Ruiqi's Design; 117141.4314 for Ryan's ID design
    A_heatsink_measured = 72647.1609/1e6     # [m^2] 90122.6521 for Ruiqi's Design; 72647.1609 for Ryan's ID design
    t_sink = V_heatsink_measured/A_heatsink_measured        # [m]
    print(f" - Sink Representative Thickness, t_sink   = {t_sink*1e3:.4f} mm")
    W_sink = 29.6957/1e3       # [m] 25.5 for Ruiqi's Design; 29.6957 for Ryan's ID design 
    # L_sink = V_heatsink_measured/t_sink/W_sink
    L_sink = A_heatsink_measured/W_sink
    L_handstouch_measured = L_sink/2     # [m]
    L_chip = 10e-3       # [m]
    W_chip = 8e-3        # [m]
    L_chip = L_chip*W_chip/W_sink
    a_bar   = L_chip/L_sink     # dimensionless half-length
    t_bar   = t_sink/L_sink      # dimensionless thickness

    # Physical geometry (for R_sp scaling)
    k_fin   = 120.0      # W/m-K (material of the sink)
    L       = L_sink       # characteristic length [m]
    A       = L          # same scale used in dimensionless definitions

    # For conduction
    sink_thickness = t_sink     # [m]
    sink_area      = L_sink*W_sink     # [m^2] cross-sectional area for conduction

    # For the rod-fin side
    P_fin = 210.8094e-3  # [m]
    r_fin   = P_fin/(2*np.pi)
    P_fin   = np.pi*r_fin*2   # perimeter of cross section [m]
    A_c_fin = np.pi*r_fin**2  # cross-sectional area of fin [m^2]
    

    # Flow conditions
    k_air   = 0.026      # W/m-K (approx for air)
    Pr      = 0.71       # approx for air

    # Heat loads / temperature
    Q_total = 5       # [W] total heat
    T_inf   = 30       # [C] ambient
    # We'll find T_b (base temperature) from the "remaining" conduction path

    g      = 9.8         # m/s^2
    beta   = 0.0025      # 1/K
    T_inf  = 30        # °C (ambient temperature)
    T_s    = T_inf + 5   # °C (surface temperature)
    alpha  = 38.3e-6     # m^2/s (thermal diffusivity)
    nu     = 26.4e-6     # m^2/s (kinematic viscosity)
    
    T_base_guess = T_inf+5  # [C], an initial guess
    
    tolerance = 1e-2
    relaxation_factor = 0.05
    max_iter = 50
    
    for i in range(max_iter):
        # -------------------------------------------------
        # CALCULATE 1-D CONDUCTION RESISTANCE
        # -------------------------------------------------
        R_cond = calc_R_cond_1D(sink_thickness, k_fin, sink_area)
        # print("1-D Conduction Resistance (R_cond) =", R_cond, "K/W")

        Ra_L = calc_rayleigh_number(g, beta, T_s, T_inf, L, alpha, nu)
        # print(f"Rayleigh number Ra_L = {Ra_L:.3e}")
        
        # -------------------------------------------------
        # (E) CALCULATE HTC OF AIR from correlation
        # -------------------------------------------------
        h_air = calc_h_air(k_air, L, Ra_L, Pr)
        # h_air = 7.5
        Bi_sink = h_air*L_sink/k_fin
        # print("Air HTC, h_air =", h_air, "[W/m^2-K]")

        # -------------------------------------------------
        # (B) CALCULATE R_sp (Spreading Resistance)
        # -------------------------------------------------
        lam_n = np.arange(0, N+1) * np.pi
        p = calc_p(t_bar, Bi_sink, lam_n)
        C = calc_C(N, a_bar, Bi_max, lam_n, p)
        R_sp = calc_R_sp(k_fin, L, a_bar, lam_n, C, p)
        # print("Spreading Resistance (R_sp) =", R_sp, "K/W")
        
        # -------------------------------------------------
        # (D) CALCULATE FIN HEAT TRANSFER RATE (ADIABATIC TIP)
        #     Example: we treat a 'fin' on top of the sink
        # -------------------------------------------------
        # We'll guess T_base to get an estimate of q_j. 
        # Typically you'd iterate if T_base is unknown, but let's do a single pass.
        
        q_j = calc_fin_heat_adiabatic_tip(h_air, k_fin, P_fin, A_c_fin,
                                        L_fin, T_base_guess, T_inf)
        # print("Fin Heat Transfer Rate (adiabatic tip) =", q_j, "W")
        
        # -------------------------------------------------
        # (F) ALLOCATE HEAT AND FIND BASE TEMPERATURE
        # -------------------------------------------------
        # Suppose a portion q_j of the total Q_total is dissipated by the fin
        # The rest Q_base must flow through conduction from base to environment:
        Q_base = Q_total - q_j
        # We'll combine them in series for an approximate total R_total:
        A_sink_surface = sink_area  # or something else, depends on geometry
        R_conv = 1.0 / (h_air * (A_sink_surface-A_c_fin))

        R_total = R_sp + R_cond + R_conv
        T_bc = T_inf + Q_base * R_total
        T_base_new = T_bc - Q_total*(R_cond+R_sp)
    
        # 4) Check convergence
        if abs(T_base_new - T_base_guess) < tolerance:
            T_base_guess = T_base_new
            break

        # 5) Update guess with relaxation
        T_base_guess += relaxation_factor * (T_base_new - T_base_guess)
        T_s = T_base_guess

    theta_handstouch = cal_theta(L_chip/L_handstouch_measured, t_bar, C, p, lam_n, N)
    T_handstouch = (T_bc-T_inf)*theta_handstouch + T_inf
    
    # -------------------------------------------------
    # Print summary
    # -------------------------------------------------
    print("=== 2-D Heat Transfer Model Results ===")
    print(f"1) Spreading Resistance, R_sp   = {R_sp:.4f} K/W")
    print(f"2) 1-D Conduction, R_cond       = {R_cond:.4f} K/W")
    print(f"3) Fin Heat Rate (adiabatic), q_j = {q_j:.4f} W   (based on T_base_guess={T_base_guess}°C)")
    print(f"4) Air HTC, h_air               = {h_air:.4f} W/m^2-K")
    print(f"5) Air convection heat transfer rate at base, Q_base    = {Q_base:.4f} W")
    print(f"6) Updated base temperature, T_base    = {T_base_new:.2f} °C")
    print(f"7) Estimated T at board-chip, T_bc     = {T_bc:.2f} °C")
    print(f"7) Estimated hands-touching temperature, T_handstouch     = {T_handstouch:.2f} °C")

if __name__ == "__main__":
    main()
