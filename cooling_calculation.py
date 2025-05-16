import numpy as np
import matplotlib.pyplot as plt

class CoolingCalculator:
    def __init__(self):
        # Air properties at 300K (approximate room temperature)
        self.k_air = 0.0263  # Thermal conductivity of air [W/m·K]
        self.v_air = 1.568e-5  # Kinematic viscosity of air [m²/s]
        self.Pr_air = 0.707  # Prandtl number of air [-]
    
    @staticmethod
    def celsius_to_kelvin(T_celsius):
        """Convert Celsius to Kelvin"""
        return T_celsius + 273.15
    
    @staticmethod
    def kelvin_to_celsius(T_kelvin):
        """Convert Kelvin to Celsius"""
        return T_kelvin - 273.15
        
    def calculate_h(self, velocity, length, natural_convection=False):
        """
        Calculate heat transfer coefficient
        
        Args:
            velocity (float): Air velocity [m/s]
            length (float): Characteristic length [m]
            natural_convection (bool): If True, use natural convection coefficient
            
        Returns:
            float: Heat transfer coefficient [W/m²·K]
        """
        if natural_convection:
            return 2.0  # Natural convection coefficient [W/m²·K]
        
        # Calculate Reynolds number
        Re = velocity * length / self.v_air
        
        # Calculate Nusselt number
        Nu = 0.664 * np.sqrt(Re) * (self.Pr_air ** (1/3))
        
        # Calculate heat transfer coefficient
        h = self.k_air * Nu / length
        
        return h
    
    def calculate_temperature(self, Q, m, cp, h, A, t, T_inf_celsius, T_initial_celsius=None):
        """
        Calculate transient temperature
        
        Args:
            Q (float): Initial heat input [J]
            m (float): Mass [kg]
            cp (float): Specific heat capacity [J/kg·K]
            h (float): Heat transfer coefficient [W/m²·K]
            A (float): Surface area [m²]
            t (float): Time [s]
            T_inf_celsius (float): Ambient temperature [°C]
            T_initial_celsius (float): Initial temperature [°C], if None uses T_inf + Q/(m*cp)
            
        Returns:
            float: Temperature at time t [°C]
        """
        T_inf_kelvin = self.celsius_to_kelvin(T_inf_celsius)
        
        if T_initial_celsius is None:
            # Calculate initial temperature rise from heat pulse
            T_initial_kelvin = T_inf_kelvin + Q/(m*cp)
        else:
            T_initial_kelvin = self.celsius_to_kelvin(T_initial_celsius)
        
        # Calculate temperature difference from initial condition
        delta_T = T_initial_kelvin - T_inf_kelvin
        T_kelvin = delta_T * np.exp(-h*A/(m*cp)*t) + T_inf_kelvin
        
        return self.kelvin_to_celsius(T_kelvin)
    
    def calculate_cyclic_temperature(self, Q, m, cp, h, A, T_inf_celsius, 
                                   pulse_duration, cooling_duration, num_cycles):
        """
        Calculate temperature evolution with periodic heat pulses
        
        Args:
            Q (float): Heat input per pulse [J]
            m (float): Mass [kg]
            cp (float): Specific heat capacity [J/kg·K]
            h (float): Heat transfer coefficient [W/m²·K]
            A (float): Surface area [m²]
            T_inf_celsius (float): Ambient temperature [°C]
            pulse_duration (float): Duration of heat pulse [s]
            cooling_duration (float): Cooling duration between pulses [s]
            num_cycles (int): Number of heating/cooling cycles
            
        Returns:
            tuple: (time array, temperature array)
        """
        cycle_duration = pulse_duration + cooling_duration
        points_per_cycle = 1000  # Number of points to plot per cycle
        
        t_cycles = []
        T_cycles = []
        T_current = T_inf_celsius
        
        for cycle in range(num_cycles):
            # Heating phase - minimum 2 points for start and end
            num_heating_points = max(2, int(points_per_cycle * pulse_duration/cycle_duration))
            t_heating = np.linspace(cycle*cycle_duration, 
                                  cycle*cycle_duration + pulse_duration,
                                  num_heating_points)
            # Calculate temperature after heat pulse
            T_after_pulse = T_current + Q/(m*cp)
            T_heating = np.full_like(t_heating, T_after_pulse)
            
            # Cooling phase
            num_cooling_points = max(2, int(points_per_cycle * cooling_duration/cycle_duration))
            t_cooling = np.linspace(cycle*cycle_duration + pulse_duration,
                                  (cycle + 1)*cycle_duration,
                                  num_cooling_points)
            T_cooling = self.calculate_temperature(0, m, cp, h, A, 
                                                t_cooling - (cycle*cycle_duration + pulse_duration),
                                                T_inf_celsius, T_after_pulse)
            
            # Store the results
            t_cycles.extend(t_heating)
            t_cycles.extend(t_cooling)
            T_cycles.extend(T_heating)
            T_cycles.extend(T_cooling)
            
            # Update initial temperature for next cycle
            T_current = T_cooling[-1]
        
        return np.array(t_cycles), np.array(T_cycles)
    
    def plot_cyclic_comparison(self, Q, m, cp, h_forced, h_natural, A, T_inf_celsius,
                             pulse_duration, cooling_duration, num_cycles):
        """
        Plot comparison of forced and natural convection with cyclic heating
        
        Args:
            Q (float): Heat input per pulse [J]
            m (float): Mass [kg]
            cp (float): Specific heat capacity [J/kg·K]
            h_forced (float): Forced convection heat transfer coefficient [W/m²·K]
            h_natural (float): Natural convection heat transfer coefficient [W/m²·K]
            A (float): Surface area [m²]
            T_inf_celsius (float): Ambient temperature [°C]
            pulse_duration (float): Duration of heat pulse [s]
            cooling_duration (float): Cooling duration between pulses [s]
            num_cycles (int): Number of heating/cooling cycles
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate and plot forced convection
        t_forced, T_forced = self.calculate_cyclic_temperature(
            Q, m, cp, h_forced, A, T_inf_celsius, pulse_duration, cooling_duration, num_cycles)
        plt.plot(t_forced/60, T_forced, 
                label=f'Forced Convection (h = {h_forced:.2f} W/m²·K)')
        
        # Calculate and plot natural convection
        t_natural, T_natural = self.calculate_cyclic_temperature(
            Q, m, cp, h_natural, A, T_inf_celsius, pulse_duration, cooling_duration, num_cycles)
        plt.plot(t_natural/60, T_natural, 
                label=f'Natural Convection (h = {h_natural:.2f} W/m²·K)')
        
        plt.title('Cyclic Heating and Cooling Comparison')
        plt.xlabel('Time [minutes]')
        plt.ylabel('Temperature [°C]')
        plt.grid(True)
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create calculator instance
    calc = CoolingCalculator()
    
    # Example parameters
    Q = 2000  # Heat input per pulse [J]
    m = 0.5  # Mass [kg]
    cp = 903  # Specific heat capacity [J/kg·K]
    A = 0.05  # Surface area [m²]
    V = 0.00006  # Volume [m³]
    L = V/A  # Characteristic length [m]
    u = 0.3  # Air velocity [m/s]
    T_inf = 25  # Ambient temperature [°C]
    
    # Calculate heat transfer coefficients
    h_forced = calc.calculate_h(u, L)
    h_natural = calc.calculate_h(u, L, natural_convection=True)
    
    print(f"Forced convection h: {h_forced:.2f} W/m²·K")
    print(f"Natural convection h: {h_natural:.2f} W/m²·K")
    
    # Plot cyclic heating and cooling comparison
    print("\nPlotting cyclic heating and cooling comparison...")
    calc.plot_cyclic_comparison(
        Q=Q,
        m=m,
        cp=cp,
        h_forced=h_forced,
        h_natural=h_natural,
        A=A,
        T_inf_celsius=T_inf,
        pulse_duration=1.0,  # ? second heating
        cooling_duration=5*60,  # ? minutes cooling
        num_cycles=100  # Number of heating/cooling cycles
    ) 