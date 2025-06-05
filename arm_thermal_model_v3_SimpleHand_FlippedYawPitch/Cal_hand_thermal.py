# Given values
R_HS_a = 4.3      # degC/W
R_YAW_HS = 1.9    # degC/W
Q_YAW = 1.2         # W
Q_hand = 7.32        # W
T_amb = 25        # degC

# Calculate T_HS
T_HS = (Q_YAW + Q_hand) * R_HS_a + T_amb

# Calculate T_YAW
T_YAW = Q_YAW * R_YAW_HS + T_HS

print(f"T_HS = {T_HS:.2f} °C")
print(f"T_YAW = {T_YAW:.2f} °C")