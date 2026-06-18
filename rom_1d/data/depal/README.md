# Depal-operation (real-robot) datasets — left arm

Real depalletizing runs (high copper load, unlike the mostly-idle UPS data). Long format:
`timestamp_utc, actuator_name, power_W, Tmotor_degC, Tambient_degC, Tbatt_min/max_degC`.
`power_W = ¾·iq²`; real copper loss = `1.5·iq²·R(T) = 2·power_W·R(T)`. Battery **max** cell
temp is the time-varying torso boundary (fallback 40 °C).

## Files

- **`c_1092_depal.csv`** / **`c_1092_depal_cleaned.csv`** / `c_1092_depal.png` — **GOOD case.**
  Robot c_1092, 2026-06-16, 2.7 h. Clean (all 7 joints, no dead sensors). Real depal load:
  J2 spikes to ~110–114 °C, copper power to ~120 W; clear active → idle-pause (35–65 min) →
  active cycles; battery 37 → 48 °C. **Use this for high-load validation** (`_cleaned` has
  `real_power_W`, `Tamb_filled`, `Tbatt_max_filled`).

- **`c_1076_depal.csv`** / `c_1076_depal.png` — **DEFECTIVE case (kept as a reference).**
  Robot c_1076, 2026-06-17, 3.2 h. The **YAW motor-temp sensor is dead** (reads −72.2 °C),
  plus a few single-sample glitches. Real depal load otherwise (power to ~120 W, J2 to ~110 °C).
  Kept to document a bad-data example; not cleaned. Mask `Tmotor < 0` before any use.
