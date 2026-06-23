# UPS-operation (real-robot) datasets — left arm

Real robot operation logs (long format: `timestamp_utc, actuator_name, power_W, Tmotor_degC,
Tambient_degC, Tbatt_min/max_degC`). `power_W = ¾·iq²`; real copper loss = `1.5·iq²·R(T) =
2·power_W·R(T)`. Battery **max** cell temp is used as the time-varying torso boundary
(fallback 40 °C). Validate with `python studies/field_accuracy.py --csv data/ups/<cleaned>.csv --dt 4`.

## Files

- **`c_1261_ups.csv`** / **`c_1261_ups_cleaned.csv`** — robot c_1261, 2026-04-28 → 05-08.
  Comes in 4 bursts (long gaps between). Burst 1 (~34 min) has real load → used for the
  arm-model held-out validation. `_cleaned` has `real_power_W`, `Tamb_filled`, `Tbatt_max_filled`.

- **`ups_0517_idle.csv`** — 2026-05-17, 21.4 h continuous. **NOT usable for UPS power
  validation**: the arm is essentially idle, copper power ≈ 0 (mean 0.001 W, max 0.39 W).
  **Kept as a reference for the J1 ↔ battery-cell thermal coupling**: with no copper load,
  the motor temperatures (esp. J1/shoulder) visibly track the battery cell's sawtooth
  (battery 36–41 °C; motors 46–64 °C). Good evidence that the torso/battery is the dominant
  boundary for the proximal joints. See `ups_0517_motors_batt_power.png`.
