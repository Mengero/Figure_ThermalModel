# Joint → Actuator / Motor / Gearbox Map

Per-joint mapping of actuator family, motor, and strain-wave gearbox.
Sources: F03 actuator chart "ACTUATORS — Locations and Sizes" (Updated 2025-01-22, FIGURE 3);
F04 actuator chart (FIGURE 8); and the "F04 joint motor design tracking" Google Sheet (F03↔F04 family mapping).

## F03 (current robot — our LPTN test data, C1078)

| Joint | Actuator family | Motor (code) | Motor size | Gearbox (strain wave) | Peak torque | Phase R (Ω) |
|-------|-----------------|--------------|:----------:|-----------------------|:-----------:|:-----------:|
| shoulder_j1 | Jade | SM7211 | SM72 | CSG-20-80 | 165 N·m | 0.096 |
| shoulder_j2 | Citrine | SM6712 | SM67 | CSG-17-50 | 91 N·m | 0.098 |
| upper_arm_twist (Humeral Twist) | Citrine | SM6712 | SM67 | CSG-17-50 | 91 N·m | 0.098 |
| elbow | Citrine | SM6712 | SM67 | CSG-17-50 | 91 N·m | 0.098 |
| wrist_roll | Papaya | SM4406 | SM44S | CSD-14-100 | 31 N·m | 0.45 |
| wrist_pitch | Papaya | SM4406 | SM44S | CSD-14-100 | 31 N·m | 0.45 |
| wrist_yaw | Papaya | SM4406 | SM44S | CSD-14-100 | 31 N·m | 0.45 |
| neck_yes | Papaya | SM4406 | SM44S | CSD-14-100 | 31 N·m | 0.45 |
| neck_no | Papaya | SM4406 | SM44S | CSD-14-100 | 31 N·m | 0.45 |
| hip_x | Jade | SM7211 | SM72 | CSG-20-80 | 165 N·m | 0.096 |
| hip_z | Jade | SM7211 | SM72 | CSG-20-80 | 165 N·m | 0.096 |
| ankle_y | Jade | SM7211 | SM72 | CSG-20-80 | 165 N·m | 0.096 |
| hip_y | Lapis | SM8512 | SM85 | CSG-25-80 | 332 N·m | 0.078 |
| knee | Lapis | SM8512 | SM85 | CSG-25-80 | 332 N·m | 0.078 |
| spine_x | Sapphire | SM8512 | SM85 | CSD-25-80 | 152 N·m | 0.078 |
| spine_z | Sapphire | SM8512 | SM85 | CSD-25-80 | 152 N·m | 0.078 |
| ankle_x | Carrot | SM4418 | SM44L | CSG-14-80 | 61 N·m | 0.72 |
| finger flex / thumb rotate (×12) | — | 16 mm slotless planetary | — | planetary | 4.8 N·m | — |

Notes:
- Motor code `SMxxyy`: first two digits = motor size (SM67/SM72/SM85/SM44); SM44 splits into **L** (Carrot, longer stack) and **S** (Papaya) — confirmed in the design-tracking sheet.
- Sapphire and Lapis share the **SM85** motor but different gearboxes (CSD-25-80 vs CSG-25-80) → same phase R (0.078 Ω), different output torque.
- Citrine gearbox shown as **CSG-17-50 / 91 N·m** on the 2025-01-22 chart (an earlier chart listed CSG-17-80 / 113 N·m).
- Phase resistances are the user-provided F03 SM-series values (SM85=0.078, SM72=0.096, SM67=0.098, SM44L=0.72, SM44S=0.45 Ω).

## F04 (next-gen equivalents)

Joint → F04 family mapping from the design-tracking sheet; motor/gearbox from FIGURE 8.

| Joint | F04 family | Motor | Gearbox (strain wave) | Peak torque | Type |
|-------|------------|:-----:|-----------------------|:-----------:|------|
| neck_yes / neck_no | Snowdrop | SM30 | CSF-8-80 | 14 N·m | Hot Dog |
| spine_x | Iris | SM59 | CSG-17-80 | 260 N·m | Hot Dog |
| spine_z | Flax | SM62 | CSD-17-80 | 140 N·m | Hot Dog |
| shoulder_j2 / upper_arm_twist / elbow | Flax | SM62 | CSD-17-80 | 140 N·m | Hot Dog |
| shoulder_j1 | Iris | SM59 | CSG-17-80 | 260 N·m | Hot Dog |
| hip_x / ankle_y | Iris | SM59 | CSG-17-80 | 260 N·m | Hot Dog |
| hip_z | Flax | SM62 | CSD-17-80 | 140 N·m | Hot Dog |
| ankle_x | Flax | SM62 | CSD-17-80 | 140 N·m | Hot Dog |
| wrist_pitch | Sakura | SM32 | CSF-11-100 | 43 N·m | Tandem |
| wrist_yaw / wrist_roll | Hibiscus | SM50 | CSD-14-80 | 75 N·m | Hot Dog |
| knee | Dahlia | SM80 | CSG-20-80 | 450 N·m | Hot Dog |
| hip_y | Lilac | SM95 | CSG-20-50 | 280 N·m | Pancake |

### F04 family reference (FIGURE 8)

| Family (count) | Motor | Gearbox | Peak torque | Type |
|----------------|:-----:|---------|:-----------:|------|
| Snowdrop (3) | SM30 | CSF-8-80 | 14 N·m | Hot Dog |
| Sakura (2) | SM32 | CSF-11-100 | 43 N·m | Tandem |
| Hibiscus (4) | SM50 | CSD-14-80 | 75 N·m | Hot Dog |
| Iris (7) | SM59 | CSG-17-80 | 260 N·m | Hot Dog |
| Flax (10) | SM62 | CSD-17-80 | 140 N·m | Hot Dog |
| Flax Max (3) | SM62 | CSD-17-80 | 140 N·m | Hot Dog, XL Bearing |
| Dahlia (2) | SM80 | CSG-20-80 | 450 N·m | Hot Dog |
| Lilac (2) | SM95 | CSG-20-50 | 280 N·m | Pancake |

Note: "Flax Max" (SM62 / CSD-17-80, XL bearing) is a Flax variant; its specific joint assignment isn't given in the chart/sheet.

## Strain-wave gearbox materials (internal — do not distribute)

| Component | Material |
|-----------|----------|
| Circular Spline | Nodular / ductile cast iron |
| Flex Spline | 4140 alloy steel (or similar) |
| Wave Generator core | 52100 bearing steel |
| Wave Generator bearing | Spring steel |
| Wave Generator balls | 52100 (assumed) |

### Working material properties for gearbox thermal calcs

Per agreement, use the **circular spline (ductile cast iron)** properties, taking the mid-point of the published ranges:

| Property | Range | **Value used** |
|----------|-------|:--------------:|
| Density | 7100–7200 kg/m³ | **7150 kg/m³** |
| Specific heat capacity | 460–515 J/(kg·K) | **488 J/(kg·K)** |
| Thermal conductivity | 31–36 W/(m·K) | **33.5 W/(m·K)** |

### Flexspline contact geometry & R_GB→Output decomposition

R_GB→Output = R_cyl + R_cont, with R_cyl = ln(OD/ID)/(2π·k·H), k = 33.5 W/(m·K) (ductile iron, above):

| Gearbox | Actuator | OD [mm] | ID [mm] | Contacting height H [mm] | R_GB→Output measured [K/W] | R_cyl [K/W] | R_cont [K/W] |
|---------|----------|:-------:|:-------:|:------------------------:|:--------------------------:|:-----------:|:------------:|
| CSG-17 GR50 | Citrine | 48 | 43.335 | 9.0 | 4.00 | 0.054 | 3.95 |
| CSG-20 | Jade (hip_x) | 54 | 51.023 | 10.5 | 10.60 ⚠ | 0.026 | 10.57 ⚠ |
| CSG-25 | Lapis (hip_y) | 67 | 63.663 | 13.0 | 2.83 | 0.019 | 2.81 |

Findings:
- **R_cyl is negligible** (<2% of total) — the gearbox→output resistance is essentially all contact resistance.
- **⚠ hip_x (CSG-20) is an outlier** (measured 10.6 K/W vs ~3.2 expected from area scaling; per-area conductance ≈53 W/m²K vs 137–207 for the others). **Excluded from the correlation** pending re-examination of its fit / output interface.
- Plot: `F03_actuator_temp/_Rcont_vs_height.png`.

### Contact-resistance correlation (Citrine + Lapis only, hip_x excluded)

Contact face area defined as **A = π·ID·H** (flexspline ID × contacting height):

| Gearbox | A [mm²] | R_cont [K/W] | h_c = 1/(R_cont·A) [W/(m²·K)] |
|---------|:-------:|:------------:|:-----------------------------:|
| CSG-17 GR50 (Citrine) | 1225 | 3.95 | 207 |
| CSG-25 (Lapis, hip_y) | 2600 | 2.81 | 137 |

Correlations (use for analogy to unmeasured gearboxes):
- **Power law (exact 2-point fit, preferred interpolator):** `R_cont [K/W] = 98.7 · A^(−0.453)`, A in mm².
- **Constant contact conductance (physical bound):** `R_cont = 1/(h_c·A)` with h_c ≈ 168 W/(m²·K) (±20% on the two anchors).
- Then **R_GB→Output = R_cyl + R_cont** with R_cyl = ln(OD/ID)/(2π·k·H), k = 33.5 W/(m·K).

Validity: interpolation A ≈ 1200–2600 mm². Exponent 0.45 (≠1) suggests contact pressure / constriction effects vary with size; the two models diverge ~2× below A≈600 mm² (small CSD-14 wrist gearboxes) — a third measurement on a small actuator (e.g. Papaya wrist) would discriminate. Plot: `F03_actuator_temp/_Rcont_correlation.png`.
