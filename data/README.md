# Input schema

`features.npz` is not redistributed: it contains the coordinates of the
thermal-spring and geothermal-well records. This file documents the keys so
that a reader holding equivalent label data can rebuild it.

## Grid and mask

| Key | Shape | Meaning |
|---|---|---|
| `lon_c` | (1260,) | cell-centre longitudes, 73.0-136.0 E at 0.05 deg |
| `lat_c` | (750,) | cell-centre latitudes, 17.0-54.5 N at 0.05 deg |
| `mask` | (750, 1260) | boolean land mask inside the land mask of mainland China |
| `cell_km2` | (750, 1260) | cell area, km2 |

## Static predictors (not derived from the target)

| Key | Meaning |
|---|---|
| `hf_bg` | background heat flow, Gaussian sigma = 1.50 deg |
| `hf_res` | residual heat flow, sigma = 0.35 deg |
| `hf_n` | heat-flow site density |
| `hf_grad` | horizontal gradient magnitude of `hf_bg` |
| `f_dist` | distance to the nearest 25 km fault chord midpoint |
| `f_den` | fault length density |
| `f_end` | fault endpoint density |
| `Ts_mean`, `Ts_max` | slip-tendency mean and maximum |
| `Td_mean`, `Td_max` | dilation-tendency mean and maximum |

`Ts` and `Td` are evaluated with the nine-region index in `nine_region.py`,
a reference depth of 3 km and a fixed 70 deg dip. `Ts` is the ratio of shear
traction to effective normal traction on the plane, `Ts = tau / sigma_n'`, with
`sigma_n' = sigma_n - P`; `Td` is the classic dilation tendency,
`Td = (sigma_1 - sigma_n') / (sigma_1 - sigma_3)`, where `sigma_1` and `sigma_3`
are the maximum and minimum effective principal stresses.

## Target-derived fields

| Key | Meaning |
|---|---|
| `s_T` | smoothed spring temperature, sigma = 0.40 deg |
| `s_lgQ` | smoothed log10(discharge + 1), sigma = 0.40 deg |
| `s_heat` | smoothed convective heat release, sigma = 0.40 deg |
| `s_den` | spring site density, sigma = 0.10 deg |

## Labels

| Key | Meaning |
|---|---|
| `lab_spring` | spring-type target, 1 on a cell containing at least one spring |
| `lab_well` | well-type target, 1 on a cell containing at least one well |

## Point arrays (restricted)

`slon`, `slat`, `sT`, `sQ`, `H` (2068 springs) and `wlon`, `wlat`, `wT`
(1118 wells). These are the reason the bundle cannot be released.
