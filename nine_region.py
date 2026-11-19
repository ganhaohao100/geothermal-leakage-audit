# -*- coding: utf-8 -*-
"""Nine-region tectonic-regime index used to evaluate slip (Ts) and
dilation (Td) tendency on mapped fault segments.

Each polygon is matched in list order; the first containing box wins.
Tuples are (lon_min, lon_max, lat_min, lat_max, regime, SHmax_azimuth_deg).
The index is a coarse screening proxy, not a stress inversion. See the main
text, Section 2.2, for its stated limitations.
"""

STRESS_DOMAINS = [
    (118.0, 136.0, 17.0, 26.0, "thrust", 115.0),   # southeast coast - Taiwan
    (105.0, 122.0, 22.0, 33.0, "strike",  75.0),   # South China
    (102.0, 112.0, 33.0, 41.0, "strike",  70.0),   # Ordos and margins
    (108.0, 124.0, 33.0, 42.5, "normal",  75.0),   # North China
    (118.0, 136.0, 42.5, 54.5, "normal",  85.0),   # Northeast China
    (95.0, 108.0, 37.0, 43.0, "strike",   30.0),   # Hexi Corridor - Alxa
    (73.0,  95.0, 35.0, 49.0, "strike",   15.0),   # Tarim - Junggar
    (73.0, 103.0, 26.0, 40.0, "strike",   45.0),   # Tibetan Plateau
    (96.0, 104.0, 20.0, 30.0, "strike",   30.0),   # Three Rivers
]

REGIME = {"normal": (0.75, 1.00, 0.50), "strike": (1.35, 1.00, 0.65),
          "thrust": (1.90, 1.15, 1.00)}          # (SH/sv, S2/sv, Sh/sv)

AXES = {"normal": ("z", "e1", "e2"), "strike": ("e1", "z", "e2"),
        "thrust": ("e1", "e2", "z")}
MAGS = {"normal": (1.00, 0.75, 0.50), "strike": (1.35, 1.00, 0.65),
        "thrust": (1.90, 1.15, 1.00)}


def stress_regime(lon, lat):
    for lo0, lo1, la0, la1, reg, az in STRESS_DOMAINS:
        if lo0 <= lon <= lo1 and la0 <= lat <= la1:
            return reg, az
    return "strike", 70.0
