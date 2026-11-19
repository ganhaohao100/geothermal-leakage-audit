# -*- coding: utf-8 -*-
"""Path configuration.

GEOTHERMAL_DATA  directory holding the input rasters (features.npz) and any
                 external tables. Defaults to ./data.
GEOTHERMAL_WORK  scratch directory for caches. Defaults to ./work.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("GEOTHERMAL_DATA", os.path.join(_HERE, "data"))
WORK_DIR = os.environ.get("GEOTHERMAL_WORK", os.path.join(_HERE, "work"))
OUTD = os.path.join(_HERE, "output")

for _d in (DATA_DIR, WORK_DIR, OUTD):
    os.makedirs(_d, exist_ok=True)
