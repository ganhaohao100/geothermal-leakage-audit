# Data sources

## Open inputs (redistributable)

| Layer | Source | Identifier |
|---|---|---|
| Surface heat flow | IHFC Global Heat Flow Database, Release 2024, v2026.03 | doi:10.5880/fidgeo.2024.014 |
| Active faults | China Active Fault Database v2023, distributed as CN-faults v0.4.0 | doi:10.12031/activefault.china.400.2023.db |
| National boundary, coastline | china-geospatial-data v0.4.0 (China GMT community) | package release v0.4.0 |

Related publications: Fuchs et al. (2023, Tectonophysics 863, 229976);
Fuchs et al. (2024, GFZ Data Services); Neumann et al. (2026, ESSD 18, 4639-4668);
Wu et al. (2024) for the fault database.

## Restricted inputs (not redistributed)

- Thermal-spring inventory: coordinates, temperature and discharge.
- Geothermal-well inventory: coordinates, depth and measured temperature.
- `features.npz`, which embeds both inventories as point arrays and as
  smoothed fields.

These are held by the Institute of Hydrogeology and Environmental Geology,
Chinese Academy of Geological Sciences. They are available from the
corresponding author on reasonable request and with the permission of the
data holders.

## What this package does not contain

No national boundary, coastline, point-level spring or well record, or per-cell
raster is redistributed. The candidate-area table gives grid-cell centroids only,
with area and favourability; polygon vertices are not included.

## Derived rasters (not redistributed)

Per-cell rasters, fold assignments and the fitted screening surface are not
included. The ten candidate areas are summarised in
`results/targets_well_geo.csv`, which gives grid-cell centroids, areas and
favourability; polygon vertices are not included. A reader holding equivalent
label data can rebuild the surface with the code here and the open layers above.
