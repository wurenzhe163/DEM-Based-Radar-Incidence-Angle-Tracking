# DEM-Based Radar Incidence Angle Tracking for Distortion Analysis Without Orbital Data

Code for the method presented in Wu et al. (2024), *IEEE TGRS* 62, 1–13
([DOI: 10.1109/TGRS.2024.3456118](https://doi.org/10.1109/TGRS.2024.3456118)).

The method maps SAR geometric distortion (layover, shadow, foreshortening)
over mountainous terrain from a DEM alone. Instead of orbit state vectors,
the radar geometry comes from the Sentinel-1 `angle` band itself: across-track
transects follow the scene heading (azimuth − 90°), each transect samples
DEM elevation and the tracked incidence angle, and local extrema of the
elevation profile are tested against the incidence angle with simple
arctan(dh/d) criteria. Everything runs on Google Earth Engine; per-point
detections are merged, rasterized to 30 m UTM grids and written locally.

## Contents

- `SAR_Geometric_Distortion_Analysis_fast.py` — the complete pipeline in one
  self-contained file (scene selection, transect construction, neighborhood
  sampling, the four-class decision rules, rasterization). The GEE helper
  functions are inlined, so no other project files are needed.
- `dem_sampling_methods_comparison_optimized.py` — a companion study
  comparing DEM neighborhood-sampling / reconstruction schemes along the same
  transects (inverse-distance weighted average, simple average, planar fit,
  quadratic fit, bilinear interpolation; 4- or 9-point neighborhoods). It is
  equally optimized and self-contained (imports the inlined helpers above,
  samples all transect points through chunked `reduceRegions` in one round
  trip) and carries an `--engine orig|fast` switch that reproduces the
  original per-point sampling path: on the sample AOI both engines return
  identical point sets with values equal to float rounding (max difference
  2.7e-10 on elevation), the fast engine being ~3.7× faster (6 s vs 22 s).

## Runtime

On the Southeast Tibet production fishnet (0.25° cells, 2019 Sentinel-1 GRD
and NASADEM, both pulled from GEE at run time):

| configuration | time |
|---|---|
| one cell × one orbit, serial | 23–30 s |
| one cell × one orbit, amortized at `--workers 8` | ~6 s |
| full 614-cell × 2-orbit production | ~2 h |

The DEM sampling study runs a whole sample AOI (≈5,800 transect points) in
~6 s (`--engine fast`, the default).

Outputs are verified against a reference run of the published
implementation: identical pixel classifications and per-class counts.

## Usage

Requires the `earthengine-api` Python package and an initialized Earth Engine
project (`earthengine authenticate`), plus `numpy/scipy/rasterio/geopandas`
for the local classification and rasterization.

```bash
python SAR_Geometric_Distortion_Analysis_fast.py \
    --fishnet Southest_doom_fishnet.shp \
    --max-cell 614 \
    --workers 8 \
    --out-dir /path/to/output
```

Inputs are a fishnet shapefile (feature *k* produces tiles `000000`…, exactly
as in the production run) and 2019 Sentinel-1 GRD / NASADEM. Outputs per cell
and orbit are `<i>_Distortion<Orbit>.tif` and `<i>_First_derivative<Orbit>.tif`
(30 m UTM, int16, nodata −128; class codes 1 left layover, 5 right layover,
7 shadow, 9 foreshortening, additive when they coincide).

Cells whose rasters already exist are skipped, so interrupted runs resume
where they stopped. Useful options: `--cells 0,1,2` processes specific cells,
`--orbit ASCENDING|DESCENDING` restricts the pass direction, `--engine loop`
runs the per-candidate reference classifier instead of the vectorized one
(identical results, slower).

## Citing

If you use this code, please cite:

> Wu, R., Liu, G., Lv, J., Bao, X., Hong, R., Yang, Z., Wu, S., Xiang, W., &
> Zhang, R. (2024). DEM-based radar incidence angle tracking for geometric
> distortion detection without orbit state information. *IEEE Transactions on
> Geoscience and Remote Sensing*, 62, 1–13.
> https://doi.org/10.1109/TGRS.2024.3456118

Contact: Renzhe Wu <rswrz@hnas.ac.cn> — Aerospace Information Research
Institute, Henan Academy of Sciences, Zhengzhou, China.
