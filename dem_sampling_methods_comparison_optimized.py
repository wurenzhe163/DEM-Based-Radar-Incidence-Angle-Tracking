#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DEM Sampling Methods Comparison Tool (optimized)
================================================================================

Compares DEM neighborhood-sampling / elevation-reconstruction schemes along
Sentinel-1 across-track transects on Google Earth Engine: weighted average,
simple average, planar fit (4 pts), quadratic fit (9 pts) and bilinear
interpolation.

Optimized over the original release (results unchanged, verified by the
built-in --engine orig|fast A/B switch):

1. Self-contained: the GEE helpers (scene selection, auxiliary-line
   construction, corner geometry, getS1Corners) are imported from the sibling
   SAR_Geometric_Distortion_Analysis_fast.py, which inlines them verbatim --
   the external GEE_Func package is no longer required.
2. One round trip: the per-transect-point reduceRegion maps are replaced by
   chunked image.reduceRegions over per-point MultiPoint features (same
   geometry, same reducer and scale; toList pixel-dedup semantics preserved),
   fetched in a single request.  Reconstruction runs client-side in float64
   with the original formulas and accumulation order.
3. ee.Authenticate() at import time removed; initialization happens in main().

Usage:
  python dem_sampling_methods_comparison_optimized.py --neighbors 4 \
      --algorithm weighted_avg_elevation --engine fast

Dependencies: earthengine-api, numpy, scipy, geopandas, shapely
================================================================================
"""

import argparse
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import ee
import numpy as np
from scipy.optimize import least_squares
import geopandas as gpd
from shapely.geometry import Point

from SAR_Geometric_Distortion_Analysis_fast import (  # verbatim-inlined GEE helpers
    S1_CalDistor, S1Corrector, load_S1collection, _del_bands, _eq_pixels)


def _select_item(collection, i):
    # GEE_Func.GEE_Tools.Select_imageNum, verbatim
    return ee.Image(collection.toList(collection.size()).get(i))


# ================================================================================
# Geometry Processing Functions (verbatim from the original release)
# ================================================================================

def line_to_points(feature: ee.Feature, region: ee.Geometry, scale: int = 30) -> ee.FeatureCollection:
    """Convert line feature to equally spaced point sequence."""
    line_geometry = ee.Feature(feature).geometry().intersection(region, maxError=1)

    coordinates = line_geometry.coordinates()
    start_point = ee.List(coordinates.get(0))
    end_point = ee.List(coordinates.get(-1))

    length = ee.Number(line_geometry.length())
    num_points = length.divide(ee.Number(scale)).floor()

    def interpolate_point(index: ee.Number) -> ee.Feature:
        fraction = ee.Number(index).divide(ee.Number(num_points))

        start_lon = ee.Number(start_point.get(0))
        end_lon = ee.Number(end_point.get(0))
        interpolated_lon = start_lon.add(end_lon.subtract(start_lon).multiply(fraction))

        start_lat = ee.Number(start_point.get(1))
        end_lat = ee.Number(end_point.get(1))
        interpolated_lat = start_lat.add(end_lat.subtract(start_lat).multiply(fraction))

        return ee.Feature(ee.Geometry.Point([interpolated_lon, interpolated_lat]))

    return ee.FeatureCollection(
        ee.List.sequence(1, ee.Number(num_points).max(1)).map(interpolate_point))


def filter_list_length(item: ee.List, min_len: int = 3) -> Optional[ee.List]:
    item_list = ee.List(item)
    return ee.Algorithms.If(item_list.size().gte(min_len), item_list, None)


# ================================================================================
# Neighborhood geometry construction (verbatim from get_neighborhood_info)
# ================================================================================

def _neighborhood_scale(neighborhood_type: str, prj_scale: int) -> int:
    return prj_scale // 2 if neighborhood_type == '4' else prj_scale


def neighborhood_feature(pair: ee.List, neighborhood_type: str, scale: int) -> ee.Feature:
    """[line_id, [lon, lat]] -> Feature whose geometry is the neighborhood
    MultiPoint of get_neighborhood_info, tagged with the point and line id."""
    line_id = ee.List(pair).get(0)
    coord = ee.List(pair).get(1)
    point_geom = ee.Geometry.Point(coord)
    bounds = point_geom.buffer(scale).bounds()
    coords = ee.List(bounds.coordinates().get(0))
    corners = [ee.List(coords.get(i)) for i in range(4)]

    if neighborhood_type == '4':
        members = corners
    elif neighborhood_type == '9':
        edge_centers = [
            ee.Geometry.LineString([corners[i], corners[(i + 1) % 4]]).centroid().coordinates()
            for i in range(4)]
        members = corners + edge_centers + [coord]
    else:
        raise ValueError(f"Unsupported neighborhood type: {neighborhood_type}")

    multipoint = ee.Geometry.MultiPoint(members)
    return ee.Feature(multipoint, {"point_coordinates": coord, "line_id": line_id})


# ================================================================================
# Elevation reconstruction algorithms (verbatim; weighted/avg moved client-side
# with the same formulas and float64 accumulation order)
# ================================================================================

def weighted_avg_client(properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Inverse-distance weighted average (weights in degrees), float64,
    same accumulation order as the server-side weighted_avg_func."""
    lons = properties.get("longitude") or []
    lats = properties.get("latitude") or []
    if len(lons) < 1:
        return None
    px, py = (float(v) for v in properties["point_coordinates"])
    weights = [1.0 / np.sqrt((lon - px) ** 2 + (lat - py) ** 2) for lon, lat in zip(lons, lats)]
    sum_weights = 0.0
    for w in weights:
        sum_weights += w

    def avg(key: str) -> float:
        acc = 0.0
        for value, w in zip(properties.get(key) or [], weights):
            acc += value * w
        return acc / sum_weights

    return {"elevation": avg("elevation"), "angle": avg("angle"),
            "x": avg("x"), "y": avg("y"), "point_coordinates": [px, py]}


def avg_client(properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lons = properties.get("longitude") or []
    if len(lons) < 1:
        return None
    n = len(lons)

    def avg(key: str) -> float:
        values = properties.get(key) or []
        acc = 0.0
        for v in values:
            acc += v
        return acc / n

    return {"elevation": avg("elevation"), "angle": avg("angle"),
            "x": avg("x"), "y": avg("y"),
            "point_coordinates": [float(v) for v in properties["point_coordinates"]]}


def Volum9_func(neighbors_info: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Quadratic polynomial fit: z = ax^2 + by^2 + cxy + dx + ey + f."""

    def equation(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a, b, c, d, e, f = params
        return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f

    def residuals(params: np.ndarray, x: np.ndarray, y: np.ndarray, z_true: np.ndarray) -> np.ndarray:
        return equation(params, x, y) - z_true

    lon = np.array(neighbors_info['longitude'])
    lat = np.array(neighbors_info['latitude'])
    lon_pred, lat_pred = np.array(neighbors_info['point_coordinates'])

    def fit_variable(values: np.ndarray) -> float:
        initial_guess = np.zeros(6)
        result = least_squares(residuals, initial_guess, args=(lon, lat, values))
        return equation(result.x, lon_pred, lat_pred)

    return {
        'elevation': fit_variable(np.array(neighbors_info['elevation'])),
        'angle': fit_variable(np.array(neighbors_info['angle'])),
        'x': fit_variable(np.array(neighbors_info['x'])),
        'y': fit_variable(np.array(neighbors_info['y'])),
        'point_coordinates': [lon_pred, lat_pred]
    }


def Flat4_func(neighbors_info: Dict[str, List[float]]) -> Dict[str, Any]:
    """Plane fit: z = ax + by + c."""
    lon = np.array(neighbors_info['longitude'])
    lat = np.array(neighbors_info['latitude'])
    lon_pred, lat_pred = np.array(neighbors_info['point_coordinates'])

    A = np.vstack([lon, lat, np.ones(len(lon))]).T

    def fit_variable(values) -> float:
        b = np.array(values)
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, b_coeff, c = coeffs
        return a * lon_pred + b_coeff * lat_pred + c

    return {
        'elevation': fit_variable(neighbors_info['elevation']),
        'angle': fit_variable(neighbors_info['angle']),
        'x': fit_variable(neighbors_info['x']),
        'y': fit_variable(neighbors_info['y']),
        'point_coordinates': [lon_pred, lat_pred]
    }


def Bilinear_interp_func(neighbors_info: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Bilinear interpolation from the 4 nearest sampled points."""
    lon = np.array(neighbors_info['longitude'])
    lat = np.array(neighbors_info['latitude'])
    lon_pred, lat_pred = np.array(neighbors_info['point_coordinates'])

    distances = np.sqrt((lon - lon_pred) ** 2 + (lat - lat_pred) ** 2)
    idx = np.argsort(distances)[:4]

    lon_near = lon[idx]
    lat_near = lat[idx]

    matrix = np.array([
        [1, lon_near[0], lat_near[0], lon_near[0] * lat_near[0]],
        [1, lon_near[1], lat_near[1], lon_near[1] * lat_near[1]],
        [1, lon_near[2], lat_near[2], lon_near[2] * lat_near[2]],
        [1, lon_near[3], lat_near[3], lon_near[3] * lat_near[3]]
    ])

    def bilinear_interp(values: np.ndarray) -> float:
        b = values[idx]
        coeffs = np.linalg.solve(matrix, b)
        return coeffs[0] + coeffs[1] * lon_pred + coeffs[2] * lat_pred + coeffs[3] * lon_pred * lat_pred

    return {
        'elevation': bilinear_interp(np.array(neighbors_info['elevation'])),
        'angle': bilinear_interp(np.array(neighbors_info['angle'])),
        'x': bilinear_interp(np.array(neighbors_info['x'])),
        'y': bilinear_interp(np.array(neighbors_info['y'])),
        'point_coordinates': [lon_pred, lat_pred]
    }


RECONSTRUCT = {
    'weighted_avg_elevation': weighted_avg_client,
    'avg_elevation': avg_client,
    'Area_elavation': Flat4_func,
    'Volum_elavation': Volum9_func,
    'Bilinear_interp': Bilinear_interp_func,
}


def _properties_to_neighbors(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Raw sampled properties -> the neighbors_info dict the fitting
    algorithms consume (band lists + the point coordinates)."""
    return {key: (properties.get(key) or []) for key in
            ('longitude', 'latitude', 'elevation', 'angle', 'x', 'y')} | \
        {'point_coordinates': properties['point_coordinates']}


# ================================================================================
# Neighborhood sampling -- two engines
# ================================================================================

def sample_orig(Templist, AOI, Prj_scale, Cal_image, Neighbors):
    """Original release path, verbatim: per-transect-point reduceRegion mapped
    over the nested line lists, fetched in one request.  Kept as the A/B
    reference for --engine orig."""
    list_length = Templist.size().getInfo()

    all_point_lines = []
    for i in range(list_length):
        points = line_to_points(Templist.get(i), region=AOI, scale=Prj_scale)
        coords = points.toList(points.size()).map(
            lambda f: ee.Feature(f).geometry().coordinates())
        all_point_lines.append(coords)

    ee_point_lines = ee.List(all_point_lines)
    ee_point_lines = ee_point_lines.map(
        partial(filter_list_length, min_len=3)).removeAll([None])

    def point_reduce(coord):
        feature = neighborhood_feature(ee.List([-1, coord]), Neighbors,
                                       _neighborhood_scale(Neighbors, Prj_scale))
        return Cal_image.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=feature.geometry(),
            scale=Prj_scale,
            maxPixels=1e9).set('point_coordinates', feature.get('point_coordinates'))

    cal_neighbors = ee_point_lines.map(
        lambda x: ee.List(x).map(point_reduce).removeAll([None]))
    return cal_neighbors.getInfo()


def sample_fast(Templist, AOI, Prj_scale, Cal_image, Neighbors, chunk=4000):
    """Optimized path: all transect points of all lines flattened into one
    ee.List of neighborhood features and sampled through chunked
    image.reduceRegions in a single round trip (same geometry / reducer /
    scale as the original per-point reduceRegion; the toList pixel-dedup
    semantics carry over unchanged)."""
    coords_per_line = Templist.map(lambda f: _line_coords(f, AOI, Prj_scale))
    ids = ee.List.sequence(0, coords_per_line.size().subtract(1))
    pairs = ids.map(lambda i: ee.List([i, coords_per_line.get(i)]))

    def line_features(pair):
        line_id = ee.List(pair).get(0)
        coords = ee.List(ee.List(pair).get(1))
        return coords.map(
            lambda c: neighborhood_feature(ee.List([line_id, c]), Neighbors,
                                           _neighborhood_scale(Neighbors, Prj_scale)))

    # ee.List.flatten is DEEP: build Features per line first (Features are
    # leaves), so flattening only removes the per-line nesting
    features = pairs.map(line_features).flatten()
    n_chunks = ee.Number(features.size()).divide(chunk).ceil()

    def chunk_reduce(k):
        part = ee.List(features.slice(
            ee.Number(k).multiply(chunk), ee.Number(k).add(1).multiply(chunk)))
        reduced = Cal_image.reduceRegions(collection=ee.FeatureCollection(part),
                                          reducer=ee.Reducer.toList(),
                                          scale=Prj_scale)
        stripped = reduced.map(lambda f: ee.Feature(None, f.toDictionary()))
        return stripped.toList(1000000)

    data = ee.List.sequence(0, n_chunks.subtract(1)).map(chunk_reduce).getInfo()
    by_line: Dict[int, List[Dict]] = {}
    for chunk_features in data:
        for feature in chunk_features:
            props = feature["properties"]
            by_line.setdefault(int(props["line_id"]), []).append(props)
    return [by_line[k] for k in sorted(by_line)]


def _line_coords(feature, region, scale):
    points = line_to_points(feature, region=region, scale=scale)
    return points.toList(points.size()).map(
        lambda f: ee.Feature(f).geometry().coordinates())


# ================================================================================
# Main pipeline
# ================================================================================

def main_calculate_neighbor(Templist, AOI, Prj_scale, Cal_image,
                            Neighbors: str = '4',
                            Elevation_model: str = 'weighted_avg_elevation',
                            engine: str = 'fast') -> List[List[Dict]]:
    """Sample the neighborhood of every transect point and reconstruct
    elevation/angle/x/y with the selected algorithm."""
    started = time.time()
    if engine == 'orig':
        neighbors_per_line = sample_orig(Templist, AOI, Prj_scale, Cal_image, Neighbors)
    else:
        neighbors_per_line = sample_fast(Templist, AOI, Prj_scale, Cal_image, Neighbors)

    reconstruct = RECONSTRUCT[Elevation_model]
    points_with_h_angle = []
    for line_points in neighbors_per_line:
        rebuilt = []
        for properties in line_points:
            entry = reconstruct(_properties_to_neighbors(properties)
                                if Elevation_model in ('Area_elavation', 'Volum_elavation',
                                                       'Bilinear_interp')
                                else properties)
            if entry is not None:
                rebuilt.append(entry)
        if len(rebuilt) >= 3:          # filter_list_length(min_len=3), applied post-fetch
            points_with_h_angle.append(rebuilt)
    print(f"sampling+reconstruction ({engine}): {time.time() - started:.1f}s, "
          f"{sum(len(l) for l in points_with_h_angle)} points")
    return points_with_h_angle


def create_sample_data():
    domain_distor_test = ee.FeatureCollection(
        'projects/ee-mrwurenzhe/assets/Test/DEM_ReConstruct')
    mountain = _select_item(domain_distor_test, 0)
    dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation')

    year = '2019'
    start_date = ee.Date(f'{year}-01-01')
    end_date = ee.Date(f'{year}-12-30')
    time_len = end_date.difference(start_date, 'days').abs()
    middle_date = start_date.advance(time_len.divide(ee.Number(2)).int(), 'days')

    return mountain, dem, start_date, end_date, middle_date


def setup_working_directory(out_root: str = None) -> str:
    from datetime import datetime
    base = Path(out_root) if out_root else Path(__file__).resolve().parent
    save_path = base / f"DEM_Sampling_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path.mkdir(parents=True, exist_ok=True)
    os.chdir(save_path)
    print(f"output directory: {save_path}")
    return str(save_path)


def process_s1_data(mountain, start_date, end_date, middle_date):
    aoi = ee.Feature(mountain).geometry()
    s1_ascending, _ = load_S1collection(aoi, start_date, end_date, middle_date, FilterSize=30)

    orbit = 'ASCENDING'
    s1_image = s1_ascending
    projection = s1_image.select(0).projection()
    mask = s1_image.select(0).mask()

    azimuth_edge, rotation_from_north, startpoint, endpoint, coordinates_dict = \
        S1Corrector.getS1Corners(s1_image, aoi, orbit)

    heading = azimuth_edge.get('azimuth')
    s1_azimuth_across = ee.Number(heading).subtract(90.0)
    auxiliary_lines = ee.Geometry.LineString([startpoint, endpoint])

    return (s1_image, aoi, projection, mask, s1_azimuth_across,
            coordinates_dict, auxiliary_lines)


def create_calculation_image(s1_image, dem, projection, mask, aoi, prj_scale=30):
    calc_image = (_eq_pixels(_del_bands(s1_image, 'VV', 'VH').resample('bicubic')).rename('angle')
                  .addBands(ee.Image.pixelCoordinates(projection))
                  .addBands(dem.select('elevation'))
                  .addBands(ee.Image.pixelLonLat())
                  .updateMask(mask)
                  .reproject(crs=projection, scale=prj_scale)
                  .clip(aoi))
    return calc_image


def results_to_geodataframe(points_with_h_angle, elevation_model):
    all_points, all_angles, all_elevations, all_x, all_y = [], [], [], [], []
    for line_points in points_with_h_angle:
        for point_data in line_points:
            all_points.append(Point(point_data['point_coordinates']))
            all_angles.append(point_data['angle'])
            all_elevations.append(point_data['elevation'])
            all_x.append(point_data['x'])
            all_y.append(point_data['y'])
    gdf = gpd.GeoDataFrame({'angle': all_angles, 'elevation': all_elevations,
                            'x': all_x, 'y': all_y, 'geometry': all_points})
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='DEM sampling methods comparison (GEE, optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithms:
  weighted_avg_elevation - inverse-distance weighted average
  avg_elevation          - simple average
  Area_elavation         - planar fit (4 parameters)
  Volum_elavation        - quadratic fit (9 points)
  Bilinear_interp        - bilinear interpolation

Neighborhoods: 4 (buffer corners) | 9 (corners + edge midpoints + point)
""")
    parser.add_argument('--scale', '-s', type=int, default=30,
                        help='projection/sampling scale in meters (default 30)')
    parser.add_argument('--neighbors', '-n', choices=['4', '9'], default='4',
                        help='neighborhood type (default 4)')
    parser.add_argument('--algorithm', '-a', default='weighted_avg_elevation',
                        choices=list(RECONSTRUCT),
                        help='elevation reconstruction algorithm')
    parser.add_argument('--engine', choices=['fast', 'orig'], default='fast',
                        help="fast = single-round-trip chunked reduceRegions (default); "
                             "orig = original per-point reduceRegion path (A/B reference)")
    parser.add_argument('--out-root', default=None,
                        help='root for the timestamped output folder (default: script dir)')
    parser.add_argument('--project', default='ee-mrwurenzhe',
                        help='Earth Engine project')
    return parser


def main():
    args = parse_arguments().parse_args()
    prj_scale, neighbors, elevation_model = args.scale, args.neighbors, args.algorithm

    print("=" * 60)
    print(f"DEM sampling analysis: scale={prj_scale}m neighbors={neighbors} "
          f"algorithm={elevation_model} engine={args.engine}")
    print("=" * 60)

    ee.Initialize(project=args.project)
    setup_working_directory(args.out_root)

    mountain, dem, start_date, end_date, middle_date = create_sample_data()
    (s1_image, aoi, projection, mask, s1_azimuth_across,
     coordinates_dict, auxiliary_lines) = process_s1_data(
        mountain, start_date, end_date, middle_date)
    calc_image = create_calculation_image(s1_image, dem, projection, mask, aoi, prj_scale)

    template_list = S1_CalDistor.AuxiliaryLine2Point(
        s1_azimuth_across, coordinates_dict, auxiliary_lines, aoi, prj_scale)
    print('template list ready')

    points_with_h_angle = main_calculate_neighbor(
        template_list, aoi, prj_scale, calc_image,
        Neighbors=neighbors, Elevation_model=elevation_model, engine=args.engine)

    gdf = results_to_geodataframe(points_with_h_angle, elevation_model)
    gdf.to_file(f'{elevation_model}.shp')
    print(f"saved {len(gdf)} points to {elevation_model}.shp")
    return gdf


if __name__ == '__main__':
    main()
