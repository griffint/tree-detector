import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from pipeline.imagery import meters_per_pixel, DEFAULT_ZOOM


def pixel_to_latlng(
    px_x: float, px_y: float,
    center_lat: float, center_lng: float,
    img_width: int, img_height: int,
    zoom: int = DEFAULT_ZOOM,
) -> tuple[float, float]:
    """Convert a pixel coordinate to lat/lng given the tile center and size."""
    mpp = meters_per_pixel(center_lat, zoom)

    # Pixel offset from image center
    dx_px = px_x - img_width / 2
    dy_px = px_y - img_height / 2

    # Convert to meters
    dx_m = dx_px * mpp
    dy_m = dy_px * mpp

    # Convert meters to degrees (y is inverted: pixel y increases downward, lat increases upward)
    deg_per_m_lat = 1 / 111_320
    deg_per_m_lng = 1 / (111_320 * math.cos(math.radians(center_lat)))

    lat = center_lat - dy_m * deg_per_m_lat
    lng = center_lng + dx_m * deg_per_m_lng
    return lat, lng


def compute_tree_risks(
    detections: pd.DataFrame,
    center_lat: float,
    center_lng: float,
    img_width: int,
    img_height: int,
    fall_multiplier: float = 2.0,  # fall radius = canopy radius * this value
    zoom: int = DEFAULT_ZOOM,
) -> pd.DataFrame:
    """Add real-world position and fall radius to each detected tree.

    New columns added:
        - tree_lat, tree_lng: centroid of the tree in lat/lng
        - canopy_radius_m: estimated canopy radius in meters
        - fall_radius_m: distance the tree could fall (canopy_radius * fall_multiplier)
    """
    mpp = meters_per_pixel(center_lat, zoom)
    rows = []

    for _, det in detections.iterrows():
        # Bounding box center in pixels
        cx = (det["xmin"] + det["xmax"]) / 2
        cy = (det["ymin"] + det["ymax"]) / 2

        # Convert centroid to lat/lng
        tree_lat, tree_lng = pixel_to_latlng(
            cx, cy, center_lat, center_lng, img_width, img_height, zoom
        )

        # Canopy radius: average of box width and height, converted to meters, halved
        box_w = (det["xmax"] - det["xmin"]) * mpp
        box_h = (det["ymax"] - det["ymin"]) * mpp
        canopy_radius_m = (box_w + box_h) / 4  # average diameter / 2

        fall_radius_m = canopy_radius_m * fall_multiplier

        rows.append({
            **det.to_dict(),
            "tree_lat": tree_lat,
            "tree_lng": tree_lng,
            "canopy_radius_m": canopy_radius_m,
            "fall_radius_m": fall_radius_m,
        })

    return pd.DataFrame(rows)


def classify_danger(trees: pd.DataFrame, buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """Classify each tree as 'danger' or 'safe' based on fall radius intersection with buildings.

    Buffers each tree's lat/lng point by its fall_radius_m (converted to approximate degrees),
    then checks if the resulting circle intersects any building polygon.

    Adds a 'danger' boolean column to the trees DataFrame.
    """
    if buildings.empty or trees.empty:
        trees = trees.copy()
        trees["danger"] = False
        return trees

    # Merge all buildings into one geometry for fast intersection checks
    all_buildings = unary_union(buildings.geometry)

    danger_flags = []
    for _, tree in trees.iterrows():
        # Convert fall radius from meters to approximate degrees
        # (using local latitude for longitude scaling)
        fall_deg_lat = tree["fall_radius_m"] / 111_320
        fall_deg_lng = tree["fall_radius_m"] / (111_320 * math.cos(math.radians(tree["tree_lat"])))
        # Average for a roughly circular buffer in degrees
        fall_deg = (fall_deg_lat + fall_deg_lng) / 2

        fall_circle = Point(tree["tree_lng"], tree["tree_lat"]).buffer(fall_deg)
        danger_flags.append(fall_circle.intersects(all_buildings))

    trees = trees.copy()
    trees["danger"] = danger_flags
    return trees
