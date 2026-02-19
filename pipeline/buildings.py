import osmnx as ox
import geopandas as gpd
from shapely.geometry import box


# Set a shorter timeout so we don't hang on slow Overpass responses
ox.settings.requests_timeout = 30


def fetch_buildings(north: float, south: float, east: float, west: float) -> gpd.GeoDataFrame:
    """Fetch building footprints from OSM within a bounding box.

    Returns a GeoDataFrame with building polygon geometries in EPSG:4326 (lat/lng).
    Returns an empty GeoDataFrame if no buildings are found.
    """
    try:
        # bbox format is (left, bottom, right, top) = (west, south, east, north)
        buildings = ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})
        # Keep only polygon geometries (drop points, lines)
        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
        return buildings
    except (ox._errors.InsufficientResponseError, Exception):
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
