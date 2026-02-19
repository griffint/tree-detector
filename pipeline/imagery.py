import math
import os
import httpx
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

GOOGLE_MAPS_API_KEY = os.environ["GOOGLE_MAPS_API_KEY"]
STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"

DEFAULT_ZOOM = 19
DEFAULT_SIZE = 640


def meters_per_pixel(lat: float, zoom: int = DEFAULT_ZOOM) -> float:
    """Meters per pixel at a given latitude and zoom level."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def tile_bounds(lat: float, lng: float, size: int = DEFAULT_SIZE, zoom: int = DEFAULT_ZOOM) -> dict:
    """Compute the lat/lng bounds of a tile centered on (lat, lng)."""
    mpp = meters_per_pixel(lat, zoom)
    half_width_m = (size / 2) * mpp
    half_height_m = (size / 2) * mpp

    # Approximate degrees per meter
    deg_per_m_lat = 1 / 111_320
    deg_per_m_lng = 1 / (111_320 * math.cos(math.radians(lat)))

    return {
        "north": lat + half_height_m * deg_per_m_lat,
        "south": lat - half_height_m * deg_per_m_lat,
        "east": lng + half_width_m * deg_per_m_lng,
        "west": lng - half_width_m * deg_per_m_lng,
    }


def fetch_satellite_tile(
    lat: float, lng: float, zoom: int = DEFAULT_ZOOM, size: int = DEFAULT_SIZE
) -> Image.Image:
    """Fetch a satellite tile from Google Maps Static API."""
    params = {
        "center": f"{lat},{lng}",
        "zoom": zoom,
        "size": f"{size}x{size}",
        "scale": 2,
        "maptype": "satellite",
        "key": GOOGLE_MAPS_API_KEY,
    }
    resp = httpx.get(STATIC_MAP_URL, params=params)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")
