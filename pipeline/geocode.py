import os
import httpx
from dotenv import load_dotenv

load_dotenv()

GOOGLE_MAPS_API_KEY = os.environ["GOOGLE_MAPS_API_KEY"]
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def geocode(address: str) -> tuple[float, float]:
    """Convert an address string to (lat, lng)."""
    resp = httpx.get(GEOCODE_URL, params={"address": address, "key": GOOGLE_MAPS_API_KEY})
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "OK":
        raise ValueError(f"Geocoding failed: {data['status']}")
    location = data["results"][0]["geometry"]["location"]
    return location["lat"], location["lng"]
