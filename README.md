# Tree Risk Assessment Tool

A Marimo notebook that takes a property address, detects trees from satellite imagery, and highlights which trees are within fall distance of structures.

![Example output](image.png)

## How it works

1. **Geocode** the address to lat/lng via Google Geocoding API
2. **Fetch** a satellite tile from Google Maps Static API
3. **Detect** tree crowns using [DeepForest](https://github.com/weecology/DeepForest)
4. **Fetch** building footprints from OpenStreetMap
5. **Classify** each tree as safe or danger based on whether its fall radius intersects a building
6. **Annotate** the satellite image: green = safe, red = danger, cyan = building outlines

## Setup

Requires Python 3.11 (for DeepForest/PyTorch compatibility). Uses [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://python-poetry.org/).

```bash
pyenv install 3.11
pyenv local 3.11
poetry install
```

Create a `.env` file with your Google Maps API key (needs Geocoding and Static Maps APIs enabled):

```
GOOGLE_MAPS_API_KEY=your_key_here
```

## Usage

```bash
poetry run marimo edit notebook.py
```

Enter a property address and adjust the fall radius multiplier to tune sensitivity.

## Stack

- **Notebook**: [Marimo](https://marimo.io)
- **Tree detection**: [DeepForest](https://github.com/weecology/DeepForest)
- **Satellite imagery**: Google Maps Static API
- **Building footprints**: OpenStreetMap via [OSMnx](https://github.com/gboeing/osmnx)
- **Geometry**: Shapely
- **Image annotation**: Pillow
