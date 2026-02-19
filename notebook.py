import marimo

app = marimo.App(width="medium")


@app.cell
def imports():
    """Load pipeline modules and dependencies."""
    from pipeline.geocode import geocode
    from pipeline.imagery import fetch_satellite_tile
    from pipeline.detect import detect_trees
    from pipeline.risk import compute_tree_risks, classify_danger
    from pipeline.buildings import fetch_buildings
    from pipeline.imagery import tile_bounds
    from deepforest.visualize import plot_results
    from PIL import ImageDraw
    import marimo as mo
    return (ImageDraw, classify_danger, compute_tree_risks, detect_trees,
            fetch_buildings, fetch_satellite_tile, geocode, mo, plot_results, tile_bounds)


@app.cell
def geocode_address(geocode, mo):
    """Convert a street address to lat/lng coordinates via Google Geocoding API."""
    address = "200 Colma Blvd, Colma, CA 94014"
    lat, lng = geocode(address)
    mo.output.replace(mo.md(f"**Address:** {address}\n\n**Coordinates:** {lat:.6f}, {lng:.6f}"))
    return address, lat, lng


@app.cell
def fetch_satellite_imagery(fetch_satellite_tile, lat, lng, mo):
    """Fetch a 640x640 satellite tile centered on the geocoded coordinates."""
    tile = fetch_satellite_tile(lat, lng)
    mo.output.replace(mo.vstack([
        mo.md(f"**Tile size:** {tile.size[0]}x{tile.size[1]}"),
        mo.image(tile),
    ]))
    return (tile,)


@app.cell
def detect_tree_crowns(detect_trees, mo, plot_results, tile):
    """Run DeepForest tree crown detection and show debug overlay."""
    detections = detect_trees(tile)
    fig = plot_results(detections, image=tile, show=False)
    mo.output.replace(mo.vstack([
        mo.md(f"**Trees detected:** {len(detections)}"),
        fig,
    ]))
    return (detections,)


@app.cell
def calibrate_trees(compute_tree_risks, detections, lat, lng, tile):
    """Convert pixel detections to real-world coordinates and compute fall radii."""
    trees = compute_tree_risks(
        detections, center_lat=lat, center_lng=lng,
        img_width=tile.size[0], img_height=tile.size[1],
    )
    return (trees,)


@app.cell
def fetch_building_footprints(fetch_buildings, lat, lng, mo, tile_bounds):
    """Fetch OSM building footprints for the tile area."""
    bounds = tile_bounds(lat, lng)
    buildings = fetch_buildings(**bounds)
    mo.output.replace(mo.md(f"**Buildings found:** {len(buildings)}"))
    return bounds, buildings


@app.cell
def classify_tree_risk(buildings, classify_danger, trees):
    """Classify each tree as danger/safe based on fall radius vs building proximity."""
    classified = classify_danger(trees, buildings)
    return (classified,)


@app.cell
def annotate_and_display(ImageDraw, bounds, buildings, classified, lat, lng, mo, tile):
    """Draw buildings, safe trees (green), and danger trees (red) on the satellite image."""
    from pipeline.risk import pixel_to_latlng
    from pipeline.imagery import meters_per_pixel, DEFAULT_ZOOM
    import math

    annotated = tile.copy()
    draw = ImageDraw.Draw(annotated)
    img_w, img_h = tile.size
    mpp = meters_per_pixel(lat)

    # Helper: convert lat/lng to pixel coordinates
    def latlng_to_pixel(pt_lat, pt_lng):
        deg_per_m_lat = 1 / 111_320
        deg_per_m_lng = 1 / (111_320 * math.cos(math.radians(lat)))
        dy_m = (lat - pt_lat) / deg_per_m_lat
        dx_m = (pt_lng - lng) / deg_per_m_lng
        px_x = dx_m / mpp + img_w / 2
        px_y = dy_m / mpp + img_h / 2
        return px_x, px_y

    # Draw building footprints in blue
    for _, bldg in buildings.iterrows():
        geom = bldg.geometry
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue
        for poly in polys:
            coords = list(poly.exterior.coords)
            px_coords = [latlng_to_pixel(c[1], c[0]) for c in coords]
            draw.polygon(px_coords, outline="cyan", width=2)

    # Draw trees: green = safe, red = danger
    for _, tree in classified.iterrows():
        color = "red" if tree["danger"] else "lime"
        # Draw bounding box
        draw.rectangle(
            [tree["xmin"], tree["ymin"], tree["xmax"], tree["ymax"]],
            outline=color, width=2,
        )
        # Draw fall radius circle for danger trees
        if tree["danger"]:
            cx = (tree["xmin"] + tree["xmax"]) / 2
            cy = (tree["ymin"] + tree["ymax"]) / 2
            fall_radius_px = tree["fall_radius_m"] / mpp
            draw.ellipse(
                [cx - fall_radius_px, cy - fall_radius_px,
                 cx + fall_radius_px, cy + fall_radius_px],
                outline="red", width=1,
            )

    n_danger = classified["danger"].sum()
    mo.output.replace(mo.vstack([
        mo.md(f"**{len(classified)} trees detected, {n_danger} within fall distance of structures**"),
        mo.image(annotated),
    ]))
    return (annotated,)


if __name__ == "__main__":
    app.run()
