import marimo

app = marimo.App(width="medium")


@app.cell
def imports():
    """Load pipeline modules and dependencies."""
    from pipeline.geocode import geocode
    from pipeline.imagery import fetch_satellite_tile
    from pipeline.detect import detect_trees
    from deepforest.visualize import plot_results
    from PIL import ImageDraw
    import marimo as mo
    return ImageDraw, detect_trees, fetch_satellite_tile, geocode, mo, plot_results


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
def annotate_and_display(ImageDraw, detections, mo, tile):
    """Draw bounding boxes around detected trees and display the result."""
    annotated = tile.copy()
    draw = ImageDraw.Draw(annotated)
    for _, row in detections.iterrows():
        draw.rectangle(
            [row["xmin"], row["ymin"], row["xmax"], row["ymax"]],
            outline="lime",
            width=2,
        )
    mo.output.replace(mo.image(annotated))
    return (annotated,)


if __name__ == "__main__":
    app.run()
