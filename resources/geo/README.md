# Geospatial resources

These files support the Dubai grid/time modeling dataset.

## Files

- `dubai.geojson`
  - Source: https://github.com/sbma44/uber-cities/blob/master/geojson/dubai.geojson
  - Local use: main Dubai study boundary for generating the H3 grid universe.
  - Geometry observed locally: one `FeatureCollection` with a `MultiPolygon`.

- `united_arab_emirates.geojson`
  - Source: https://github.com/glynnbird/countriesgeojson/blob/master/united%20arab%20emirates.geojson
  - Local use: broad country-level QA boundary, not the main modeling boundary.
  - Geometry observed locally: one `Feature` with a coarse `Polygon`.

## Modeling rule

Use the Dubai polygon for the main grid universe. Retain observed valid incident H3 cells outside the Dubai polygon as `peripheral_observed` so boundary-edge records are not dropped too early. Use the UAE polygon only to flag records that may be outside the country.
