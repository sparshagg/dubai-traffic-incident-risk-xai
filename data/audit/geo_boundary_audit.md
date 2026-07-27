# Geo boundary audit

- Dubai GeoJSON: `resources/geo/dubai.geojson`
- Dubai source URL: `https://github.com/sbma44/uber-cities/blob/master/geojson/dubai.geojson`
- Dubai feature count: `1`
- Dubai geometry types: `MultiPolygon`
- Dubai coordinate count: `3706`
- Dubai bounds lon/lat: `55.07699005714011`, `24.954345826097494`, `55.53257169753073`, `25.356549000000143`
- Dubai H3 resolution 8 cells: `1182`

- UAE GeoJSON: `resources/geo/united_arab_emirates.geojson`
- UAE source URL: `https://github.com/glynnbird/countriesgeojson/blob/master/united%20arab%20emirates.geojson`
- UAE feature count: `1`
- UAE geometry types: `Polygon`
- UAE coordinate count: `22`
- UAE bounds lon/lat: `51.579519`, `22.496948`, `56.396847`, `26.055464`

## Cell universe

- Observed H3 cells: `3549`
- Study universe H3 cells: `3584`
- Universe inside Dubai cells: `1305`
- Universe peripheral observed cells: `2107`
- Universe outside UAE flagged cells: `172`

## Incident point scope

- Incident points inside Dubai: `608571`
- Incident points peripheral observed: `105706`
- Incident points outside UAE flagged: `3338`
