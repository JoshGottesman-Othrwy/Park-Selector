# Park Meetup Selector

A small [Streamlit](https://streamlit.io) tool for choosing the best park for a group
meetup in **Bristol**, using **multi-criteria decision analysis (MCDA)**. Instead of
computing one "winner", it keeps the raw data transparent and lets each user weight the
criteria they care about (park quality, travel time by walk/drive/transit, parking, and
Clean Air Zone impact). Drivers, public-transport users, families and climbers each get
a different ranking from the same data.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

Run the tests with:

```bash
pytest
```

## The three views

- **Map** — park locations coloured by their live composite score, plus your origin
  suburbs. Includes a "current top pick" callout.
- **Compare** — a travel-time heatmap (origin × park, per mode), a radar chart of park
  quality dimensions, and a full sortable table of every metric.
- **Rank** — a ranked bar chart and per-park score breakdown that update in real time as
  you move the weight sliders.

## Editing the data

All data lives in editable CSVs under [`data/`](data/). Add or change rows and the app
picks them up on reload.

| File | What it holds |
|---|---|
| `data/parks.csv` | One row per park: location, subjective quality sub-scores (1–5), parking + CAZ info |
| `data/origins.csv` | Fixed suburban origin points (name, lat/lon) and a relative `weight` |
| `data/travel_times.csv` | One row per origin × park × mode: door-to-door `duration_min` |
| `data/weights_default.csv` | Default slider values for the ranking |

### Adding a park
1. Add a row to `data/parks.csv` with a unique `park_id`.
2. Add travel-time rows to `data/travel_times.csv` for every origin × mode
   (`walk`, `drive`, `transit`).

### Subjective quality sub-scores (1–5)
`q_scenery`, `q_space`, `q_facilities`, `q_climbing`, `q_family` — your own judgement.
These drive the radar chart and roll up (via the sidebar sub-weights) into the overall
quality criterion.

### Parking & CAZ
- `parking_type`: `free` / `paid` / `street` / `none`
- `parking_cost_per_day`: £ (0 if free)
- `parking_spaces`: `plenty` / `limited` / `scarce`
- `in_caz`: `true` if the park sits inside Bristol's Clean Air Zone. The CAZ charge
  (~£9/day for a non-compliant car) only penalises the **drive** mode.

## How the score is built

For each park, six criteria are normalised to 0–1 (1 = best): `quality`, `walk_time`,
`drive_time`, `transit_time`, `parking`, `caz`. The composite score is the weighted sum
of these, using the sidebar weights (normalised to sum to 1). See
[`src/scoring.py`](src/scoring.py).

## Roadmap (not in this first pass)

- Auto-populate travel times via a routing API (Google Maps / OpenRouteService /
  TravelTime) — same CSV schema.
- Travel-time isochrones on the map.
- User-entered custom origin address at runtime.
- Swap CSV storage for SQLite/Parquet behind `src/loading.py`.
