# Park Meetup Selector

A small [Streamlit](https://streamlit.io) tool for choosing the best park for a group
meetup in **Bristol**, using **multi-criteria decision analysis (MCDA)**. Instead of
computing one "winner", it keeps the raw data transparent and lets each user weight the
criteria they care about (park quality, travel time by walk / drive / transit / drive
while avoiding the Clean Air Zone, and parking). Drivers, CAZ-avoiders, public-transport
users and families each get a different ranking from the same data.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).
On Windows you can instead just double-click `run.bat`.

Run the tests with:

```bash
pip install -r requirements-dev.txt
pytest
```

## Deploy to Streamlit Community Cloud (free)

This app is deploy-ready — it reads only the CSVs in `data/`, with no database, API
keys, or secrets.

1. **Push this repo to GitHub** (once):
   ```bash
   git add -A
   git commit -m "Prepare for deployment"
   # create an empty repo on github.com first, then:
   git remote add origin https://github.com/<you>/park-selector.git
   git branch -M main
   git push -u origin main
   ```
   The repo can be **private** — Community Cloud will ask for access.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and click **Create app →
   Deploy a public app from GitHub** (private repos work too).
3. Set:
   - **Repository**: your `park-selector` repo
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **Advanced settings → Python version**: `3.13` (matches `.python-version`)
4. Click **Deploy**. First build takes a couple of minutes; you'll get a
   `https://<name>.streamlit.app` URL.

Notes for low-traffic use:
- The app **sleeps after a few days idle** and wakes on the next visit (a short spin-up).
  That's expected on the free tier.
- Dependencies come from `requirements.txt` (pinned with upper bounds so an unattended
  app won't break on a future major release).
- If your GitHub repo is public, the `data/` CSVs are public too. Keep the repo private
  if that matters. You can also restrict who can open the app under the app's
  **Settings → Sharing**.
- To update the live app, just `git push` — Community Cloud redeploys automatically.

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
   (`walk`, `drive`, `transit`, `drive_no_caz`).

### Subjective quality sub-scores (1–5)
`q_scenery`, `q_space`, `q_facilities`, `q_tree_cover`, `q_flatness` — your own judgement.
These drive the radar chart and roll up (via the sidebar sub-weights) into the overall
quality criterion.

### Parking & CAZ
- `parking_type`: `free` / `paid` / `street` / `none`
- `parking_cost_per_day`: £ (0 if free)
- `parking_spaces`: `plenty` / `limited` / `scarce`
- `in_caz`: `true` if the park sits inside Bristol's Clean Air Zone.

### Travel modes, and the `drive_no_caz` metric
Each origin × park has four modes in `travel_times.csv`:
- `walk`, `drive`, `transit` — must be filled for **every** park.
- `drive_no_caz` — driving while **avoiding** the Clean Air Zone. For parks inside the
  CAZ this is **N/A**: leave `duration_min` blank (those parks are unreachable without
  paying to enter). For parks outside the CAZ, enter the door-to-door time on a route
  that skirts the zone. This lets people who won't enter the CAZ rank parks separately
  from people who will (who use plain `drive`).

The loader enforces this: `drive_no_caz` must be blank for CAZ parks and filled for
non-CAZ parks.

## How the score is built

For each park, six criteria are normalised to 0–1 (1 = best): `quality`, `walk_time`,
`drive_time`, `drive_no_caz`, `transit_time`, `parking`. Lower travel times / parking
friction score higher; an N/A (`drive_no_caz` for a CAZ park) scores 0 (worst). The
composite score is the weighted sum of these, using the sidebar weights (normalised to
sum to 1). See [`src/scoring.py`](src/scoring.py).

## Roadmap (not in this first pass)

- Auto-populate travel times via a routing API (Google Maps / OpenRouteService /
  TravelTime) — same CSV schema.
- Travel-time isochrones on the map.
- User-entered custom origin address at runtime.
- Swap CSV storage for SQLite/Parquet behind `src/loading.py`.
