# EuroLeague Strength of Schedule Dashboard

A static web dashboard for analyzing **Strength of Schedule (SoS)** in the EuroLeague using both **efficiency-based** (Net Rating) and **results-based** (Win%) approaches, with additional **forward-looking difficulty analysis** for upcoming opponents.

Charts are precomputed offline and published as JSON, so the site itself is plain static files — no backend, no database, and no API calls from the browser.

---

## What this project does

Team records alone do not fully capture how difficult a team’s schedule has been or will be.  
This project provides context by answering:

- How difficult has each team’s schedule been?
- How difficult is each team’s upcoming schedule?
- How should team performance be interpreted given opponent strength?

The dashboard visualizes season-long SoS, SoS vs Net Rating relationships, and upcoming opponent difficulty.

---

## Data sources

- **Official EuroLeague schedule**  
  Fetched live from the EuroLeague schedule endpoint via `sos.data.load_schedule_from_api`.
  Earlier versions parsed the Regular Season PDF into a hand-maintained
  `EL_2025_26_EL_RS_Schedule.csv`; that file is gone. Because the schedule is now read
  from the API on every run, postponed games and home/away swaps are picked up
  automatically instead of needing a manual re-export each time the calendar changes.

- **Team statistics**  
Fetched using the excellent open-source euroleague_api package by Giannis Giasemidis (github.com/giasemidis/euroleague_api), which made pulling this data far easier.

This API supplies the scheduled game statistics required to compute my advanced metrics.

---

## Core metrics

**Net Rating**


- OffRtg: points scored per 100 possessions  
- DefRtg: points allowed per 100 possessions  

Higher NetRtg indicates a stronger team.

---

## Strength of Schedule methodology

The project adapts the **Hack-a-Stat** Strength of Schedule framework and extends it using **Net Rating**.

### Win%-based SoS


- OW%: opponents’ winning percentage  
- OOW%: opponents’ opponents’ winning percentage  

### Net Rating–based SoS


- OppNetRtg: average Net Rating of opponents  
- OONetRtg: opponents’ opponents Net Rating  

The NetRtg-based approach is more stable and less sensitive to close-game variance.

---

## Four Factors

Alongside the schedule work, the dashboard carries a **Four Factors** team profile — Dean
Oliver's four levers that decide games, computed for each team and for what it allows:

- **eFG%** = (FGM + 0.5 × 3PM) / FGA — shooting, with threes credited for the extra point
- **TOV%** = TOV / (FGA + 0.44 × FTA + TOV) — share of possessions given away
- **OR%** = OREB / (OREB + Opponent DREB) — share of own misses recovered
- **FTR** = FTM / FGA — free-throw scoring generated per shot

The **DEFENSE** half is the same four formulas applied to the opponents' totals; the defensive
rebounding column is the mirror of the opponents' OR%. Ratios are built from summed season
totals rather than averaged per game, so high-volume games count for more.

The eight columns collapse into a single **4F Rating** using Oliver's weights (40% shooting,
25% turnovers, 20% rebounding, 15% free throws). The factors are in incompatible units, so each
is standardized across the league at that round, signed so positive is always good, then
weighted. The rating is therefore relative by construction — comparable between teams within a
round, but not across rounds.

This is a **standalone profile**: it feeds none of the SoS numbers, and the SoS views are
unchanged by it. Both sides of every game are already in the box scores the ratings read, so it
needs no extra data source.

---

## Forward-looking difficulty (Next-N)

Upcoming schedule difficulty is computed using the next **N** scheduled opponents:


Opponent logos and color-coded cells provide a quick visual summary of future difficulty.

---

## Visualizations

- **Season SoS Table** — NetRtg vs Win% comparison  
- **SoS(Net) vs NetRtg Scatter** — contextual quadrants and side table  
- **Next-N Games Table** — upcoming opponent difficulty with logos  
- **Four Factors Table** — per-team offensive and defensive Four Factors with the weighted 4F Rating  

---

## Seasons

The site is not tied to one season. Every chart is published under
`frontend/data/seasons/<season>/rounds/<round>/`, and `frontend/data/latest.json`
carries the manifest the page boots from:

```json
{"season": 2025, "seasons": [{"id": 2025, "label": "2025-26", "latest_round": 38}], "round": 38}
```

That manifest is derived by scanning what is actually on disk, so it can never offer
a season whose JSON is missing. The sidebar's **Season** selector is driven entirely
by it — with one season published the arrows are inert, and a second one appears in
the picker with no frontend change. Switching season retargets the round stepper to
that season's own last round.

Adding a season is therefore a data operation: point `DEFAULT_SEASON` at it and run
the pipeline. Team logos stay at the shared `frontend/data/logos/`, since clubs
outlive seasons.

---

## To do : <br />
    - Upload the code for fetching the Euroleague team's logos.