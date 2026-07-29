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

## Forward-looking difficulty (Next-N)

Upcoming schedule difficulty is computed using the next **N** scheduled opponents:


Opponent logos and color-coded cells provide a quick visual summary of future difficulty.

---

## Visualizations

- **Season SoS Table** — NetRtg vs Win% comparison  
- **SoS(Net) vs NetRtg Scatter** — contextual quadrants and side table  
- **Next-N Games Table** — upcoming opponent difficulty with logos  

---

## To do : <br />
    - Upload the code for fetching the Euroleague team's logos.