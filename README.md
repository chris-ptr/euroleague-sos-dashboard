# EuroLeague Strength of Schedule Dashboard

A Streamlit dashboard for analyzing **Strength of Schedule (SoS)** in the EuroLeague using both **efficiency-based** (Net Rating) and **results-based** (Win%) approaches, with additional **forward-looking difficulty analysis** for upcoming opponents.

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
  Parsed from the official EuroLeague Regular Season PDF and exported to:


- **Team statistics**  
Fetched with the open-source project:
https://github.com/giasemidis/euroleague_api

This provides OffRtg, DefRtg, NetRtg, standings, and game results.

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

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To do : <br />
        Upload the code for fetching the Euroleague team's logos and the code for the euroleague program parsing. <br />
        Make "ui" changes in steamlit dashboard <br />
        Make efficient caching, store on the loaded round computations at the loading of the page. <br />