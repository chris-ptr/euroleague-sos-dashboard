# Setup: Supabase + GitHub Actions + Vercel

This replaces the manual "recompute locally, commit a parquet file, bump a round number" workflow
with a self-updating pipeline:

- **GitHub Actions** (`.github/workflows/refresh.yml`) runs on a schedule (every 6h) and computes
  any newly-finished round using the existing `sos/` logic, unchanged.
- **Supabase Storage** holds two buckets: `sos-cache` (private — the parquet working cache, replaces
  committing `cache/rounds/*.parquet` to git) and `sos-public` (public read — precomputed Vega-Lite
  JSON per round, fetched directly by the frontend).
- **Vercel** hosts `frontend/` as a static site (plain HTML/CSS/JS + vega-embed via CDN, no backend,
  no build step).

`app.py` (Streamlit) still works as a local fallback in the meantime — see step 7.

## 1. Create the Supabase project and buckets

1. Sign up / log in at [supabase.com](https://supabase.com) and create a new project (free tier).
2. In the project, go to **Storage** and create two buckets:
   - `sos-cache` — **private**
   - `sos-public` — **public**
3. Go to **Project Settings → API** and copy:
   - **Project URL** (e.g. `https://abcdefgh.supabase.co`)
   - **`service_role` key** (not the `anon` key — this one bypasses RLS and must never be exposed
     to the frontend or committed to git)

## 2. Local credentials

Create a `.env` file in the repo root (already covered by `.gitignore`, will never be committed):

```
SUPABASE_URL=https://abcdefgh.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
```

Install the updated dependencies (adds `pyarrow`, `requests`, `python-dotenv` to the existing list):

```bash
pip install -r requirements.txt
```

## 3. One-time backfill

Uploads the ~37 already-computed local `cache/rounds/*.parquet` files into Supabase — no
recomputation, no extra API calls:

```bash
python scripts/seed_supabase.py
```

Expected output: `Found N locally-cached round(s): [...]` followed by one line per round, ending
with `Done. Seeded rounds 1..N.`

Spot-check it landed:

```bash
curl "$SUPABASE_URL/storage/v1/object/public/sos-public/latest.json"
```

## 4. Try the recurring refresh job locally

This is what the GitHub Actions workflow runs every 6 hours. Safe to run manually — it only
computes rounds that aren't already cached:

```bash
python scripts/refresh_and_publish.py
```

If no new round has finished since the backfill, it should print `No new rounds — latest.json
timestamp refreshed, nothing else to publish.`

## 5. Push to GitHub and wire up the scheduled job

1. Push this repo to a **private** GitHub repository (public is not required).
2. In the repo, go to **Settings → Secrets and variables → Actions** and add two repository secrets:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
3. Go to the **Actions** tab, select **Refresh EuroLeague SoS data**, and click **Run workflow** to
   trigger it manually once — confirm it completes without errors.

From here on it runs unattended on its cron schedule; no manual steps per round.

## 6. Point the frontend at your Supabase project

Edit `frontend/app.js`:

```js
const CONFIG = {
  supabaseUrl: "https://abcdefgh.supabase.co",  // <-- your project URL from step 1
  publicBucket: "sos-public",
};
```

This is **not a secret** — it's just the base URL for reading the public bucket.

## 7. Deploy the frontend to Vercel

1. Sign up / log in at [vercel.com](https://vercel.com) and import the GitHub repo (private repos
   are supported on the free Hobby tier).
2. Set **Root Directory** to `frontend`.
3. No build command / output directory needed — it's a static site.
4. Deploy. Every push to the repo auto-redeploys.

Confirm the live site matches what you saw locally: nav across the 4 tabs, round selector bounded
by the real latest round, charts rendering with logos.

## 8. Retire the old workflow

Once the live site has been running for a few days and you've seen `latest.json`'s round bump on
its own after a real EuroLeague round completes:

```bash
git rm --cached cache/rounds/*.parquet
```

Then delete `app.py` (or move it to a `legacy/` folder) and remove `streamlit` from
`requirements.txt` — the manual Streamlit fallback is no longer needed.
