# Downloading the public PSID

> **Update, 4 November 2026.** The PSID server refuses scripted downloads (HTTP 403), so the data and docs steps don't work. Download the zips by hand from https://simba.isr.umich.edu/Zips/ZipMain.aspx into `data/raw/psid/zips/`, then run only the conversion:
>
> ```bash
> .venv/bin/python data_cleaning/psid/00_download_psid.py --steps convert
> ```
>
> This makes 173 parquet files (about 310 MB) from the 83 zips. The documentation the zips don't include is listed in `psid_documentation_to_download.md`, next to this README.

`00_download_psid.py` downloads every public PSID file and the documentation, then converts the data to parquet. You run it on your Mac from the repo root.

## Before you start

1. Register for a free account at https://psidonline.isr.umich.edu (the Register link, top right). Accept the conditions of use, then confirm the account from the email they send.
2. Check that you can log in at https://simba.isr.umich.edu/U/Login.aspx. The script uses the same username and password.
3. Make sure there's about 15 GB free on your disk. That's my rough estimate for the zips, the unzipped text and the parquet together.

## Running it

```bash
cd ~/Documents/GitHub/prevention-health-clustering
.venv/bin/python data_cleaning/psid/00_download_psid.py
```

It will ask for your PSID username, then your password. The password isn't shown as you type and isn't saved. After that it runs three steps:

1. **data**: about 85 zips from the Packaged Data page. "Sensitive Data Variables" files are skipped, because they need a separate agreement.
2. **docs**: every codebook, questionnaire, QxQ, user guide and weights note linked from the documentation page, plus the cross-year variable index.
3. **convert**: unzips each file, reads the fixed-width text using the SAS setup file inside it, and writes parquet.

Expect this to take one to a few hours, mostly in the downloads. It's safe to stop with Ctrl-C and run the same command again, because anything already finished is skipped. At the end it lists anything that failed. Run it again to retry those.

Useful variants:

```bash
# Main study only (skip the child and transition-to-adulthood supplements)
.venv/bin/python data_cleaning/psid/00_download_psid.py --supplements none

# Just one or two files, e.g. as a quick first test: 2023 family file + individual file
.venv/bin/python data_cleaning/psid/00_download_psid.py --only 1214,1053 --steps data,convert

# Only one step
.venv/bin/python data_cleaning/psid/00_download_psid.py --steps docs
```

## Where the files appear

Everything goes under `data/raw/psid/`, which is gitignored, so none of it can be committed by accident:

| Folder | Contents |
|---|---|
| `data/raw/psid/zips/` | The original zips as downloaded, named `<file id>__<PSID name>.zip`, plus `manifest.csv` (sizes, checksums, dates) |
| `data/raw/psid/extracted/<name>/` | Each zip unpacked: the fixed-width `.txt` data, SAS/SPSS/Stata setup files, and the codebook and readme files PSID ships inside the zip |
| `data/raw/psid/parquet/` | One `<NAME>.parquet` per data file, e.g. `FAM2023ER.parquet` or `IND2023ER.parquet`, and `<NAME>_labels.csv` with each variable's label and column positions |
| `data/raw/psid/docs/` | The documentation, mirroring the PSID site's folders (codebooks under `documents/psid/codebook/`, questionnaires under `documents/psid/questionnaires/`, weights under `data/weights/`), plus `index.csv` linking every local file to its link text and URL |

## If something goes wrong

- **"Login failed"**: check the login works in a browser first. If it does and the script still fails, send me `login_page.html` from the repo root, because the site may have changed.
- **"server did not return a zip"**: the HTML the server sent back is saved next to the zips as `<id>__not_a_zip.html`. This usually means the session expired. The script logs in again once by itself, and re-running it also fixes it.
- **A conversion fails**: the raw text and setup files are still in `extracted/`, so nothing is lost. Send me the error.

## After the download: the PSID panel and health measures

Run these from the repo root, in this order. Everything except the check figures is written under `data/processed/psid/` or `artifacts/`, which are gitignored.

| Step | Command | Writes |
|---|---|---|
| Panel | `.venv/bin/python data_cleaning/psid/01_build_psid_panel.py` | `data/processed/psid/psid_panel.parquet`: one row per Reference Person or Spouse per wave, 1999–2023 |
| Health measure | `.venv/bin/python data_cleaning/psid/02_build_psid_health.py` | `data/processed/psid/measures/psid_health_measure*`: θ, h and the weighted sum, the item parameters, Q3, the latent age profile, and the 2017–2023 six-testlet check |
| Frailty index | `.venv/bin/python data_cleaning/psid/03_build_psid_frailty.py` | `data/processed/psid/measures/psid_frailty_index.parquet` and its comparison with Hosseini, Kopecky and Zhao (2022) |
| Checks | `.venv/bin/python descriptives/psid/01_psid_health_checks.py` | three figures in `descriptives/psid/figures/` and `artifacts/descriptives/psid/psid_health_checks.csv` |

The checks compare PSID with the UKHLS using only aggregate files the UKHLS scripts already write (the item parameters, the latent age profile, the Exhibit 1 profiles, and the ceiling and mortality tables). None of these scripts reads UKHLS microdata or writes outside the PSID folders. `PSID_CROSSWALK.md` documents the variables and the choices behind them.
