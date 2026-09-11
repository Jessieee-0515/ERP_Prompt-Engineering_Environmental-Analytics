# Data

This study uses daily air temperature (ATMP) and water temperature (WTMP) observations from
**NOAA National Data Buoy Center (NDBC) station 45004**, located on Lake Superior.

## Raw data

Raw NDBC historical data is publicly available from:

https://www.ndbc.noaa.gov/station_history.php?station=45004

Raw data files are **not redistributed in this repository**. To reproduce the processed
CSVs used in this study:

1. Download the annual historical data files for station 45004 covering 1990–2004 from the
   NDBC site above.
2. Place the downloaded files in `data/raw/`.
3. Run:
   ```bash
   python data/data_preparation.py
   ```
   This resamples the raw observations to daily means, selects the `ATMP` and `WTMP`
   columns, drops rows with missing values in either column, and writes:
   - `data/processed/45004_1990_1999.csv` (calibration period)
   - `data/processed/45004_2000_2004.csv` (validation period)

## Processed data

`data/processed/*.csv` are included in this repository. Each file has three columns:
`Datetime`, `ATMP` (°C), `WTMP` (°C).

## Note on missing values

The `dropna()` step in `data_preparation.py` removes any day with a missing `ATMP` or
`WTMP` value, which shortens the effective time index but does not insert placeholder days.
Several of this study's ODE implementations index the forcing series by integer day-position
(`Ta[int(t)]`) rather than by calendar date; if the raw record has non-trivial gaps, this can
desynchronize "day position in the array" from "actual elapsed calendar days" in an
integration. See `docs/known_issues.md` for further discussion of this risk.
