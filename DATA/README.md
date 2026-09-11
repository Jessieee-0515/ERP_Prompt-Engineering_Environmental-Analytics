# Data

This study uses daily air temperature (ATMP) and water temperature (WTMP) observations from
**NOAA National Data Buoy Center (NDBC) station 45004**, located on Lake Superior.

## Raw data

Raw NDBC historical data is publicly available from:

https://www.ndbc.noaa.gov/station_history.php?station=45004

Raw data files are **not redistributed in this repository**. To reproduce the processed
CSVs used in this study:

In the data preprocessing script, the raw data are downloaded directly from the above website using web scraping and processed within the same workflow. Therefore, the time period for reproduction and analysis can be modified simply by changing the year settings in the code, allowing the corresponding dataset to be retrieved automatically.

Run:
   ```bash
   python DATA/data_preparation.py
   ```
   This resamples the raw observations to daily means, selects the `ATMP` and `WTMP`
   columns, drops rows with missing values in either column, and writes:
   - `data/processed/45004_1990_1999.csv` (calibration period)
   - `data/processed/45004_2000_2004.csv` (validation period)


## Note on missing values
Missing values were handled by removing observations with missing air temperature (ATMP) or water temperature (WTMP) using dropna() in data_preparation.py. No interpolation or gap-filling was applied, so the processed dataset contains only valid paired observations.
