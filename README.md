# redback-data-warehouse
Data Warehouse storage of code and configurations

## Garmin Run Data – ETL Pipeline Update

This ETL pipeline processes `Garmin_run_data.csv` and includes:

### Data cleaning:
- Removes duplicate rows
- Standardizes column names (lowercase, underscores)
- Converts timestamps to datetime
- Fills missing numeric values with column means
- Removes outliers in `heart_rate` (keeps values between 30–220 bpm)
- Converts distance from meters to kilometers
- Converts speed from m/s to km/h

### Data aggregation:
- Groups data by year and week
- Calculates total runs, total distance (km), average speed (km/h), and average pace (min/km) per week

### Outputs:
- `cleaned_garmin_run_data.csv` → cleaned dataset

