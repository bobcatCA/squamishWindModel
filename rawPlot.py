import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('swsWind.csv')
df1['datetime'] = pd.to_datetime(df1['datetime'], utc=True).dt.tz_convert('America/Vancouver')
df2 = pd.read_csv('swsWind_2025.csv')
df2['datetime'] = pd.to_datetime(df2['datetime'], utc=True).dt.tz_convert('America/Vancouver')


# Function to resample, compute cumulative speed per year, and return tidy DataFrame
def get_cumulative_by_year(df):
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['year'] = df['datetime'].dt.year
    all_years = df['year'].unique()

    result = []

    for year in all_years:
        df_year = df[df['year'] == year].copy()

        # Build full hourly datetime index for the year
        start = pd.Timestamp(f'{year}-01-01T00:00:00Z')
        end = pd.Timestamp(f'{year + 1}-01-01T00:00:00Z') - pd.Timedelta(hours=1)
        full_range = pd.date_range(start, end, freq='1H')

        # Resample and reindex to include all hours
        df_year = df_year.set_index('datetime').resample('1H').sum(numeric_only=True)
        df_year = df_year.reindex(full_range).fillna(0).reset_index()
        df_year.rename(columns={'index': 'datetime'}, inplace=True)

        df_year['year'] = year
        df_year['hours_since_start'] = (df_year['datetime'] - start).dt.total_seconds() / 3600
        df_year['cumulative_speed'] = df_year['speed'].cumsum()

        result.append(df_year[['year', 'hours_since_start', 'cumulative_speed']])

    return pd.concat(result, ignore_index=True)


# Process both DataFrames
df1_cum = get_cumulative_by_year(df1)
df2_cum = get_cumulative_by_year(df2)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))

# Plot all years from df1 and df2
for year in df1_cum['year'].unique():
    subset = df1_cum[df1_cum['year'] == year]
    ax.plot(subset['hours_since_start'], subset['cumulative_speed'], label=f'df1 {year}', alpha=0.8)

for year in df2_cum['year'].unique():
    subset = df2_cum[df2_cum['year'] == year]
    ax.plot(subset['hours_since_start'], subset['cumulative_speed'], label=f'df2 {year}', alpha=0.8, linestyle='--')

# Final touches
ax.set_title('Cumulative Hourly Speed by Year')
ax.set_xlabel('Hours Since Jan 1')
ax.set_ylabel('Cumulative Speed')
ax.legend()
plt.tight_layout()
plt.show()
