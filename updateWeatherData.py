import pandas as pd
from envCanadaForecastPull import pull_forecast
from envCanadaStationPull import pull_past_24hrs

def get_conditions_table():
    """
    :return: DataFrame, Concat'd with observed values from the past 24hrs, and forecast hourly
    """
    # Pull the past and forecast data
    df_forecast = pull_forecast()
    df_past24 = pull_past_24hrs()

    # Put the two dataframes together
    df = pd.concat([df_forecast, df_past24], ignore_index=True, sort=False)
    df.sort_values(by='datetime', inplace=True)

    return df

if __name__=='__main__':
    pass