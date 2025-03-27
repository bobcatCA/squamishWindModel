import webbrowser

html_file = 'combined.html'

# Open both dataframes, generate HTML content
with open('df_forecast_hourly.html') as f1, open('df_forecast_daily.html') as f2:
    html_content = f"""
        <html>
        <head><title>DataFrames</title></head>
        <body>
            <h2>Hourly Forecast for Today</h2>
            {f1.read()}
            <h2>Daily Forecast</h2>
            {f2.read()}
        </body>
        </html>
        """

    # Save combined file
with open(html_file, "w") as f:
    f.write(html_content)

    # Open in browser and display
webbrowser.open(html_file)
