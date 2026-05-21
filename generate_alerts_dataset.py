import os
import pandas as pd
import numpy as np

def generate_alerts_dataset():
    print("Loading data...")
    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv")
        date_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df = df.sort_index()
            
        col_map = {
            'pm25': 'PM2.5', 'pm2.5': 'PM2.5', 'pm10': 'PM10',
            'nh3': 'NH3', 'co': 'CO', 'no': 'NO', 'no2': 'NO2', 'so2': 'SO2',
            'city': 'City', 'location': 'City'
        }
        df.rename(columns={c: col_map.get(c.lower(), c) for c in df.columns}, inplace=True)
        required_cols = ['PM2.5', 'PM10', 'NH3', 'CO', 'NO', 'NO2', 'SO2']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        df = df.ffill().bfill().fillna(0)
        if 'City' not in df.columns:
            df['City'] = 'Unknown'
    else:
        # Generate Synthetic Data
        dates = pd.date_range("2024-07-01", "2024-12-31", freq="h")
        cities = ["Lahore", "Karachi", "Islamabad"]
        data = []
        for city in cities:
            base_pm25 = 100 if city == "Lahore" else 50
            pm25 = np.random.normal(base_pm25, 20, len(dates))
            for _ in range(5):
                spike_idx = np.random.randint(100, len(dates)-100)
                pm25[spike_idx:spike_idx+12] += np.random.randint(150, 300)
            for i, d in enumerate(dates):
                data.append({
                    "Datetime": d,
                    "City": city,
                    "PM2.5": pm25[i],
                    "PM10": pm25[i] * 1.5 + np.random.normal(10, 5),
                    "NH3": np.random.uniform(0, 60),
                    "CO": np.random.uniform(0, 120),
                    "NO": np.random.uniform(0, 60),
                    "NO2": np.random.uniform(0, 60),
                    "SO2": np.random.uniform(0, 60)
                })
        df = pd.DataFrame(data)
        df.set_index("Datetime", inplace=True)
        
    print("Running Stage 1: Spike Detection...")
    # Stage 1: Spike Detection
    all_city_dfs = []
    for city in df['City'].unique():
        city_df = df[df['City'] == city].copy()
        window = 24 * 7 
        city_df['Rolling_Mean'] = city_df['PM2.5'].rolling(window=window, min_periods=1).mean()
        city_df['Rolling_Std'] = city_df['PM2.5'].rolling(window=window, min_periods=1).std()
        city_df['Z_Score'] = (city_df['PM2.5'] - city_df['Rolling_Mean']) / city_df['Rolling_Std']
        city_df['Z_Score'] = city_df['Z_Score'].fillna(0)
        city_df['Is_Spike'] = city_df['Z_Score'] > 2.5
        all_city_dfs.append(city_df)
        
    final_df = pd.concat(all_city_dfs)
    
    print("Running Stage 2: Source Classification...")
    def classify_source(row):
        if not row['Is_Spike']: return "Normal"
        if row['NH3'] > 45 and row['CO'] > 80: return "Crop Burning"
        elif row['NO'] + row['NO2'] > 80: return "Vehicular Emissions"
        elif row['SO2'] > 45: return "Industrial Emissions"
        elif row['PM10'] / (row['PM2.5'] + 0.1) > 2.0: return "Dust Storm"
        else: return "Mixed Sources"

    final_df['Source'] = final_df.apply(classify_source, axis=1)

    print("Running Stage 3: Public Alert Generation...")
    def generate_alert(row):
        if not row['Is_Spike']: return None
        city, source, pm25 = row['City'], row['Source'], row['PM2.5']
        alert = f"⚠️ Alert for {city}: We have detected a sudden pollution spike (PM2.5: {pm25:.0f} µg/m³) likely caused by {source}. Children, the elderly, and respiratory patients are highly vulnerable."
        if source == "Crop Burning": alert += " Please limit outdoor activities and keep windows closed to avoid smoke."
        elif source == "Dust Storm": alert += " Wear N95 masks if you must go outside and expect reduced visibility."
        else: alert += " Please use air purifiers if available and minimize outdoor exertion."
        return alert

    final_df['Generated_Alert'] = final_df.apply(generate_alert, axis=1)
    
    # Filter spikes only and drop duplicates by Date to avoid spam
    spikes_df = final_df[final_df['Is_Spike']].copy()
    spikes_df['Date'] = spikes_df.index.date
    unique_alerts = spikes_df.drop_duplicates(subset=['City', 'Date', 'Source'])
    
    # Select top 10 best alerts
    top_alerts = unique_alerts[['City', 'PM2.5', 'Source', 'Generated_Alert']].head(10)
    
    # Save to CSV
    top_alerts.to_csv("public_alerts.csv")
    print("\n✅ Successfully generated 'public_alerts.csv' with 10 sample alerts!")
    print(top_alerts)

if __name__ == "__main__":
    generate_alerts_dataset()
