import csv
import math

def calculate_mean(data):
    return sum(data) / len(data) if data else 0

def calculate_std(data, mean):
    if len(data) < 2: return 0
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def main():
    print("Reading dataset.csv (Zero dependencies!)...")
    try:
        with open('dataset.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print("dataset.csv not found.")
        return

    pm25_window = []
    alerts = []
    
    print("Processing rows and running Spike Detection...")
    for i, row in enumerate(rows):
        try:
            dt = row.get('datetime', '')
            pm25 = float(row.get('components_pm2_5', 0))
            pm10 = float(row.get('components_pm10', 0))
            nh3 = float(row.get('components_nh3', 0))
            co = float(row.get('components_co', 0))
            no = float(row.get('components_no', 0))
            no2 = float(row.get('components_no2', 0))
            so2 = float(row.get('components_so2', 0))
        except ValueError:
            continue
            
        pm25_window.append(pm25)
        if len(pm25_window) > 168:
            pm25_window.pop(0)
            
        if len(pm25_window) == 168:
            mean = calculate_mean(pm25_window)
            std = calculate_std(pm25_window, mean)
            if std > 0:
                z_score = (pm25 - mean) / std
                if z_score > 3.0: 
                    # Classify
                    source = "Mixed Sources"
                    if nh3 > 40 and co > 80: source = "Crop Burning"
                    elif no + no2 > 80: source = "Vehicular Emissions"
                    elif so2 > 40: source = "Industrial Emissions"
                    elif pm10 / (pm25 + 0.1) > 2.0: source = "Dust Storm"
                    
                    alert = f"⚠️ Alert for Lahore: We have detected a sudden pollution spike (PM2.5: {pm25:.0f} µg/m³) likely caused by {source}. Children, the elderly, and respiratory patients are highly vulnerable."
                    if source == "Crop Burning": alert += " Please limit outdoor activities and keep windows closed."
                    elif source == "Dust Storm": alert += " Wear N95 masks if you must go outside."
                    else: alert += " Please use air purifiers if available and minimize outdoor exertion."
                    
                    date_str = dt.split(' ')[0]
                    if not any(a['Date'] == date_str for a in alerts):
                        alerts.append({
                            'Date': date_str,
                            'City': 'Lahore',
                            'PM2.5': pm25,
                            'Source': source,
                            'Generated_Alert': alert
                        })
                        if len(alerts) >= 10:
                            break

    # Write CSV
    with open('public_alerts.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Date', 'City', 'PM2.5', 'Source', 'Generated_Alert'])
        writer.writeheader()
        writer.writerows(alerts)
    print("✅ public_alerts.csv successfully generated!")

if __name__ == "__main__":
    main()
