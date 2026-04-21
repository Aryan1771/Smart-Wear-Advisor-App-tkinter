import os
import requests

def get_weather(query):
    # Use the actual key string directly if not using environment variables
    api_key = ""
    
    # Default fallback data
    fallback = {
        "city": query,
        "temp": 25,
        "humidity": 50,
        "condition": "clear"
    }

    try:
        # Determine if query is coordinates (lat,lon) or a city name
        params = {"appid": api_key, "units": "metric"}
        
        if "," in str(query) and any(char.isdigit() for char in str(query)):
            lat, lon = query.split(",")
            params["lat"] = lat.strip()
            params["lon"] = lon.strip()
        else:
            params["q"] = query

        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params=params,
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()

        # payload.get("name") converts the coordinates back into the actual city name
        return {
            "city": payload.get("name", query),
            "temp": int(payload.get("main", {}).get("temp", 25)),
            "humidity": payload.get("main", {}).get("humidity", 50),
            "condition": payload.get("weather", [{}])[0].get("main", "clear").lower(),
        }
    except Exception as e:
        print(f"Weather API Error: {e}")
        return fallback
