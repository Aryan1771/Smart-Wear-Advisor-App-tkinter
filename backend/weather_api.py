import os

import requests

OWM_API_KEY = os.getenv("OWM_API_KEY", "").strip()


def _fallback(query):
    return {
        "city": str(query or "Unknown"),
        "temp": 25,
        "humidity": 50,
        "condition": "clear",
        "description": "Clear sky",
    }


def get_weather(query):
    fallback = _fallback(query)
    if not OWM_API_KEY:
        return fallback

    params = {"appid": OWM_API_KEY, "units": "metric"}
    query_value = str(query or "").strip()
    if "," in query_value and any(char.isdigit() for char in query_value):
        lat, lon = query_value.split(",", 1)
        params["lat"] = lat.strip()
        params["lon"] = lon.strip()
    else:
        params["q"] = query_value or "Delhi"

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params=params,
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()

        return {
            "city": payload.get("name", fallback["city"]),
            "temp": int(round(payload.get("main", {}).get("temp", fallback["temp"]))),
            "humidity": int(payload.get("main", {}).get("humidity", fallback["humidity"])),
            "condition": payload.get("weather", [{}])[0].get("main", fallback["condition"]).lower(),
            "description": payload.get("weather", [{}])[0].get("description", fallback["description"]).title(),
        }
    except Exception as error:
        print(f"[Weather] API error: {error}")
        return fallback
