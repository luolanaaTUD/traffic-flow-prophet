from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass
class QWeatherClient:
    api_key: str
    location: str
    base_url: str = "https://devapi.qweather.com/v7/weather/7d"
    timeout_seconds: int = 10

    def fetch_daily_forecast(self) -> list[dict]:
        if not self.api_key:
            raise ValueError("Missing QWeather API key.")
        if not self.location:
            raise ValueError("Missing QWeather location.")

        query = urlencode({"location": self.location, "key": self.api_key})
        request_url = f"{self.base_url}?{query}"
        try:
            with urlopen(request_url, timeout=self.timeout_seconds) as response:
                raw = response.read()
                encoding = (response.headers.get("Content-Encoding") or "").lower()
                if encoding == "gzip" or raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                payload = json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"QWeather HTTP error: {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"QWeather connection error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("QWeather request timed out.") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("QWeather returned invalid JSON.") from exc

        if payload.get("code") != "200":
            raise RuntimeError(f"QWeather API error code: {payload.get('code', 'unknown')}")

        rows = payload.get("daily") or []
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("QWeather response missing daily forecast rows.")
        return rows
