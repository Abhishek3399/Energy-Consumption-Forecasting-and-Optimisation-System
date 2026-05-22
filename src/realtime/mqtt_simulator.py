from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

from ..config import settings


def start_mqtt_simulation(
    df: pd.DataFrame,
    topic: str = "energy/meter",
    interval_sec: float = 1.0,
    on_tick: Optional[Callable[[dict], None]] = None,
) -> threading.Thread:
    """
    Push a stream of synthetic smart-meter readings over MQTT.
    Each message contains timestamp, building_id, energy_kwh, temperature, occupancy.
    """

    client = mqtt.Client()
    client.connect(settings.mqtt_broker_host, settings.mqtt_broker_port, 60)

    df_sorted = df.sort_values(["timestamp", "building_id"]).reset_index(drop=True)

    def _loop():
        for _, row in df_sorted.iterrows():
            payload = {
                "building_id": int(row["building_id"]),
                "timestamp": row["timestamp"].isoformat(),
                "energy_kwh": float(row["energy_kwh"]),
                "temperature_c": float(row["temperature_c"]),
                "occupancy": float(row["occupancy"]),
                "price_per_kwh": float(row["price_per_kwh"]),
            }
            msg = json.dumps(payload)
            client.publish(topic, msg)
            if on_tick:
                on_tick(payload)
            time.sleep(interval_sec)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t

