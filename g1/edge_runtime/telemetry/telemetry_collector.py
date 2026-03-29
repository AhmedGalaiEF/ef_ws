#!/usr/bin/env python3
"""
telemetry_collector.py
======================

Collects and reports telemetry data from the robot.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

import psutil

logger = logging.getLogger(__name__)


class TelemetryCollector:
    """Collects and reports telemetry data from the robot."""

    def __init__(self, interval: int = 60) -> None:
        self.interval = interval
        self.running = False

    async def start(self) -> None:
        """Start the telemetry collection loop."""
        self.running = True
        logger.info("Starting telemetry collection...")
        
        while self.running:
            telemetry_data = self.collect_telemetry()
            logger.debug(f"Collected telemetry: {telemetry_data}")
            
            # Here you would typically send the data to the gateway
            # await gateway_client.send_telemetry(telemetry_data)
            
            await asyncio.sleep(self.interval)

    def collect_telemetry(self) -> Dict[str, Any]:
        """Collect telemetry data from the system."""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        # Disk usage
        disk_usage = psutil.disk_usage("/").percent
        
        # Network stats
        net_io = psutil.net_io_counters()
        
        # Battery status (placeholder - would be robot-specific)
        battery_level = 85  # Placeholder
        
        return {
            "timestamp": int(time.time()),
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "network_bytes_sent": net_io.bytes_sent,
            "network_bytes_recv": net_io.bytes_recv,
            "battery_level": battery_level,
        }

    async def stop(self) -> None:
        """Stop the telemetry collection loop."""
        self.running = False
        logger.info("Stopped telemetry collection")


async def main() -> None:
    """Example usage of the TelemetryCollector."""
    logging.basicConfig(level=logging.INFO)
    
    collector = TelemetryCollector(interval=10)
    
    try:
        await collector.start()
    except asyncio.CancelledError:
        logger.info("Telemetry collection cancelled")
    finally:
        await collector.stop()


if __name__ == "__main__":
    asyncio.run(main())
