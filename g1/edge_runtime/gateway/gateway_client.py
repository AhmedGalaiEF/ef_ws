#!/usr/bin/env python3
"""
gateway_client.py
=================

Client for communicating with the cloud marketplace platform.
Handles authentication, device registration, and API communication.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


class GatewayClient:
    """Client for communicating with the cloud marketplace platform."""

    def __init__(
        self,
        endpoint: str,
        device_id: str,
        api_key: str,
        timeout: int = 30,
    ) -> None:
        self.endpoint = endpoint
        self.device_id = device_id
        self.api_key = api_key
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize the client session."""
        self.session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
                "X-Device-ID": self.device_id,
                "X-API-Key": self.api_key,
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        )

    async def close(self) -> None:
        """Close the client session."""
        if self.session:
            await self.session.close()

    async def register_device(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register the device with the cloud platform."""
        url = f"{self.endpoint}/api/v1/devices/register"
        logger.info(f"Registering device {self.device_id} with cloud platform...")
        
        async with self.session.post(url, json=device_data) as response:
            response.raise_for_status()
            return await response.json()

    async def send_telemetry(self, telemetry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send telemetry data to the cloud platform."""
        url = f"{self.endpoint}/api/v1/telemetry"
        logger.debug(f"Sending telemetry data for device {self.device_id}...")
        
        async with self.session.post(url, json=telemetry_data) as response:
            response.raise_for_status()
            return await response.json()

    async def check_skill_compatibility(
        self, skill_id: str
    ) -> Dict[str, Any]:
        """Check if a skill is compatible with the device."""
        url = f"{self.endpoint}/api/v1/skills/{skill_id}/compatibility"
        params = {"device_id": self.device_id}
        logger.info(f"Checking compatibility for skill {skill_id}...")
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def download_skill(self, skill_id: str, version: str) -> bytes:
        """Download a skill from the cloud platform."""
        url = f"{self.endpoint}/api/v1/skills/{skill_id}/download"
        params = {"version": version}
        logger.info(f"Downloading skill {skill_id} version {version}...")
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.read()

    async def send_heartbeat(self) -> Dict[str, Any]:
        """Send a heartbeat to the cloud platform."""
        url = f"{self.endpoint}/api/v1/devices/{self.device_id}/heartbeat"
        logger.debug(f"Sending heartbeat for device {self.device_id}...")
        
        async with self.session.post(url, json={"status": "active"}) as response:
            response.raise_for_status()
            return await response.json()


async def main() -> None:
    """Example usage of the GatewayClient."""
    logging.basicConfig(level=logging.INFO)
    
    client = GatewayClient(
        endpoint="https://marketplace.example.com",
        device_id="G1_Robot_001",
        api_key="your_api_key_here",
    )
    
    await client.initialize()
    
    try:
        # Example: Register device
        device_data = {
            "manufacturer": "Unitree",
            "model": "G1",
            "os_version": "3.2.1",
        }
        await client.register_device(device_data)
        
        # Example: Send telemetry
        telemetry_data = {
            "cpu_usage": 45.2,
            "memory_usage": 60.1,
            "battery_level": 85,
        }
        await client.send_telemetry(telemetry_data)
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
