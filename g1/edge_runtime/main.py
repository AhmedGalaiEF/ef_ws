#!/usr/bin/env python3
"""
main.py
=======

Main entry point for the G1 Edge Runtime.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

from gateway.gateway_client import GatewayClient
from skills.skill_manager import SkillManager
from telemetry.telemetry_collector import TelemetryCollector

logger = logging.getLogger(__name__)


class EdgeRuntime:
    """Main edge runtime class."""

    def __init__(self) -> None:
        self.shutdown_event = asyncio.Event()
        self.gateway_client = None
        self.skill_manager = None
        self.telemetry_collector = None

    async def initialize(self) -> None:
        """Initialize the edge runtime."""
        logger.info("Initializing G1 Edge Runtime...")
        
        # Load configuration
        config_path = Path(__file__).parent / "config" / "runtime_config.yaml"
        # TODO: Load and parse config
        
        # Initialize components
        self.gateway_client = GatewayClient(
            endpoint="http://localhost:8080",
            device_id="G1_Robot_001",
            api_key="test_api_key",
        )
        
        self.skill_manager = SkillManager()
        self.telemetry_collector = TelemetryCollector()
        
        await self.gateway_client.initialize()
        
        logger.info("G1 Edge Runtime initialized")

    async def run(self) -> None:
        """Run the edge runtime."""
        logger.info("Starting G1 Edge Runtime...")
        
        # Register device
        device_data = {
            "manufacturer": "Unitree",
            "model": "G1",
            "os_version": "3.2.1",
            "sensors": ["rgbd_camera", "lidar"],
        }
        await self.gateway_client.register_device(device_data)
        
        # Start telemetry collection
        telemetry_task = asyncio.create_task(
            self.telemetry_collector.start()
        )
        
        # Main runtime loop
        try:
            while not self.shutdown_event.is_set():
                # Send heartbeat
                await self.gateway_client.send_heartbeat()
                
                # Check skill compatibility (example)
                compatibility = await self.gateway_client.check_skill_compatibility("navigation_pro")
                logger.info(f"Skill compatibility: {compatibility}")
                
                # Sleep for a while
                await asyncio.sleep(30)
        
        except asyncio.CancelledError:
            logger.info("Edge Runtime shutting down...")
        
        finally:
            # Cleanup
            telemetry_task.cancel()
            await self.telemetry_collector.stop()
            await self.gateway_client.close()
            
            logger.info("G1 Edge Runtime stopped")

    def shutdown(self) -> None:
        """Trigger shutdown of the edge runtime."""
        self.shutdown_event.set()


async def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    runtime = EdgeRuntime()
    
    # Handle shutdown signals
    def handle_shutdown(signame: str) -> None:
        logger.info(f"Received {signame}, shutting down...")
        runtime.shutdown()
    
    for signame in ("SIGINT", "SIGTERM"):
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(
            getattr(signal, signame),
            lambda s=signame: handle_shutdown(s),
        )
    
    try:
        await runtime.initialize()
        await runtime.run()
    except Exception as e:
        logger.error(f"Edge Runtime failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
