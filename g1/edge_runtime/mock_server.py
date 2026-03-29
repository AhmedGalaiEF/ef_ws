#!/usr/bin/env python3
"""
mock_server.py
==============

Mock server for testing the G1 Edge Runtime without a real cloud platform.
This server simulates the cloud marketplace API endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from aiohttp import web

logger = logging.getLogger(__name__)


# Mock database
DEVICES: Dict[str, Dict[str, Any]] = {}
TELEMETRY_DATA: Dict[str, list] = {}
SKILLS = {
    "navigation_pro": {
        "versions": ["1.0.0", "1.1.0"],
        "compatibility": {
            "requires": ["rgbd_camera", "lidar"],
            "min_os_version": "3.0.0",
        },
    },
    "object_detection": {
        "versions": ["2.0.0"],
        "compatibility": {
            "requires": ["rgbd_camera"],
            "min_os_version": "3.1.0",
        },
    },
}


async def register_device(request: web.Request) -> web.Response:
    """Handle device registration."""
    data = await request.json()
    device_id = request.headers.get("X-Device-ID", "unknown")
    
    DEVICES[device_id] = data
    logger.info(f"Registered device: {device_id}")
    
    return web.json_response({
        "status": "success",
        "device_id": device_id,
        "message": "Device registered successfully",
    })


async def receive_telemetry(request: web.Request) -> web.Response:
    """Handle telemetry data."""
    data = await request.json()
    device_id = request.headers.get("X-Device-ID", "unknown")
    
    if device_id not in TELEMETRY_DATA:
        TELEMETRY_DATA[device_id] = []
    
    TELEMETRY_DATA[device_id].append(data)
    logger.debug(f"Received telemetry from {device_id}: {data}")
    
    return web.json_response({
        "status": "success",
        "message": "Telemetry received",
    })


async def check_skill_compatibility(request: web.Request) -> web.Response:
    """Check skill compatibility."""
    skill_id = request.match_info.get("skill_id", "")
    device_id = request.query.get("device_id", "")
    
    if skill_id not in SKILLS:
        return web.json_response(
            {
                "status": "error",
                "message": "Skill not found",
            },
            status=404,
        )
    
    if device_id not in DEVICES:
        return web.json_response(
            {
                "status": "error",
                "message": "Device not registered",
            },
            status=404,
        )
    
    device = DEVICES[device_id]
    skill = SKILLS[skill_id]
    
    # Check compatibility
    compatible = True
    constraints = []
    
    for req in skill["compatibility"]["requires"]:
        if req not in device.get("sensors", []):
            compatible = False
            constraints.append(f"requires_{req}")
    
    if device.get("os_version", "0.0.0") < skill["compatibility"]["min_os_version"]:
        compatible = False
        constraints.append(f"requires_os_{skill['compatibility']['min_os_version']}")
    
    status = "compatible" if compatible else "incompatible"
    
    return web.json_response({
        "skill_id": skill_id,
        "device_id": device_id,
        "status": status,
        "required_dependencies": [],
        "constraints": constraints,
    })


async def download_skill(request: web.Request) -> web.Response:
    """Simulate skill download."""
    skill_id = request.match_info.get("skill_id", "")
    version = request.query.get("version", "")
    
    if skill_id not in SKILLS:
        return web.json_response(
            {
                "status": "error",
                "message": "Skill not found",
            },
            status=404,
        )
    
    if version not in SKILLS[skill_id]["versions"]:
        return web.json_response(
            {
                "status": "error",
                "message": "Version not found",
            },
            status=404,
        )
    
    # Return a mock skill file
    skill_content = f"# Mock skill: {skill_id} v{version}\nprint('Running {skill_id}')\n"
    
    return web.Response(
        body=skill_content.encode(),
        content_type="application/octet-stream",
    )


async def heartbeat(request: web.Request) -> web.Response:
    """Handle device heartbeat."""
    device_id = request.match_info.get("device_id", "")
    data = await request.json()
    
    logger.debug(f"Heartbeat from {device_id}: {data}")
    
    return web.json_response({
        "status": "success",
        "message": "Heartbeat received",
    })


def create_app() -> web.Application:
    """Create the web application."""
    app = web.Application()
    
    # Routes
    app.router.add_post("/api/v1/devices/register", register_device)
    app.router.add_post("/api/v1/telemetry", receive_telemetry)
    app.router.add_get("/api/v1/skills/{skill_id}/compatibility", check_skill_compatibility)
    app.router.add_get("/api/v1/skills/{skill_id}/download", download_skill)
    app.router.add_post("/api/v1/devices/{device_id}/heartbeat", heartbeat)
    
    return app


async def main() -> None:
    """Run the mock server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, "localhost", 8080)
    await site.start()
    
    logger.info("Mock server running at http://localhost:8080")
    logger.info("Press Ctrl+C to stop")
    
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Mock server stopped")
