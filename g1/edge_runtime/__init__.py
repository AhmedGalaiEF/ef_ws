"""
Edge Runtime for Unitree G1
============================

This package provides the edge runtime for the Unitree G1 robot, designed to
integrate with a cloud marketplace platform. The runtime manages device
identity, hardware fingerprinting, skill execution, and telemetry.

Core Components:
- Device Agent: Manages communication with the cloud platform.
- Skill Runtime Manager: Handles installation and execution of skills.
- Local Safety Monitor: Ensures safe operation of skills.
- Telemetry Collector: Gathers and reports system metrics.
- Secure Update Client: Manages software updates.
"""

from .device_identity import hardware_fingerprint, user_config
from .gateway import GatewayClient
from .skills import SkillManager
from .telemetry import TelemetryCollector

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"
