#!/usr/bin/env python3
"""
skill_manager.py
================

Manages the installation, execution, and monitoring of skills on the robot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SkillManager:
    """Manages the installation, execution, and monitoring of skills."""

    def __init__(
        self,
        skills_dir: str = "/tmp/g1/skills",
        max_concurrent: int = 3,
    ) -> None:
        self.skills_dir = Path(skills_dir)
        self.max_concurrent = max_concurrent
        self.running_skills: Dict[str, subprocess.Popen] = {}
        
        # Ensure skills directory exists
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    async def install_skill(
        self,
        skill_id: str,
        version: str,
        skill_data: bytes,
    ) -> Path:
        """Install a skill from the given data."""
        skill_dir = self.skills_dir / f"{skill_id}_{version}"
        skill_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the skill data (placeholder - would depend on skill format)
        skill_file = skill_dir / "skill.py"
        with open(skill_file, "wb") as f:
            f.write(skill_data)
        
        logger.info(f"Installed skill {skill_id} version {version} to {skill_dir}")
        return skill_file

    async def start_skill(
        self,
        skill_id: str,
        version: str,
        args: Optional[List[str]] = None,
    ) -> None:
        """Start a skill."""
        if len(self.running_skills) >= self.max_concurrent:
            raise RuntimeError(
                f"Maximum concurrent skills reached ({self.max_concurrent})"
            )
        
        skill_dir = self.skills_dir / f"{skill_id}_{version}"
        skill_file = skill_dir / "skill.py"
        
        if not skill_file.exists():
            raise FileNotFoundError(f"Skill {skill_id} version {version} not found")
        
        # Start the skill process
        cmd = ["python3", str(skill_file)]
        if args:
            cmd.extend(args)
        
        process = subprocess.Popen(
            cmd,
            cwd=skill_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        self.running_skills[skill_id] = process
        logger.info(f"Started skill {skill_id} version {version} (PID: {process.pid})")

    async def stop_skill(self, skill_id: str) -> None:
        """Stop a running skill."""
        if skill_id not in self.running_skills:
            raise KeyError(f"Skill {skill_id} is not running")
        
        process = self.running_skills[skill_id]
        process.terminate()
        
        try:
            await asyncio.wait_for(
                asyncio.to_thread(process.wait),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            process.kill()
            logger.warning(f"Forcefully killed skill {skill_id}")
        else:
            logger.info(f"Stopped skill {skill_id}")
        
        del self.running_skills[skill_id]

    async def list_skills(self) -> List[Dict[str, Any]]:
        """List all installed skills."""
        skills = []
        
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                parts = skill_dir.name.split("_")
                if len(parts) >= 2:
                    skill_id = "_".join(parts[:-1])
                    version = parts[-1]
                    
                    skills.append({
                        "skill_id": skill_id,
                        "version": version,
                        "path": str(skill_dir),
                        "running": skill_id in self.running_skills,
                    })
        
        return skills

    async def get_skill_status(self, skill_id: str) -> Dict[str, Any]:
        """Get the status of a skill."""
        if skill_id not in self.running_skills:
            return {"status": "stopped"}
        
        process = self.running_skills[skill_id]
        return {
            "status": "running",
            "pid": process.pid,
            "returncode": process.returncode,
        }


async def main() -> None:
    """Example usage of the SkillManager."""
    logging.basicConfig(level=logging.INFO)
    
    manager = SkillManager(skills_dir="/tmp/g1_skills")
    
    # Example: List skills
    skills = await manager.list_skills()
    logger.info(f"Installed skills: {skills}")


if __name__ == "__main__":
    asyncio.run(main())
