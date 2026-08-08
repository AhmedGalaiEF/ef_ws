"""Speech + gesture announcement layer (spec section 16).

Does not bind hardware itself: it composes on top of ``agent/skills.py``'s
``"announce"`` and ``"gesture"`` skills -- whichever ``SkillRegistry`` is
in use, offline mock or live -- and only decides *whether* to speak/
gesture, per the four independently-controlled settings. Reusing the
existing gesture scheduler (``chatbot_with_tactile_dex3.MotionPlayer``,
via the live registry) and TTS path (``sdk_client.Robot.say`` /
``nav_bot.Speaker``, ditto) rather than adding a second announcement
transport.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .capabilities import CapabilityResolver
from .models import IntentAnnouncement, RobotStateSnapshot
from .settings.models import AgentSettings
from .skills import SkillRegistry, SkillResult, invoke_with_capability_check


@dataclass
class AnnouncementOutcome:
    spoke: bool = False
    gestured: bool = False
    speech_result: Optional[SkillResult] = None
    gesture_result: Optional[SkillResult] = None
    suppressed_reason: Optional[str] = None


def announce(
    announcement: Optional[IntentAnnouncement],
    *,
    registry: SkillRegistry,
    resolver: CapabilityResolver,
    settings: AgentSettings,
    robot_state: RobotStateSnapshot,
    is_denial: bool = False,
) -> AnnouncementOutcome:
    """Speak/gesture an intent announcement or denial notice.

    - ``announcements.audio_enabled`` / ``announcements.gesture_enabled``
      gate each channel independently -- all four combinations from spec
      section 16 are reachable: both on, audio-only, gesture-only, or
      neither (silent execution).
    - ``announcements.announce_intent_before_action`` additionally gates
      whether a *pre-action* announcement happens at all.
    - ``announcements.announce_denials`` gates whether a grounded denial
      gets spoken/gestured, independently of the above (a denial is not a
      pre-action announcement).
    """
    if announcement is None:
        return AnnouncementOutcome(suppressed_reason="no announcement requested")

    if is_denial and not settings.announcements.announce_denials:
        return AnnouncementOutcome(suppressed_reason="announcements.announce_denials=false")
    if not is_denial and not settings.announcements.announce_intent_before_action:
        return AnnouncementOutcome(suppressed_reason="announcements.announce_intent_before_action=false")

    outcome = AnnouncementOutcome()

    if announcement.speech and settings.announcements.audio_enabled:
        _decision, result = invoke_with_capability_check(
            registry,
            resolver,
            "announce",
            settings=settings,
            robot_state=robot_state,
            text=announcement.speech,
        )
        outcome.speech_result = result
        outcome.spoke = bool(result and result.ok)

    if announcement.gesture and settings.announcements.gesture_enabled:
        _decision, result = invoke_with_capability_check(
            registry,
            resolver,
            "gesture",
            settings=settings,
            robot_state=robot_state,
            name=announcement.gesture,
        )
        outcome.gesture_result = result
        outcome.gestured = bool(result and result.ok)

    return outcome
