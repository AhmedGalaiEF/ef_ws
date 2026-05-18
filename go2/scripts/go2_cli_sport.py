import argparse
import curses
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_


TOPIC_HIGHSTATE = "rt/sportmodestate"

BODYHEIGHT_API_ID = 1013
FOOTRAISEHEIGHT_API_ID = 1014
SWITCHGAIT_API_ID = 1011
CONTINUOUSGAIT_API_ID = 1019

BODY_HEIGHT_MIN = -0.18
BODY_HEIGHT_MAX = 0.03
BODY_HEIGHT_STEP = 0.01
MOVE_X_MAX = 0.35
MOVE_Y_MAX = 0.20
MOVE_YAW_MAX = 0.60

GAIT_NAMES = {
    0: "idle",
    1: "trot",
    2: "run",
    3: "climb",
    4: "downstairs",
}
MODE_NAMES = {
    0: "idle",
    1: "balanceStand",
    2: "pose",
    3: "locomotion",
    5: "lieDown",
    6: "jointLock",
    7: "damping",
    8: "recoveryStand",
    10: "sit",
    11: "frontFlip",
    12: "frontJump",
    13: "frontPounce",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def vec3(values):
    return f"{values[0]: .2f} {values[1]: .2f} {values[2]: .2f}"


def age_text(ts: float) -> str:
    if ts <= 0.0:
        return "--"
    return f"{max(0.0, time.time() - ts):4.1f}s"


@dataclass
class SportSnapshot:
    state: Optional[SportModeState_] = None
    state_time: float = 0.0
    last_status: str = "idle"
    status_time: float = 0.0
    last_error: str = ""
    last_codes: str = ""
    worker_busy: bool = False


class SportController:
    def __init__(self, enable_legacy_height_rpc: bool = False):
        self.lock = threading.Lock()
        self.snapshot = SportSnapshot()
        self.enable_legacy_height_rpc = enable_legacy_height_rpc
        self.body_height = 0.0
        self.foot_raise_height = 0.0
        self.speed_level = 0
        self.gait_id = 1
        self.continuous_gait = True
        self.move_x = 0.0
        self.move_y = 0.0
        self.move_yaw = 0.0
        self.command_queue = queue.Queue()
        self.worker_running = True

    def init(self):
        self.subscriber = ChannelSubscriber(TOPIC_HIGHSTATE, SportModeState_)
        self.subscriber.Init(self._state_handler, 10)

        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        if self.enable_legacy_height_rpc:
            self._register_api(BODYHEIGHT_API_ID)
        self._register_api(FOOTRAISEHEIGHT_API_ID)
        self._register_api(SWITCHGAIT_API_ID)
        self._register_api(CONTINUOUSGAIT_API_ID)

        self.worker = threading.Thread(target=self._command_worker, name="go2_sport_worker", daemon=True)
        self.worker.start()

        self.enqueue(self._balance_stand_impl)
        self.enqueue(lambda: self._set_continuous_gait_impl(True))
        self.enqueue(lambda: self._set_gait_impl(self.gait_id))
        if self.enable_legacy_height_rpc:
            self.enqueue(self._apply_body_height_impl)
        else:
            self._set_error("Motion Services V2.0 does not document BodyHeight; legacy height RPC is disabled.")

    def _register_api(self, api_id: int):
        self.sport_client._RegistApi(api_id, 0)

    def _call_data_api(self, api_id: int, value):
        code, _ = self.sport_client._Call(api_id, json.dumps({"data": value}))
        return code

    def _set_status(self, text: str):
        with self.lock:
            self.snapshot.last_status = text
            self.snapshot.status_time = time.time()

    def _set_error(self, text: str):
        with self.lock:
            self.snapshot.last_error = text
            self.snapshot.status_time = time.time()

    def _set_codes(self, text: str):
        with self.lock:
            self.snapshot.last_codes = text

    def _set_busy(self, busy: bool):
        with self.lock:
            self.snapshot.worker_busy = busy

    def _state_handler(self, msg: SportModeState_):
        with self.lock:
            self.snapshot.state = msg
            self.snapshot.state_time = time.time()

    def enqueue(self, fn):
        self.command_queue.put(fn)

    def shutdown(self):
        self.worker_running = False
        self.command_queue.put(None)
        if hasattr(self, "worker"):
            self.worker.join(timeout=1.0)

    def _command_worker(self):
        while self.worker_running:
            fn = self.command_queue.get()
            if fn is None:
                break
            self._set_busy(True)
            try:
                fn()
            except Exception as exc:
                self._set_error(str(exc))
                self._set_codes("exception")
                self._set_status(f"Command failed: {exc}")
            finally:
                self._set_busy(False)

    def balance_stand(self):
        self.enqueue(self._balance_stand_impl)

    def _balance_stand_impl(self):
        code = self.sport_client.BalanceStand()
        self._set_codes(f"BalanceStand={code}")
        self._set_status(f"BalanceStand -> {code}")
        return code

    def sit(self):
        self.move_x = 0.0
        self.move_y = 0.0
        self.move_yaw = 0.0
        self.enqueue(self._sit_impl)

    def _sit_impl(self):
        self._stop_move_impl()
        code = self.sport_client.StandDown()
        self._set_codes(f"StandDown={code}")
        self._set_status(f"StandDown -> {code}")
        return code

    def recovery_stand(self):
        self.enqueue(self._recovery_stand_impl)

    def _recovery_stand_impl(self):
        code = self.sport_client.RecoveryStand()
        self._set_codes(f"RecoveryStand={code}")
        self._set_status(f"RecoveryStand -> {code}")
        return code

    def stop_move(self):
        self.move_x = 0.0
        self.move_y = 0.0
        self.move_yaw = 0.0
        self.enqueue(self._stop_move_impl)

    def _stop_move_impl(self):
        code = self.sport_client.StopMove()
        self._set_codes(f"StopMove={code}")
        self._set_status(f"StopMove -> {code}")
        return code

    def apply_move(self):
        self.enqueue(self._apply_move_impl)

    def _apply_move_impl(self):
        code = self.sport_client.Move(self.move_x, self.move_y, self.move_yaw)
        self._set_codes(f"Move={code}")
        self._set_status(
            f"Move -> {code} ({self.move_x:+.2f}, {self.move_y:+.2f}, {self.move_yaw:+.2f})"
        )
        return code

    def increase_body_height(self):
        self.body_height = clamp(self.body_height + BODY_HEIGHT_STEP, BODY_HEIGHT_MIN, BODY_HEIGHT_MAX)
        self.apply_body_height()

    def decrease_body_height(self):
        self.body_height = clamp(self.body_height - BODY_HEIGHT_STEP, BODY_HEIGHT_MIN, BODY_HEIGHT_MAX)
        self.apply_body_height()

    def apply_body_height(self):
        if not self.enable_legacy_height_rpc:
            self._set_codes("BodyHeight=unsupported")
            self._set_status(f"BodyHeight command ignored ({self.body_height:+.2f})")
            self._set_error("BodyHeight is not part of Motion Services V2.0. Re-run with --enable-legacy-height-rpc only on legacy-compatible firmware.")
            return
        self.enqueue(self._apply_body_height_impl)

    def _apply_body_height_impl(self):
        stand_code = self.sport_client.BalanceStand()
        time.sleep(0.05)
        body_code = self._call_data_api(BODYHEIGHT_API_ID, self.body_height)
        self._set_codes(f"BalanceStand={stand_code}, BodyHeight={body_code}")
        self._set_status(
            f"BalanceStand -> {stand_code}, BodyHeight({self.body_height:+.2f}) -> {body_code}"
        )
        if stand_code != 0 or body_code != 0:
            self._set_error(
                f"BodyHeight request failed or was rejected: BalanceStand={stand_code}, BodyHeight={body_code}"
            )
        else:
            self._set_error("")
        return body_code

    def apply_foot_raise_height(self):
        self.enqueue(self._apply_foot_raise_height_impl)

    def _apply_foot_raise_height_impl(self):
        code = self._call_data_api(FOOTRAISEHEIGHT_API_ID, self.foot_raise_height)
        self._set_codes(f"FootRaiseHeight={code}")
        self._set_status(f"FootRaiseHeight({self.foot_raise_height:+.2f}) -> {code}")
        return code

    def set_gait(self, gait_id: int):
        self.gait_id = gait_id
        self.enqueue(lambda: self._set_gait_impl(gait_id))

    def _set_gait_impl(self, gait_id: int):
        code = self._call_data_api(SWITCHGAIT_API_ID, gait_id)
        self._set_codes(f"SwitchGait={code}")
        self._set_status(f"SwitchGait({gait_id}) -> {code}")
        return code

    def cycle_gait(self):
        gait_ids = [0, 1, 2, 3, 4]
        idx = gait_ids.index(self.gait_id) if self.gait_id in gait_ids else 0
        return self.set_gait(gait_ids[(idx + 1) % len(gait_ids)])

    def set_continuous_gait(self, flag: bool):
        self.continuous_gait = flag
        self.enqueue(lambda: self._set_continuous_gait_impl(flag))

    def _set_continuous_gait_impl(self, flag: bool):
        code = self._call_data_api(CONTINUOUSGAIT_API_ID, flag)
        self._set_codes(f"ContinuousGait={code}")
        self._set_status(f"ContinuousGait({flag}) -> {code}")
        return code

    def toggle_continuous_gait(self):
        return self.set_continuous_gait(not self.continuous_gait)

    def set_speed_level(self, level: int):
        self.speed_level = max(-1, min(1, level))
        self.enqueue(self._set_speed_level_impl)

    def _set_speed_level_impl(self):
        code = self.sport_client.SpeedLevel(self.speed_level)
        self._set_codes(f"SpeedLevel={code}")
        self._set_status(f"SpeedLevel({self.speed_level}) -> {code}")
        return code

    def cycle_speed_level(self):
        levels = [-1, 0, 1]
        idx = levels.index(self.speed_level)
        return self.set_speed_level(levels[(idx + 1) % len(levels)])

    def set_move_command(self, vx: float, vy: float, vyaw: float):
        self.move_x = clamp(vx, -MOVE_X_MAX, MOVE_X_MAX)
        self.move_y = clamp(vy, -MOVE_Y_MAX, MOVE_Y_MAX)
        self.move_yaw = clamp(vyaw, -MOVE_YAW_MAX, MOVE_YAW_MAX)
        return self.apply_move()

    def snapshot_data(self):
        with self.lock:
            return {
                "state": self.snapshot.state,
                "state_time": self.snapshot.state_time,
                "status": self.snapshot.last_status,
                "status_time": self.snapshot.status_time,
                "last_error": self.snapshot.last_error,
                "last_codes": self.snapshot.last_codes,
                "worker_busy": self.snapshot.worker_busy,
                "body_height_cmd": self.body_height,
                "foot_raise_height_cmd": self.foot_raise_height,
                "gait_id_cmd": self.gait_id,
                "speed_level_cmd": self.speed_level,
                "continuous_gait_cmd": self.continuous_gait,
                "move_cmd": (self.move_x, self.move_y, self.move_yaw),
            }


def draw(stdscr, controller: SportController):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    snap = controller.snapshot_data()
    state = snap["state"]
    vx, vy, vyaw = snap["move_cmd"]

    lines = [
        "Go2 CLI Sport",
        "Arrows: move/turn  +/-: body height  g: gait  c: continuous gait  v: speed lvl  b: balance stand",
        "s: stand down  r: recovery stand  x: stop  q: quit",
        "",
        f"Cmd body height: {snap['body_height_cmd']:+.2f} m",
        f"Cmd move vx/vy/vyaw: {vx:+.2f} {vy:+.2f} {vyaw:+.2f}",
        f"Cmd gait: {GAIT_NAMES.get(snap['gait_id_cmd'], snap['gait_id_cmd'])}",
        f"Cmd speed level: {snap['speed_level_cmd']}",
        f"Cmd continuous gait: {snap['continuous_gait_cmd']}",
        f"Worker busy: {snap['worker_busy']}",
        f"Last status: {snap['status']} (age {age_text(snap['status_time'])})",
        f"Last RPC codes: {snap['last_codes'] or '--'}",
        f"Last error: {snap['last_error'] or '--'}",
        "",
        f"Sport state topic: {TOPIC_HIGHSTATE} age={age_text(snap['state_time'])}",
    ]

    if state is None:
        lines.append("Waiting for rt/sportmodestate ...")
    else:
        lines.extend(
            [
                f"Mode: {MODE_NAMES.get(int(state.mode), int(state.mode))}  progress={float(state.progress):.2f}",
                f"Gait: {GAIT_NAMES.get(int(state.gait_type), int(state.gait_type))}",
                f"Body height actual/cmd: {float(state.body_height):+.2f} / {snap['body_height_cmd']:+.2f} m",
                f"Foot raise actual/cmd: {float(state.foot_raise_height):+.2f} / {snap['foot_raise_height_cmd']:+.2f} m",
                f"Position xyz: {vec3([float(v) for v in state.position])}",
                f"Velocity xyz: {vec3([float(v) for v in state.velocity])}",
                f"Yaw speed: {float(state.yaw_speed):+.2f} rad/s",
                f"IMU rpy: {vec3([float(v) for v in state.imu_state.rpy])}",
                f"Foot force: {' '.join(str(int(v)) for v in state.foot_force)}",
                f"Error code: {int(state.error_code)}",
            ]
        )

    if h < 18 or w < 100:
        lines.append("")
        lines.append("Terminal is small; widen it for the full panel.")

    for idx, line in enumerate(lines[: max(0, h - 1)]):
        stdscr.addnstr(idx, 0, line, max(0, w - 1))
    stdscr.refresh()


def tui_main(stdscr, controller: SportController):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    while True:
        draw(stdscr, controller)
        key = stdscr.getch()
        if key == curses.KEY_UP:
            controller.set_move_command(MOVE_X_MAX, 0.0, 0.0)
        elif key == curses.KEY_DOWN:
            controller.set_move_command(-MOVE_X_MAX, 0.0, 0.0)
        elif key == curses.KEY_LEFT:
            controller.set_move_command(0.0, 0.0, MOVE_YAW_MAX)
        elif key == curses.KEY_RIGHT:
            controller.set_move_command(0.0, 0.0, -MOVE_YAW_MAX)
        elif key in (ord("+"), ord("=")):
            controller.increase_body_height()
        elif key in (ord("-"), ord("_")):
            controller.decrease_body_height()
        elif key in (ord("g"), ord("G")):
            controller.cycle_gait()
        elif key in (ord("c"), ord("C")):
            controller.toggle_continuous_gait()
        elif key in (ord("v"), ord("V")):
            controller.cycle_speed_level()
        elif key in (ord("b"), ord("B")):
            controller.balance_stand()
        elif key in (ord("s"), ord("S")):
            controller.sit()
        elif key in (ord("r"), ord("R")):
            controller.recovery_stand()
        elif key in (ord("x"), ord("X")):
            controller.stop_move()
        elif key in (ord("q"), ord("Q")):
            controller.stop_move()
            break


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-level Go2 sport-mode CLI using Move and BodyHeight."
    )
    parser.add_argument("--iface", required=True, help="Robot network interface")
    parser.add_argument(
        "--enable-legacy-height-rpc",
        action="store_true",
        help="Try the undocumented legacy BodyHeight RPC. Use only on firmware that still supports it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("WARNING: Ensure the robot has clearance before using sport mode control.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, args.iface)

    controller = SportController(enable_legacy_height_rpc=args.enable_legacy_height_rpc)
    controller.init()
    try:
        curses.wrapper(tui_main, controller)
    finally:
        controller.stop_move()
        controller.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
