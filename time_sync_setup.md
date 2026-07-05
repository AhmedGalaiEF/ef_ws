# G1 Companion Computer Time Sync Setup

This sets up `chrony` on the companion computer (PC2 — the Jetson/NUC that
runs this repo's scripts) so its clock stays synchronized to G1's built-in
time source (PC1), reachable at `192.168.123.161` over the robot's internal
network. Firmware > 1.5.1 is required on the G1 side.

Run this **on the Jetson/companion computer**, not in a dev sandbox — it
edits system files (`/etc/chrony/chrony.conf`, `/etc/default/chrony`) and
restarts a system service. It was not run automatically in this session
because the sandbox that produced it is Alpine/Termux (no `systemd`, no
`apt`, not networked to the robot).

## Why this matters here

PC1 (arm/lidar/SLAM controller) and PC2 (this repo's companion computer —
RealSense, Ollama, `VLA.py`) are two independent clocks. Anything that stays
on a single host is already safe: `arm_sdk.py`'s ramping and `WBC`'s control
loop pace themselves with `time.monotonic()`, which NTP never touches. The
risk is specifically wherever a timestamp produced on one host is compared
against a timestamp produced on the other.

**EE control.** Joint state (`rt/lowstate`) and IMU come from PC1; RGBD
frames and the VLM detection are timestamped on PC2. `sdk_client.Robot
.sensors_stale()` and the `age_s` staleness checks in `naive_VLA.py`/
`VLA.py` assume both timestamps share a clock. If PC2 is skewed, a stale
depth frame can look fresh (or a fresh one can look stale), so the IK target
gets built from an object pose that's actually older or newer than believed
— the error grows with how fast the arm or the object is moving.

**Navigation.** `rt/slam_info`/`rt/slam_key_info` poses come from PC1.
Anything on PC2 that tags a detection or waypoint with "the robot was here
when I saw this" is doing a nearest-timestamp lookup against that pose
stream; clock skew becomes a straight position error in that lookup,
independent of how accurate SLAM itself is.

**Odometry.** Same mechanism: fusing PC1 odometry with PC2-side logging or
replay only lines up if the timestamps are comparable. The bigger risk is
the step change this doc's WiFi-mode warning is about — if a control or
integration loop computes `dt` from wall-clock (`time.time()`) instead of a
monotonic clock, a WiFi-triggered clock step reads as a huge or negative
`dt` and can produce a lurch or a bogus large odometry jump.

**Caveat.** This does not fix anything already paced by
`time.monotonic()` (most of this repo). It only matters once a PC1
timestamp is being correlated with a PC2 timestamp — today that's mostly a
logging/debugging concern, not something `VLA.py`'s current control loop
strictly depends on.

## Fluctuations caused by time synchronization

When PC1 performs its own network time sync, it can cause short-term time
jumps. Disable G1's WiFi mode when running programs that rely on system time
being monotonic/stable (e.g. long grasp sequences, logging with wall-clock
timestamps).

## 1. Run the setup script

```bash
sudo ./time_sync_setup.bash            # uses the default PC1 address 192.168.123.161
sudo ./time_sync_setup.bash 192.168.123.161   # or pass the address explicitly
```

The script is idempotent — re-running it is safe. It will:

1. Install `chrony` if it isn't already installed.
2. Back up any existing `/etc/chrony/chrony.conf` to `chrony.conf.orig`
   (first run only) and write a new one pointing at PC1.
3. Enable and restart the `chrony` service.
4. If `chrony` fails to start with a seccomp core-dump (a known issue on some
   images — see below), apply the documented workaround and retry.
5. Print `systemctl status chrony`, `chronyc sources -v`, and
   `chronyc tracking` so you can confirm sync in the same run.

## 2. What the config does

```
server 192.168.123.161 iburst prefer   # sync to G1's PC1 over the internal network
makestep 1.0 3                          # allow a hard step if the clock is far off at boot
rtcsync                                 # keep the hardware RTC in step too
log tracking measurements statistics
logdir /var/log/chrony
```

## 3. The seccomp core-dump issue

On some images, `chronyd` is killed by the kernel's seccomp filter on
startup:

```
Main PID: 36519 (code=dumped, signal=SYS)
...
chrony.service: Main process exited, code=dumped, status=31/SYS
```

The documented fix is to disable chrony's seccomp filtering in
`/etc/default/chrony`:

```
DAEMON_OPTS="-F 0"
```

`time_sync_setup.bash` only applies this if the first `systemctl restart
chrony` actually fails with that signature — it does not disable seccomp
filtering unconditionally, since that is a security hardening feature.

## 4. Verify sync manually

```bash
chronyc sources -v
chronyc tracking
```

`chronyc tracking` should report `Leap status : Normal` and a small (sub-100ms,
usually sub-10ms on the internal network) `System time` offset once
synchronized. This can take up to a minute after the first restart.

## 5. Reverting

```bash
sudo cp /etc/chrony/chrony.conf.orig /etc/chrony/chrony.conf   # if it existed before
sudo systemctl restart chrony
```

Or fully remove chrony: `sudo apt-get remove --purge chrony`.

## Alternative: systemd-timesyncd

The Unitree doc also documents `systemd-timesyncd` as an alternative, but
labels it "not recommended" and notes it is mutually exclusive with
chrony (installing chrony removes systemd-timesyncd). This setup does not
automate that path; if you need it instead, install `systemd-timesyncd` and
point `NTP=192.168.123.161` in `/etc/systemd/timesyncd.conf`.

## References

- Unitree G1 SDK Development Guide → Services Interface → Time Sync
  Interface (applies to G1 firmware > 1.5.1).
