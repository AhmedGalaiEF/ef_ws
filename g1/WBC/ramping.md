# Why Joint-Target Ramping Is Necessary

`arm_sdk.py`'s `_ramp_to_desired` and `WBC`'s `MAX_JOINT_STEP` (0.05 rad,
`wbc.py`) both exist for the same reason: **the low-level joint controller
must never receive a large instantaneous change in `q` (desired position)**.
This note explains why, in terms of the actual control law the motors run,
and why publish frequency, clock sync, and the other command arguments
(`dq`, `kp`, `kd`, `tau`) cannot substitute for ramping the reference itself.

## 1. The low-level control law

Every joint in `rt/lowcmd`/`rt/arm_sdk` is commanded with five numbers per
tick: `q`, `dq`, `kp`, `kd`, `tau`. The motor's own controller (running
locally, independent of whatever host published the packet) computes the
torque it actually applies as:

```
tau_cmd = kp * (q_des − q_meas) + kd * (dq_des − dq_meas) + tau_ff
```

| Symbol | Meaning |
|---|---|
| `q_des` (`q`) | Desired joint position (rad) for this tick. |
| `q_meas` | Measured joint position (rad), read from the motor encoder. |
| `dq_des` (`dq`) | Desired joint velocity (rad/s) for this tick — the feedforward velocity a *consistent* trajectory would have at this instant, not an independent knob. |
| `dq_meas` | Measured joint velocity (rad/s). |
| `kp` | Position gain (N·m/rad) — how stiffly the joint corrects position error. |
| `kd` | Velocity gain / damping (N·m·s/rad) — how strongly the joint resists velocity error (and damps oscillation). |
| `tau_ff` (`tau`) | Feedforward torque (N·m) — a model-based term (e.g. gravity/inertia compensation for the *intended* trajectory), added directly, not derived from any error. |
| `tau_cmd` | The resulting commanded torque, clamped to the actuator's torque/current limit. |

This repo uses exactly this law: `arm_sdk.py`'s `_ArmSdkPublisher.publish`
sets `mc.q`, `mc.dq = 0.0`, `mc.tau = 0.0`, `mc.kp`, `mc.kd` per joint every
tick (`ARM_KP=30.0`, `ARM_KD=1.5`, `WAIST_KP=200.0`/`480.0` depending on the
caller); `WBC`'s `_tick` does the same via `publish_targets`.

## 2. Block diagram

```
                              e_q = q_des - q_meas
        q_des ----+
                  |--> ( Sigma ) --e_q--> [ Kp ] --+
        q_meas ---+       (-)                      |
                                                    v
                                              ( Sigma ) --tau_cmd--> [ Joint / Actuator ] --+--> q_meas
                                                    ^        (torque-limited plant)          |
        dq_des ---+                                |                                        +--> dq_meas
                  |--> ( Sigma ) -e_dq-> [ Kd ] ----+
        dq_meas --+       (-)                       ^
                                                      |
                                                   tau_ff
                                                (feedforward,
                                              e.g. gravity comp)

        q_meas, dq_meas are read back from the plant every tick and fed
        into the two Sigma(-) junctions above (closed-loop feedback).
```

## 3. Why a big instantaneous `q` jump is dangerous

If `q_des` jumps by `dq_step` in a single published packet while the joint
hasn't moved yet, the very next control tick sees `q_meas` essentially
unchanged, so:

```
tau_cmd (approx) = kp * dq_step
```

For this repo's arm gains (`ARM_KP = 30`), a 0.5 rad (~29 deg) jump demands
~15 N·m instantly — comparable to or above what some G1 arm joints can
continuously deliver. Two failure modes follow, and both are bad:

- **If the torque limit is not hit**: the joint accelerates as hard as
  `tau_cmd / inertia` allows toward the target — an uncommanded high-speed
  snap, not the smooth motion the caller intended. On a kinematic chain
  (shoulder -> elbow -> wrist) this couples into a large, sudden
  end-effector velocity, and on the G1 it couples back into `WBC`'s balance
  loop, which assumes quasi-static upper-body disturbances.
- **If the torque limit is hit**: the motor saturates and the response
  becomes a nonlinear, effectively bang-bang slew at max torque — no longer
  the linear PD response the gains were tuned for. This is unpredictable,
  can excite backlash/cable resonances, and is exactly the "discontinuous IK
  output" `arm_sdk.py`'s docstring warns against publishing directly.

Ramping bounds `q_des(t) - q_meas(t)` at every tick by construction, which
is the only thing that directly bounds `tau_cmd` at its source.

## 4. Why the alternatives don't reliably fix it

**Publish frequency.** Increasing how often you publish the same jump
target does not shrink the jump. The first packet the motor receives after
the jump still contains the full `q_des - q_meas` error regardless of
publish rate — a 1000 Hz publisher sending the same 0.5 rad jump produces
the same instantaneous `tau_cmd = kp * 0.5` as a 10 Hz publisher would.
What actually helps is reducing the *per-tick increment* of `q_des` — which
is precisely ramping (trajectory interpolation), not a rate change. This is
why `_ramp_to_desired` computes step count from
`max_delta / (speed_rad_s / rate_hz)`: it is deliberately shrinking the
per-publish `q_des` delta, not just publishing more often.

**Clock synchronization.** The PD law above runs locally, timed by
whichever host owns the motor loop; it does not depend on agreement between
PC1's and PC2's wall clocks (see `time_sync_setup.md`). Clock sync fixes
*cross-host timestamp correlation* (e.g. matching a PC2 camera frame to a
PC1 joint-state reading); it has no bearing on how large a single published
`q_des` step is, so it cannot reduce the resulting torque spike.

**Changing `dq`, `kp`, `kd`, or `tau` instead of ramping `q`:**

- *Raising `dq_des`* only feeds the `kd` term; it does not touch the
  `kp * (q_des - q_meas)` term, which is what spikes on a position jump. A
  `dq_des` that doesn't match the actual derivative of a smoothly-changing
  `q_des` is an inconsistent (non-physical) feedforward — it can add torque
  that fights the position term rather than resolving it, since `dq_des`
  by itself carries no information about *how* `q_des` should be reached.
- *Lowering `kp`* reduces the magnitude of a given jump's torque spike, but
  it does so by weakening the joint everywhere, all the time — worse
  steady-state accuracy (more sag under gravity/load) and weaker rejection
  of real disturbances during normal operation. It is a global
  stiffness/accuracy trade-off, not a targeted fix for one bad setpoint.
- *Raising `kd`* damps oscillation and overshoot *after* the joint starts
  moving, but at `t=0` of the jump `dq_meas` (and often `dq_des`) is ~0, so
  the `kd` term contributes almost nothing to the initial spike, which is
  dominated by the `kp` term. Overdamping also makes the joint sluggish for
  every legitimate (non-jump) command.
- *Adding `tau_ff`* is meant to feedforward known dynamics (gravity,
  expected inertial torque) *of the intended trajectory* — it is not a
  mechanism for cancelling an arbitrary `kp * step_error` spike, which would
  require computing, in real time, a correction exactly equal and opposite
  to a discontinuity that has no smooth physical model to feed forward from
  in the first place.

None of these four arguments change the size of `q_des - q_meas` on the
tick immediately after a jump; only ramping the reference trajectory itself
does. That is why `arm_sdk.ArmSdk.ik_move_EE` always ramps its IK solution
through `_ramp_to_desired` before the final target is held, and why `WBC`
independently clamps every tick's joint delta to `MAX_JOINT_STEP` as a
second, unconditional backstop.
