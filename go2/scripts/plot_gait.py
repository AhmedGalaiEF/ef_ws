#!/usr/bin/env python3
"""Plot and save all gait recording graphs from a .jsonl file."""

import json
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

LEG_GROUPS  = {'FR': [0,1,2], 'FL': [3,4,5], 'RR': [6,7,8], 'RL': [9,10,11]}
DOF_LABELS  = ['Hip (0)', 'Thigh (1)', 'Calf (2)']
FOOT_LABELS = ['FR', 'FL', 'RR', 'RL']
LEG_COLORS  = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
AXES_KEYS   = ['lx', 'rx', 'ry', 'ly']
AXIS_LABELS = {
    'lx': 'Left stick X  (strafe)',
    'ly': 'Left stick Y  (forward)',
    'rx': 'Right stick X (yaw)',
    'ry': 'Right stick Y (unused)',
}
BUTTONS = ['R1','L1','Start','Select','R2','L2','F1','F3','A','B','X','Y','Up','Right','Down','Left']


def load(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    t0 = records[0]['wall_time']
    t  = np.array([r['wall_time'] - t0 for r in records])

    imu_rpy  = np.array([r['imu_rpy']  for r in records])
    imu_gyro = np.array([r['imu_gyro'] for r in records])
    imu_acc  = np.array([r['imu_acc']  for r in records])

    foot_force     = np.array([r['foot_force']     for r in records])
    foot_force_est = np.array([r['foot_force_est'] for r in records])

    power_v = np.array([r['power_v'] for r in records])
    power_a = np.array([r['power_a'] for r in records])

    remote_axes    = np.array([[r['remote'][k]            for k in AXES_KEYS] for r in records])
    remote_buttons = np.array([[r['remote']['buttons'][k] for k in BUTTONS]   for r in records], dtype=np.int8)

    joint_names = [j['name'] for j in records[0]['joints']]
    N, nj = len(records), len(joint_names)
    q = np.zeros((N, nj)); dq = np.zeros((N, nj))
    tau_est = np.zeros((N, nj)); temp = np.zeros((N, nj))
    for i, r in enumerate(records):
        for j in r['joints']:
            idx = joint_names.index(j['name'])
            q[i, idx] = j['q']; dq[i, idx] = j['dq']
            tau_est[i, idx] = j['tau_est']; temp[i, idx] = j['temperature']

    print(f"Loaded {N} records | duration {t[-1]:.1f} s | {N/t[-1]:.1f} Hz")
    return dict(t=t, imu_rpy=imu_rpy, imu_gyro=imu_gyro, imu_acc=imu_acc,
                foot_force=foot_force, foot_force_est=foot_force_est,
                power_v=power_v, power_a=power_a,
                remote_axes=remote_axes, remote_buttons=remote_buttons,
                joint_names=joint_names, q=q, dq=dq, tau_est=tau_est, temp=temp)


def save(fig, out_dir: Path, name: str):
    p = out_dir / f"{name}.png"
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {p.name}")


def plot_imu_rpy(d, out_dir):
    t = d['t']
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    for i, (ax, label, color) in enumerate(zip(axes, ['Roll','Pitch','Yaw'],
                                               ['tab:blue','tab:orange','tab:green'])):
        ax.plot(t, np.degrees(d['imu_rpy'][:, i]), color=color, lw=0.8)
        ax.set_ylabel(f'{label} (deg)')
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('IMU — Orientation (RPY)', fontweight='bold')
    plt.tight_layout()
    save(fig, out_dir, '01_imu_rpy')


def plot_imu_gyro_acc(d, out_dir):
    t = d['t']
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for i, label in enumerate(['x','y','z']):
        axes[0].plot(t, d['imu_gyro'][:, i], lw=0.7, label=label)
        axes[1].plot(t, d['imu_acc'][:,  i], lw=0.7, label=label)
    axes[0].set_ylabel('Gyro (rad/s)');  axes[0].legend(loc='upper right', ncol=3)
    axes[1].set_ylabel('Accel (m/s²)'); axes[1].legend(loc='upper right', ncol=3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('IMU — Gyroscope & Accelerometer', fontweight='bold')
    plt.tight_layout()
    save(fig, out_dir, '02_imu_gyro_acc')


def plot_foot_forces(d, out_dir):
    t = d['t']
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for i in range(4):
        axes[0].plot(t, d['foot_force'][:,i],     lw=0.8, color=LEG_COLORS[i], label=FOOT_LABELS[i])
        axes[1].plot(t, d['foot_force_est'][:,i], lw=0.8, color=LEG_COLORS[i], label=FOOT_LABELS[i])
    axes[0].set_ylabel('Measured force');  axes[0].legend(ncol=4)
    axes[1].set_ylabel('Estimated force'); axes[1].legend(ncol=4)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Foot Forces', fontweight='bold')
    plt.tight_layout()
    save(fig, out_dir, '03_foot_forces')


def plot_joints(d, out_dir, key, unit, title, fname):
    t = d['t']
    data = d[key]
    scale = np.degrees if unit == 'deg' else lambda x: x
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for leg_i, (leg, idxs) in enumerate(LEG_GROUPS.items()):
        for dof, (ax, idx) in enumerate(zip(axes, idxs)):
            ax.plot(t, scale(data[:, idx]), lw=0.7, color=LEG_COLORS[leg_i], label=leg)
            if leg_i == 0:
                ax.set_ylabel(f'{DOF_LABELS[dof]} ({unit})')
    axes[0].legend(ncol=4, loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    save(fig, out_dir, fname)


def plot_temperatures(d, out_dir):
    t = d['t']
    fig, ax = plt.subplots(figsize=(14, 4))
    cmap = plt.get_cmap('tab20', len(d['joint_names']))
    for i, name in enumerate(d['joint_names']):
        ax.plot(t, d['temp'][:, i], lw=0.8, color=cmap(i), label=name)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Temperature (°C)')
    ax.legend(ncol=6, loc='upper right', fontsize=8)
    ax.set_title('Motor Temperatures', fontweight='bold')
    plt.tight_layout()
    save(fig, out_dir, '07_temperatures')


def plot_power(d, out_dir):
    t = d['t']
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(t, d['power_v'],                   color='tab:blue',   lw=0.8); axes[0].set_ylabel('Voltage (V)')
    axes[1].plot(t, d['power_a'],                   color='tab:orange', lw=0.8); axes[1].set_ylabel('Current (A)')
    axes[2].plot(t, d['power_v'] * d['power_a'],    color='tab:red',    lw=0.8); axes[2].set_ylabel('Power (W)')
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Battery / Power', fontweight='bold')
    plt.tight_layout()
    save(fig, out_dir, '08_power')


def plot_phase_portraits(d, out_dir):
    t = d['t']
    hip_idxs  = [0, 3, 6, 9]
    hip_names = ['FR_0', 'FL_0', 'RR_0', 'RL_0']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, idx, name in zip(axes.flat, hip_idxs, hip_names):
        sc = ax.scatter(np.degrees(d['q'][:, idx]), d['dq'][:, idx],
                        c=t, cmap='viridis', s=1, alpha=0.6)
        ax.set_xlabel('q (deg)'); ax.set_ylabel('dq (rad/s)'); ax.set_title(name)
        plt.colorbar(sc, ax=ax, label='time (s)')
    fig.suptitle('Hip Phase Portraits (colour = time)', fontweight='bold')
    plt.tight_layout()
    save(fig, out_dir, '09_phase_portraits')


def plot_controller_axes(d, out_dir):
    t = d['t']
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    for i, (ax, key) in enumerate(zip(axes, AXES_KEYS)):
        ax.plot(t, d['remote_axes'][:, i], color=LEG_COLORS[i], lw=0.8)
        ax.axhline(0, color='k', lw=0.4, ls='--')
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel(AXIS_LABELS[key], fontsize=9)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Controller — Joystick Axes', fontweight='bold')
    plt.tight_layout()
    save(fig, out_dir, '10_controller_axes')


def plot_controller_buttons(d, out_dir):
    t = d['t']
    rb = d['remote_buttons']
    active_mask = rb.any(axis=0)
    active_btns = [b for b, m in zip(BUTTONS, active_mask) if m]
    active_data = rb[:, active_mask]

    if not active_btns:
        print("  no buttons pressed — skipping 11_controller_buttons")
        return

    fig, ax = plt.subplots(figsize=(14, max(3, len(active_btns) * 0.5)))
    cmap = plt.get_cmap('tab20', len(active_btns))
    for i, name in enumerate(active_btns):
        ax.fill_between(t, i + active_data[:, i], i, step='mid',
                        color=cmap(i), alpha=0.8, label=name)
    ax.set_yticks(range(len(active_btns)))
    ax.set_yticklabels(active_btns)
    ax.set_xlabel('Time (s)')
    ax.set_title('Controller — Button Presses (active only)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3); ax.grid(axis='y', alpha=0)
    plt.tight_layout()
    save(fig, out_dir, '11_controller_buttons')


def main():
    parser = argparse.ArgumentParser(description='Plot gait recording from .jsonl file')
    parser.add_argument('jsonl', nargs='?',
                        default='record_gait_20260518_093912.jsonl',
                        help='.jsonl recording file')
    parser.add_argument('-o', '--out', default='gait_plots',
                        help='output directory (default: gait_plots/)')
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")

    d = load(jsonl_path)

    plot_imu_rpy(d, out_dir)
    plot_imu_gyro_acc(d, out_dir)
    plot_foot_forces(d, out_dir)
    plot_joints(d, out_dir, 'q',       'deg',   'Joint Positions (q)',      '04_joint_positions')
    plot_joints(d, out_dir, 'dq',      'rad/s', 'Joint Velocities (dq)',    '05_joint_velocities')
    plot_joints(d, out_dir, 'tau_est', 'N·m',   'Joint Torques (τ_est)',    '06_joint_torques')
    plot_temperatures(d, out_dir)
    plot_power(d, out_dir)
    plot_phase_portraits(d, out_dir)
    plot_controller_axes(d, out_dir)
    plot_controller_buttons(d, out_dir)

    print(f"\nDone — {len(list(out_dir.glob('*.png')))} plots saved to {out_dir.resolve()}")


if __name__ == '__main__':
    main()
