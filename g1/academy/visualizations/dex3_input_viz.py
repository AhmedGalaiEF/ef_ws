#!/usr/bin/env python3
"""
Dex3 tactile input visualizer.

Live view of both Dex3 hands' fingertip pressure sensors
(`sdk_wrapper.G1.get_dex3_hand_sensors()`, i.e. `rt/dex3/{left,right}/state`'s
`press_sensor_state`), compared in real time against one configurable contact
threshold — the same raw-value convention `recognition_app_v3.py`'s tactile
readout and `test_dex3_tactile.py` use (fixed here as
`TACTILE_INVALID_VALUE`/`TACTILE_VALID_THRESHOLD`, not rediscovered).

Each hand has an unlabeled list of pressure pads (one `press_sensor_state`
entry per pad — the DDS message doesn't name them, so neither does this
script: they're shown as "pad 0", "pad 1", ...), each itself an array of
per-taxel raw readings. A reading equal to `TACTILE_INVALID_VALUE` means "no
data this tick" (sensor dropout), not "zero pressure", and is excluded from
every max/plot rather than treated as a real low reading.

Three views, all from the same polled snapshot:
  * Per-pad bar chart (current instant) — one bar per pad per hand, the max
    valid taxel reading in that pad, colored by whether it clears the
    threshold.
  * Rolling strip chart (last `--history-s` seconds) — each hand's overall
    max-valid-pressure over time against the threshold line, so you can see
    a grip approaching/leaving contact.
  * A raw per-taxel table per hand, for the pad selected in the bar chart —
    the actual numbers behind the bar, invalid taxels called out.

Run:
    python3 dex3_input_viz.py [--iface eth0] [--domain-id 0] [--port 8071]
Then open http://<host>:8071 in a browser on the same network. The page loads
even with no robot reachable; click Connect once the network is up.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, dash_table

# Make `from sdk_wrapper import G1` work regardless of cwd. This file lives in
# academy/visualizations/; sdk_wrapper.py sits one level up in academy/.
_SCRIPT_DIR = Path(__file__).resolve().parent      # academy/visualizations
_ACADEMY_DIR = _SCRIPT_DIR.parent                    # academy
_G1_ROOT_DIR = _ACADEMY_DIR.parent                   # g1
for _p in (_SCRIPT_DIR, _ACADEMY_DIR, _G1_ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Tactile constants — same raw-value convention recognition_app_v3.py and
# modules/scripts/test_dex3_tactile.py use for this same DDS field.
# ---------------------------------------------------------------------------
TACTILE_INVALID_VALUE = 30000.0     # sensor-dropout sentinel, not "low pressure"
TACTILE_INVALID_TOL = 0.5
DEFAULT_THRESHOLD_RAW = 100000.0    # matches recognition_app_v3.TACTILE_VALID_THRESHOLD
MAX_THRESHOLD_RAW = 200000.0

POLL_HZ = 8.0
HISTORY_S = 60.0
PLOT_INTERVAL_MS = 400
HAND_COLOR = {"left": "#3987e5", "right": "#d95926"}


def _is_invalid(value: float) -> bool:
    return abs(float(value) - TACTILE_INVALID_VALUE) < TACTILE_INVALID_TOL


def _pad_max(pad_values: list[float]) -> Optional[float]:
    valid = [float(v) for v in pad_values if not _is_invalid(v)]
    return max(valid) if valid else None


# ---------------------------------------------------------------------------
# Robot connection + poll loop
# ---------------------------------------------------------------------------

class HandLink:
    """Owns the G1 connection and the tactile poll loop.

    Read-only — this dashboard never commands the hands, only reads
    `get_dex3_hand_sensors()`, so there is no publish loop to mirror
    joint_control_dashboard.py's RobotLink.
    """

    def __init__(self, iface: str, domain_id: int):
        self.iface = str(iface)
        self.domain_id = int(domain_id)
        self.lock = threading.RLock()

        self.g1 = None
        self.connect_requested = False
        self.init_err: Optional[str] = None

        self.snapshot: dict[str, Any] = {"left": None, "right": None}
        self.snapshot_ts = 0.0
        self.poll_err: Optional[str] = None

        # (timestamp, max_valid_pressure_or_None) per hand, most recent last.
        self.history: dict[str, deque] = {
            "left": deque(maxlen=int(HISTORY_S * POLL_HZ) + 4),
            "right": deque(maxlen=int(HISTORY_S * POLL_HZ) + 4),
        }

        self._stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

    def connect(self) -> None:
        with self.lock:
            self.connect_requested = True
            if self.g1 is not None:
                return
            try:
                from sdk_wrapper import G1  # deferred: only needed once connecting
                self.g1 = G1(self.iface, domain_id=self.domain_id)
                self.init_err = None
            except Exception as exc:
                self.init_err = str(exc)
                self.g1 = None
                return
            self._stop.clear()
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def disconnect(self) -> None:
        with self.lock:
            self.connect_requested = False
            self._stop.set()
            self.g1 = None

    def status(self) -> tuple[str, str]:
        with self.lock:
            if self.g1 is not None:
                return "Connected", "success"
            if not self.connect_requested:
                return "Disconnected", "secondary"
            if self.init_err is not None:
                return "Error", "danger"
            return "Connecting…", "warning"

    def _poll_loop(self) -> None:
        period = 1.0 / POLL_HZ
        while not self._stop.is_set():
            g1 = self.g1
            if g1 is not None:
                try:
                    snap = g1.get_dex3_hand_sensors(hand="both")
                    now = time.time()
                    with self.lock:
                        self.snapshot = snap
                        self.snapshot_ts = now
                        self.poll_err = None
                        for side in ("left", "right"):
                            side_snap = snap.get(side) if isinstance(snap, dict) else None
                            pads = (side_snap or {}).get("tactile_pressures") or []
                            pad_maxes = [m for m in (_pad_max(p) for p in pads) if m is not None]
                            overall = max(pad_maxes) if pad_maxes else None
                            self.history[side].append((now, overall))
                except Exception as exc:
                    with self.lock:
                        self.poll_err = str(exc)
            time.sleep(period)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _empty_bar_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", title=title, height=280,
        margin={"l": 40, "r": 10, "t": 36, "b": 30},
        xaxis={"title": "pad"}, yaxis={"title": "pressure (raw)"},
    )
    return fig


def build_bar_figure(side: str, side_snap: Optional[dict], threshold: float) -> go.Figure:
    fig = _empty_bar_figure(f"{side.title()} hand — pad pressure (now)")
    pads = (side_snap or {}).get("tactile_pressures") or []
    if not pads:
        fig.add_annotation(text="no data", showarrow=False, x=0.5, y=0.5,
                            xref="paper", yref="paper", font={"color": "#888"})
        return fig
    labels = [f"pad {i}" for i in range(len(pads))]
    maxes = [_pad_max(p) for p in pads]
    colors = []
    text = []
    for m in maxes:
        if m is None:
            colors.append("#4a4f59")
            text.append("n/a")
        elif m >= threshold:
            colors.append("#e0294f")
            text.append(f"{m:,.0f}")
        else:
            colors.append("#3aa876")
            text.append(f"{m:,.0f}")
    values = [0.0 if m is None else m for m in maxes]
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=colors, text=text, textposition="outside"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#f5c84b",
                  annotation_text="threshold", annotation_position="top left")
    y_top = max([threshold * 1.15] + [v for v in values])
    fig.update_yaxes(range=[0, y_top])
    return fig


def build_history_figure(history: dict[str, deque], threshold: float) -> go.Figure:
    fig = go.Figure()
    now = time.time()
    for side in ("left", "right"):
        points = [(t, v) for t, v in history[side] if v is not None]
        if not points:
            continue
        xs = [t - now for t, _ in points]  # seconds ago, 0 = now
        ys = [v for _, v in points]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"{side} max",
            line={"color": HAND_COLOR[side], "width": 2},
        ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#f5c84b",
                  annotation_text="threshold", annotation_position="top left")
    fig.update_layout(
        template="plotly_dark", title=f"Max pad pressure — last {HISTORY_S:.0f}s",
        height=300, margin={"l": 50, "r": 10, "t": 36, "b": 36},
        xaxis={"title": "seconds ago", "range": [-HISTORY_S, 0]},
        yaxis={"title": "pressure (raw)", "rangemode": "tozero"},
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def build_taxel_table(side_snap: Optional[dict]) -> list[dict]:
    pads = (side_snap or {}).get("tactile_pressures") or []
    rows = []
    for pad_idx, taxels in enumerate(pads):
        for taxel_idx, value in enumerate(taxels):
            rows.append({
                "pad": pad_idx,
                "taxel": taxel_idx,
                "raw": "invalid" if _is_invalid(value) else f"{value:,.0f}",
            })
    return rows


# ---------------------------------------------------------------------------
# Dash app / layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "G1 Dex3 Tactile"


def _hand_panel(side: str) -> dbc.Col:
    return dbc.Col([
        dcc.Graph(id=f"bar-{side}", figure=_empty_bar_figure(f"{side.title()} hand — pad pressure (now)")),
        html.Div(id=f"status-{side}", className="mb-2", style={"fontSize": "13px"}),
        dash_table.DataTable(
            id=f"table-{side}",
            columns=[{"name": c, "id": c} for c in ("pad", "taxel", "raw")],
            data=[],
            page_size=8,
            style_table={"maxHeight": "220px", "overflowY": "auto"},
            style_header={"backgroundColor": "#1c1a1e", "color": "#e8ecf3", "fontSize": "11px"},
            style_cell={"backgroundColor": "#14171d", "color": "#e8ecf3", "fontSize": "12px",
                        "border": "1px solid #2b303b"},
        ),
    ], md=6)


app.layout = dbc.Container([
    html.H3("G1 Dex3 Tactile Input", className="mt-3"),
    dbc.Row([
        dbc.Col(dbc.Input(id="iface-input", value="eth0", placeholder="iface"), width="auto"),
        dbc.Col(dbc.Input(id="domain-input", type="number", value=0, placeholder="domain id"), width="auto"),
        dbc.Col(dbc.Button("Connect", id="btn-connect", color="primary"), width="auto"),
        dbc.Col(dbc.Button("Disconnect", id="btn-disconnect", color="secondary"), width="auto"),
        dbc.Col(dbc.Badge("Disconnected", id="conn-badge", color="secondary"), width="auto"),
        dbc.Col(html.Div([
            html.Span("Contact threshold (raw): ", style={"fontSize": "13px"}),
            dcc.Input(id="threshold-input", type="number", value=DEFAULT_THRESHOLD_RAW,
                      min=0, max=MAX_THRESHOLD_RAW, step=1000, style={"width": "110px"}),
        ]), width="auto"),
    ], align="center", className="mb-3 gy-2"),
    dbc.Row([_hand_panel("left"), _hand_panel("right")]),
    dcc.Graph(id="history-graph", figure=build_history_figure({"left": deque(), "right": deque()}, DEFAULT_THRESHOLD_RAW)),
    html.Div(id="poll-err", className="mt-2", style={"fontSize": "12px", "color": "#e0294f"}),
    dcc.Interval(id="status-interval", interval=1000, n_intervals=0),
    dcc.Interval(id="plot-interval", interval=PLOT_INTERVAL_MS, n_intervals=0),
], fluid=True)


def _status_text(side: str, side_snap: Optional[dict], threshold: float) -> tuple[str, str]:
    pads = (side_snap or {}).get("tactile_pressures") or []
    if not pads:
        return f"{side.title()}: no data", "#888888"
    maxes = [m for m in (_pad_max(p) for p in pads) if m is not None]
    if not maxes:
        return f"{side.title()}: all pads invalid", "#888888"
    peak = max(maxes)
    peak_pad = max(range(len(pads)), key=lambda i: (_pad_max(pads[i]) if _pad_max(pads[i]) is not None else -1))
    state = "CONTACT" if peak >= threshold else "no contact"
    color = "#e0294f" if peak >= threshold else "#3aa876"
    return f"{side.title()}: {state} — pad {peak_pad} = {peak:,.0f} (threshold {threshold:,.0f})", color


@app.callback(
    Output("conn-badge", "children"),
    Output("conn-badge", "color"),
    Input("btn-connect", "n_clicks"),
    Input("btn-disconnect", "n_clicks"),
    Input("status-interval", "n_intervals"),
    State("iface-input", "value"),
    State("domain-input", "value"),
    prevent_initial_call=False,
)
def on_connection(_connect, _disconnect, _tick, iface, domain_id):
    trig = dash.ctx.triggered_id
    if trig == "btn-connect":
        LINK.iface = str(iface or "eth0")
        try:
            LINK.domain_id = int(domain_id or 0)
        except (TypeError, ValueError):
            LINK.domain_id = 0
        LINK.connect()
    elif trig == "btn-disconnect":
        LINK.disconnect()
    label, color = LINK.status()
    return label, color


@app.callback(
    Output("bar-left", "figure"),
    Output("bar-right", "figure"),
    Output("status-left", "children"),
    Output("status-left", "style"),
    Output("status-right", "children"),
    Output("status-right", "style"),
    Output("table-left", "data"),
    Output("table-right", "data"),
    Output("history-graph", "figure"),
    Output("poll-err", "children"),
    Input("plot-interval", "n_intervals"),
    Input("threshold-input", "value"),
)
def on_plot_tick(_n, threshold):
    threshold = float(threshold or DEFAULT_THRESHOLD_RAW)
    with LINK.lock:
        snap = dict(LINK.snapshot)
        history = {side: deque(LINK.history[side]) for side in ("left", "right")}
        err = LINK.poll_err

    left_snap = snap.get("left")
    right_snap = snap.get("right")

    left_text, left_color = _status_text("left", left_snap, threshold)
    right_text, right_color = _status_text("right", right_snap, threshold)

    return (
        build_bar_figure("left", left_snap, threshold),
        build_bar_figure("right", right_snap, threshold),
        left_text, {"fontSize": "13px", "color": left_color},
        right_text, {"fontSize": "13px", "color": right_color},
        build_taxel_table(left_snap),
        build_taxel_table(right_snap),
        build_history_figure(history, threshold),
        f"poll error: {err}" if err else "",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 Dex3 tactile input visualizer.")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8071)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    global LINK
    LINK = HandLink(args.iface, args.domain_id)
    print(f"Dex3 tactile viz: http://{args.host}:{args.port} iface={args.iface} domain={args.domain_id}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


LINK = HandLink("eth0", 0)

if __name__ == "__main__":
    raise SystemExit(main())
