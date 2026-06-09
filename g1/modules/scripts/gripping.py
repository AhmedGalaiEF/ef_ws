import argparse
import os
import sys

# Add parent directory to path to import sdk_hand
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QComboBox, QPushButton, QDoubleSpinBox, QHBoxLayout
    from PyQt5.QtCore import Qt, QTimer
except ImportError:
    print("PyQt5 not found. Please install it using: pip install PyQt5")
    sys.exit(1)

try:
    from sdk_hand import Dex3HandController, hand_grip_targets
except ImportError:
    print("Could not import sdk_hand. Ensure it is in the parent directory.")
    sys.exit(1)


class GrippingApp(QWidget):
    def __init__(self, hand: str = "right", iface: str = "eth0", domain_id: int = 0):
        super().__init__()
        self.hand = hand
        self.iface = str(iface)
        self.domain_id = int(domain_id)
        self.controller = None
        self._pinned_targets = None
        self._pin_interval_ms = 100
        self._pin_kp_default = 100
        self._set_controller(self.hand)

        self.initUI()

    def _set_controller(self, hand: str) -> None:
        try:
            self.controller = Dex3HandController(
                hand=hand,
                iface=self.iface,
                domain_id=self.domain_id,
            )
            self.hand = hand
        except Exception as e:
            print(f"Failed to initialize hand controller: {e}")
            self.controller = None

    def initUI(self):
        layout = QVBoxLayout()

        self.hand_selector = QComboBox(self)
        self.hand_selector.addItems(["right", "left"])
        self.hand_selector.setCurrentText(self.hand)
        self.hand_selector.currentTextChanged.connect(self.on_hand_change)
        layout.addWidget(self.hand_selector)

        self.label = QLabel(f"{self.hand.title()} Finger Grip: Open (0%)", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.slider.sliderReleased.connect(self.on_slider_released)
        layout.addWidget(self.slider)

        self.pin_button = QPushButton("Pin Pose", self)
        self.pin_button.setCheckable(True)
        self.pin_button.toggled.connect(self.on_pin_toggled)
        layout.addWidget(self.pin_button)

        resistance_row = QHBoxLayout()
        resistance_label = QLabel("Pin Resistance (kp):", self)
        resistance_row.addWidget(resistance_label)

        self.pin_kp_input = QDoubleSpinBox(self)
        self.pin_kp_input.setRange(0.1, 100.0)
        self.pin_kp_input.setSingleStep(0.1)
        self.pin_kp_input.setDecimals(2)
        self.pin_kp_input.setValue(self._pin_kp_default)
        self.pin_kp_input.valueChanged.connect(self.on_pin_resistance_changed)
        resistance_row.addWidget(self.pin_kp_input)
        layout.addLayout(resistance_row)

        self.pin_timer = QTimer(self)
        self.pin_timer.setInterval(self._pin_interval_ms)
        self.pin_timer.timeout.connect(self.publish_pinned_targets)

        self.setLayout(layout)
        self.setWindowTitle('3-Finger Dex Hand Grip Control')
        self.setMinimumWidth(400)
        self.show()

    def on_hand_change(self, hand: str):
        self._set_controller(hand)
        self._pinned_targets = None
        self.pin_timer.stop()
        self.pin_button.setChecked(False)
        self.on_slider_change()

    def on_slider_change(self):
        val = self.slider.value()
        targets = hand_grip_targets(self.hand, val)

        # Update label text
        status = "Open"
        if val == 100:
            status = "Fully Closed"
        elif val > 0:
            status = "Closing..."

        self.label.setText(f"{self.hand.title()} Finger Grip: {status} ({val}%)")

        # Send command to the hand
        if self.controller:
            # We use a very short hold_s to maintain responsiveness in the UI.
            # 0.02s corresponds to one step at 50Hz.
            self.controller.set_targets(targets, hold_s=0.02, rate_hz=50.0)

    def on_slider_released(self):
        if not self.pin_button.isChecked():
            return
        self._pinned_targets = hand_grip_targets(self.hand, self.slider.value())
        self.publish_pinned_targets()
        self.pin_timer.start()

    def on_pin_toggled(self, checked: bool):
        if checked:
            self._pinned_targets = hand_grip_targets(self.hand, self.slider.value())
            self.publish_pinned_targets()
            self.pin_timer.start()
            self.pin_button.setText("Unpin Pose")
            return

        self._pinned_targets = None
        self.pin_timer.stop()
        self.pin_button.setText("Pin Pose")

    def on_pin_resistance_changed(self, _value: float):
        if self.pin_button.isChecked():
            self.publish_pinned_targets()

    def publish_pinned_targets(self):
        if not self.controller or self._pinned_targets is None:
            return
        self.controller.set_targets(
            self._pinned_targets,
            hold_s=0.02,
            rate_hz=50.0,
            kp=self.pin_kp_input.value(),
            kd=0.1,
            tau=0.08,
        )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Dex3 hand grip control UI.")
    parser.add_argument("hand", nargs="?", choices=("left", "right"), default="right")
    parser.add_argument("--iface", default="eth0", help="Network interface for DDS traffic.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    args, remaining = parser.parse_known_args()
    return args, [sys.argv[0], *remaining]


if __name__ == '__main__':
    args, qt_argv = parse_args()
    app = QApplication(qt_argv)
    ex = GrippingApp(hand=args.hand, iface=args.iface, domain_id=args.domain_id)
    sys.exit(app.exec_())
