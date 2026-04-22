import os
import sys

# Add parent directory to path to import sdk_hand
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QComboBox
    from PyQt5.QtCore import Qt
except ImportError:
    print("PyQt5 not found. Please install it using: pip install PyQt5")
    sys.exit(1)

try:
    from sdk_hand import Dex3HandController, HAND_OPEN, HAND_CLOSED
except ImportError:
    print("Could not import sdk_hand. Ensure it is in the parent directory.")
    sys.exit(1)

class GrippingApp(QWidget):
    def __init__(self, hand: str = "right"):
        super().__init__()
        self.hand = hand
        self.controller = None
        self._set_controller(self.hand)

        self.initUI()

    def _set_controller(self, hand: str) -> None:
        try:
            self.controller = Dex3HandController(hand=hand)
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
        layout.addWidget(self.slider)

        self.setLayout(layout)
        self.setWindowTitle('3-Finger Dex Hand Grip Control')
        self.setMinimumWidth(400)
        self.show()

    def on_hand_change(self, hand: str):
        self._set_controller(hand)
        self.on_slider_change()

    def on_slider_change(self):
        val = self.slider.value()
        alpha = val / 100.0
        
        # Linear interpolation between HAND_OPEN and HAND_CLOSED
        targets = [s + (e - s) * alpha for s, e in zip(HAND_OPEN, HAND_CLOSED)]
        
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    default_hand = "right"
    if len(sys.argv) > 1 and sys.argv[1] in {"left", "right"}:
        default_hand = sys.argv[1]
    ex = GrippingApp(hand=default_hand)
    sys.exit(app.exec_())
