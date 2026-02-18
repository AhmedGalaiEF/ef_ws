# G1 SDK Details: Arme und "Hände" (High-Level + Low-Level)

Diese Datei fasst zusammen, was ein neuer Lernender braucht, um beim Unitree G1 Arme und handnahe Funktionen (Gesten/Wrist) sicher zu steuern.

## 1. Wichtige Einordnung

- In diesem Repo gibt es fuer G1 keine separate Finger-/Greifer-API (kein explizites "Dexterous Hand"-Beispiel).
- "Hand"-Funktionen sind hier:
  - High-Level Gesten/Tasks (`WaveHand`, `ShakeHand`, Arm-Action-Posen)
  - Wrist-Gelenke auf Low-Level (`LeftWristRoll`, `LeftWristPitch`, `LeftWristYaw`, rechts analog; je nach DOF verfuegbar)
- G1 nutzt `idl/unitree_hg` (nicht `unitree_go`).

## 2. Schnellstart und Laufvoraussetzungen

- Netzwerkinterface mitgeben, z. B.:
  - `python3 g1_arm_action_example.py enp3s0`
  - `python3 g1_arm7_sdk_dds_example.py enp3s0`
- DDS initialisieren:
  - `ChannelFactoryInitialize(0, sys.argv[1])`
- Sicherheitsdialog in Beispielen beachten (`WARNING ... no obstacles`).

## 3. High-Level Steuerung (einfachster Einstieg)

High-Level ist robust fuer erste Schritte, weil der Roboter intern Trajektorien/Tasks behandelt.

### 3.1 Variante A: `LocoClient` (sport/locomotion + Arm-Tasks)

### Notwendige Imports

```python
import time
import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
```

### Minimalbeispiel: Winken + Handschlag-Task

```python
import time
import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} networkInterface")
    sys.exit(-1)

ChannelFactoryInitialize(0, sys.argv[1])

client = LocoClient()
client.SetTimeout(10.0)
client.Init()

client.WaveHand()          # task_id 0
time.sleep(2)
client.WaveHand(True)      # task_id 1 (mit Drehen)
time.sleep(2)
client.ShakeHand()         # toggelt intern zwischen task_id 2 und 3
```

### Relevante interne IDs

- Service: `"sport"`
- `SetVelocity`: API 7105
- `SetTaskId` (Arm-Task): API 7106
- Arm-Task IDs:
  - `WaveHand(False)` -> `task_id=0`
  - `WaveHand(True)` -> `task_id=1`
  - `ShakeHand()` -> wechselt zwischen `task_id=2` und `task_id=3`

## 3.2 Variante B: `G1ArmActionClient` (vordefinierte Armgesten)

### Notwendige Imports

```python
import sys
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
```

### Minimalbeispiel

```python
import sys
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} networkInterface")
    sys.exit(-1)

ChannelFactoryInitialize(0, sys.argv[1])

client = G1ArmActionClient()
client.SetTimeout(10.0)
client.Init()

client.ExecuteAction(action_map["shake hand"])   # 27
time.sleep(2)
client.ExecuteAction(action_map["release arm"])  # 99
```

### Wichtige Action-IDs (`action_map`)

- `"release arm"`: 99
- `"two-hand kiss"`: 11
- `"left kiss"`: 12
- `"right kiss"`: 13
- `"hands up"`: 15
- `"clap"`: 17
- `"high five"`: 18
- `"hug"`: 19
- `"heart"`: 20
- `"right heart"`: 21
- `"reject"`: 22
- `"right hand up"`: 23
- `"x-ray"`: 24
- `"face wave"`: 25
- `"high wave"`: 26
- `"shake hand"`: 27

Hinweis: Die Demo `g1_arm_action_example.py` nutzt eigene lokale Menue-IDs `0..15`. Fuer echte API-Aufrufe zaehlen die IDs aus `action_map`.

## 4. Low-Level Steuerung fuer Arme/Wrist

Low-Level bedeutet: du setzt direkt `q/dq/kp/kd/tau` pro Gelenk.

Es gibt zwei relevante Wege:

- `rt/arm_sdk` (arm-spezifisch, siehe `g1_arm5_sdk_dds_example.py`, `g1_arm7_sdk_dds_example.py`)
- `rt/lowcmd` (voller Roboter, siehe `../low_level/g1_low_level_example.py`)

## 4.1 Joint-Indizes und Konstanten (wichtig)

### Arm/Waist Indizes

- `WaistYaw = 12`
- `WaistRoll = 13` (bei waist-locked ungueltig)
- `WaistPitch = 14` (bei waist-locked ungueltig)
- `LeftShoulderPitch = 15`
- `LeftShoulderRoll = 16`
- `LeftShoulderYaw = 17`
- `LeftElbow = 18`
- `LeftWristRoll = 19`
- `LeftWristPitch = 20` (ungueltig bei G1 23DOF)
- `LeftWristYaw = 21` (ungueltig bei G1 23DOF)
- `RightShoulderPitch = 22`
- `RightShoulderRoll = 23`
- `RightShoulderYaw = 24`
- `RightElbow = 25`
- `RightWristRoll = 26`
- `RightWristPitch = 27` (ungueltig bei G1 23DOF)
- `RightWristYaw = 28` (ungueltig bei G1 23DOF)

### Sonstige Schluesselkonstanten

- `G1_NUM_MOTOR = 29` (voller `lowcmd` Pfad)
- `kNotUsedJoint = 29` (im `arm_sdk` Beispiel als Enable-Channel genutzt)
  - `motor_cmd[29].q = 1` -> arm_sdk aktiv
  - `motor_cmd[29].q = 0` -> arm_sdk freigeben/deaktivieren

## 4.2 Notwendige Imports fuer `rt/arm_sdk`

```python
import time
import sys
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
```

### Minimalmuster (`rt/arm_sdk`)

```python
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

kp, kd = 60.0, 1.5
low_cmd = unitree_hg_msg_dds__LowCmd_()
crc = CRC()

pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
sub = ChannelSubscriber("rt/lowstate", LowState_)
pub.Init()

latest_state = {"msg": None}
def on_state(msg):
    latest_state["msg"] = msg

sub.Init(on_state, 10)
while latest_state["msg"] is None:
    time.sleep(0.01)

arm_joints = [15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 12]
low_cmd.motor_cmd[29].q = 1.0  # arm_sdk enable

for j in arm_joints:
    low_cmd.motor_cmd[j].tau = 0.0
    low_cmd.motor_cmd[j].q = latest_state["msg"].motor_state[j].q
    low_cmd.motor_cmd[j].dq = 0.0
    low_cmd.motor_cmd[j].kp = kp
    low_cmd.motor_cmd[j].kd = kd

low_cmd.crc = crc.Crc(low_cmd)
pub.Write(low_cmd)
```

## 4.3 Notwendige Imports fuer `rt/lowcmd` (voller Low-Level Pfad)

```python
import time
import sys
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
```

Wichtiger Unterschied:

- Bei `rt/lowcmd` musst du vor Kontrolle aktive Modi freigeben (`MotionSwitcherClient.CheckMode/ReleaseMode`).
- Bei `rt/arm_sdk` uebernimmst du gezielt Armkontrolle mit dem Enable-Wert auf Index `29`.

## 5. Sicherheitsregeln (unbedingt)

- Arbeitsraum freihalten, keine Personen im Schwenkbereich.
- Roboter stabil aufstellen (fester, rutschfester Boden).
- Immer weich einblenden:
  - Start mit Interpolation vom Istzustand (`ratio` von 0 auf 1).
  - Keine Spruenge in `q`.
- Erst steuern, wenn gueltiger `LowState` empfangen wurde.
- Moderate Gains fuer Einstieg:
  - Armbeispiele nutzen typischerweise `kp=60`, `kd=1.5` (arm_sdk).
- Kontrollrate sinnvoll halten:
  - arm_sdk-Beispiel: `control_dt=0.02` (50 Hz)
  - full lowcmd Beispiel: `control_dt=0.002` (500 Hz)
- Immer sauber freigeben:
  - arm_sdk zum Ende deaktivieren (`motor_cmd[29].q -> 0`).
- Not-Aus und Fernbedienung griffbereit halten.
- Bei unbekanntem DOF-Setup (23DOF/29DOF):
  - `WristPitch/WristYaw` sowie ggf. `WaistRoll/WaistPitch` nicht blind anfahren.

## 6. Typischer Lernpfad (empfohlen)

1. `g1_arm_action_example.py` ausfuehren und Action-IDs verstehen.
2. `g1_loco_client_example.py` fuer `WaveHand/ShakeHand` testen.
3. `g1_arm5_sdk_dds_example.py` (konservativer Armkanal) verstehen.
4. `g1_arm7_sdk_dds_example.py` (mehr Wrist-DOF) erweitern.
5. Erst danach `../low_level/g1_low_level_example.py` fuer umfassende Low-Level-Kontrolle nutzen.

## 7. Wichtige lokale Dateien

- `g1_arm_action_example.py`
- `g1_loco_client_example.py`
- `g1_arm5_sdk_dds_example.py`
- `g1_arm7_sdk_dds_example.py`
- `../low_level/g1_low_level_example.py`
- `../../../unitree_sdk2py/g1/arm/g1_arm_action_client.py`
- `../../../unitree_sdk2py/g1/loco/g1_loco_client.py`
