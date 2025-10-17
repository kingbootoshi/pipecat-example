import sys
import time
import math
import threading

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


# Joint indices for G1 29-DoF (matching Unitree examples)
class J:
    # Left arm
    L_SH_P = 15
    L_SH_R = 16
    L_SH_Y = 17
    L_EL = 18
    L_WR_R = 19
    L_WR_P = 20
    L_WR_Y = 21
    # Right arm
    R_SH_P = 22
    R_SH_R = 23
    R_SH_Y = 24
    R_EL = 25
    R_WR_R = 26
    R_WR_P = 27
    R_WR_Y = 28
    # Waist
    WAIST_Y = 12
    # Special arm_sdk switch (per examples)
    ARM_SDK_SWITCH = 29


# Conservative PD (from examples)
KP = 60.0
KD = 1.5


def ease_quint(s: float) -> float:
    # Minimum-jerk polynomial: 10s^3 - 15s^4 + 6s^5
    s = max(0.0, min(1.0, float(s)))
    return s * (s * (s * 10.0 - 15.0) + 6.0)


class GestureTest:
    """
    Minimal presenter-gesture test using rt/arm_sdk, safe and smooth.

    Run: python3 scripts/g1_presenter_gestures.py en13
    """

    def __init__(self, iface: str = "en13", seated: bool = False):
        self.iface = iface
        self.seated = seated
        # DDS init
        ChannelFactoryInitialize(0, self.iface)

        # Channels
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)

        # State
        self.crc = CRC()
        self.have_state = False
        self.q = {}
        self.q0 = {}

        # Control
        self.dt = 0.02
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="g1_gestures", daemon=True)
        self.queue = []  # list of dicts: {q_end:dict, T:float, t:float, q_start:dict}

        # which joints we drive
        self.JOINTS = [
            J.L_SH_P, J.L_SH_R, J.L_SH_Y, J.L_EL, J.L_WR_R, J.L_WR_P, J.L_WR_Y,
            J.R_SH_P, J.R_SH_R, J.R_SH_Y, J.R_EL, J.R_WR_R, J.R_WR_P, J.R_WR_Y,
            J.WAIST_Y,
        ]

    # ---------- Lowstate callback ----------
    def _on_state(self, msg: LowState_):
        for j in self.JOINTS:
            self.q[j] = msg.motor_state[j].q
            if not self.have_state:
                self.q0[j] = self.q[j]
        self.have_state = True

    # ---------- Engine control ----------
    def start(self):
        while not self.have_state:
            time.sleep(0.01)
        self.thread.start()

    def stop(self):
        self._stop.set()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass

    def _enqueue(self, q_end: dict, T: float):
        self.queue.append({
            "q_end": q_end,
            "T": max(0.01, float(T)),
            "t": 0.0,
            "q_start": {j: self.q.get(j, self.q0[j]) for j in q_end.keys()},
        })

    def _tick(self):
        if not self.queue:
            return
        item = self.queue[0]
        item["t"] += self.dt
        s = ease_quint(item["t"] / item["T"])

        cmd = unitree_hg_msg_dds__LowCmd_()
        # Enable arm sdk
        cmd.motor_cmd[J.ARM_SDK_SWITCH].q = 1

        for j in self.JOINTS:
            # default hold
            q_des = self.q.get(j, self.q0[j])
            if j in item["q_end"]:
                q0 = item["q_start"][j]
                q1 = item["q_end"][j]
                q_des = q0 + (q1 - q0) * s
            m = cmd.motor_cmd[j]
            m.tau = 0.0
            m.q = float(q_des)
            m.dq = 0.0
            m.kp = KP
            m.kd = KD

        cmd.crc = self.crc.Crc(cmd)
        self.pub.Write(cmd)

        if item["t"] >= item["T"]:
            self.queue.pop(0)

    def _run(self):
        while not self._stop.is_set():
            self._tick()
            time.sleep(self.dt)

    # ---------- Utilities ----------
    def _amp(self, level: str) -> float:
        if level == "S":
            return 0.6
        if level == "L":
            return 1.3
        return 1.0

    def _scale(self, j: int) -> float:
        if self.seated and (j == J.L_SH_P or j == J.R_SH_P):
            return 0.7
        if self.seated and (j == J.L_EL or j == J.R_EL):
            return 0.85
        return 1.0

    def _abs_from_delta(self, d: dict, amp: str) -> dict:
        a = self._amp(amp)
        out = {}
        for j, dj in d.items():
            out[j] = self.q0[j] + dj * a * self._scale(j)
        return out

    # ---------- Gesture builders (relative to neutral q0) ----------
    def beat(self, hand: str = "right", amp: str = "M", T: float = 0.45):
        d = {}
        if hand in ("right", "both"):
            d.update({J.R_SH_P: +0.12, J.R_EL: +0.08, J.R_WR_Y: +0.10})
        if hand in ("left", "both"):
            d.update({J.L_SH_P: +0.12, J.L_EL: +0.08, J.L_WR_Y: -0.10})
        self._enqueue(self._abs_from_delta(d, amp), T)

    def present_open(self, both: bool = True, amp: str = "M", T: float = 0.8):
        d = {
            J.L_SH_P: +0.35, J.L_SH_Y: +0.20, J.L_EL: +1.00, J.L_WR_P: -0.60, J.L_WR_Y: +0.15,
            J.R_SH_P: +0.35, J.R_SH_Y: -0.20, J.R_EL: +1.00, J.R_WR_P: -0.60, J.R_WR_Y: -0.15,
        }
        if not both:
            d = {k: v for k, v in d.items() if k >= J.R_SH_P}
        self._enqueue(self._abs_from_delta(d, amp), T)

    def emphasize_chop(self, hand: str = "right", amp: str = "M", T: float = 0.35):
        if hand == "right":
            d = {J.R_SH_P: +0.25, J.R_SH_Y: -0.10, J.R_EL: +0.70, J.R_WR_R: +1.20}
        else:
            d = {J.L_SH_P: +0.25, J.L_SH_Y: +0.10, J.L_EL: +0.70, J.L_WR_R: -1.20}
        self._enqueue(self._abs_from_delta(d, amp), T)

    def contrast_left(self, amp: str = "M", T: float = 0.7):
        d = {J.L_SH_P: +0.25, J.L_SH_Y: +0.25, J.L_EL: +0.95, J.L_WR_P: -0.50}
        self._enqueue(self._abs_from_delta(d, amp), T)

    def contrast_right(self, amp: str = "M", T: float = 0.7):
        d = {J.R_SH_P: +0.25, J.R_SH_Y: -0.25, J.R_EL: +0.95, J.R_WR_P: -0.50}
        self._enqueue(self._abs_from_delta(d, amp), T)

    def size_spread(self, amp: str = "M", T: float = 0.9):
        d = {
            J.L_SH_P: +0.30, J.L_SH_Y: +0.35, J.L_EL: +0.80,
            J.R_SH_P: +0.30, J.R_SH_Y: -0.35, J.R_EL: +0.80,
        }
        self._enqueue(self._abs_from_delta(d, amp), T)

    def hands_together(self, amp: str = "M", T: float = 0.7):
        d = {
            J.L_SH_P: +0.35, J.L_SH_Y: +0.05, J.L_EL: +1.20, J.L_WR_Y: +0.10,
            J.R_SH_P: +0.35, J.R_SH_Y: -0.05, J.R_EL: +1.20, J.R_WR_Y: -0.10,
        }
        self._enqueue(self._abs_from_delta(d, amp), T)

    def hands_up(self, amp: str = "M", T: float = 0.7):
        d = {
            J.L_SH_P: +0.60, J.L_SH_R: +0.15, J.L_EL: +1.00, J.L_WR_Y: +0.25,
            J.R_SH_P: +0.60, J.R_SH_R: -0.15, J.R_EL: +1.00, J.R_WR_Y: -0.25,
        }
        self._enqueue(self._abs_from_delta(d, amp), T)

    def face_to_side(self, left: bool = True, amp: str = "M", T: float = 0.45):
        d = {J.WAIST_Y: (+0.25 if left else -0.25)}
        self._enqueue(self._abs_from_delta(d, amp), T)


def main():
    print("WARNING: Ensure clear space around the robot.")
    input("Press Enter to continue...")

    iface = sys.argv[1] if len(sys.argv) > 1 else "en13"
    ge = GestureTest(iface=iface, seated=False)
    ge.start()

    # Demo sequence: simple presenter flow
    ge.present_open(both=True, amp="M", T=0.8)
    time.sleep(0.2)
    ge.beat("right", "S", 0.45)
    ge.contrast_left("M", 0.7)
    ge.contrast_right("M", 0.7)
    ge.emphasize_chop("right", "M", 0.35)
    ge.size_spread("L", 0.9)
    ge.hands_together("S", 0.7)
    ge.hands_up("M", 0.7)
    ge.face_to_side(left=True, amp="M", T=0.45)

    # Keep process alive while queue drains
    while ge.queue:
        time.sleep(0.1)

    # Small hold to maintain last pose before exiting
    time.sleep(0.5)


if __name__ == "__main__":
    main()
