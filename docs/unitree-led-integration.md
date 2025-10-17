# Unitree LED Integration (G1) — Developer Guide

This doc explains how we control the Unitree G1 LEDs from the Pipecat voice agent, how network interfaces (the `en*` device on macOS) are chosen, how to run the test tools, and how to tune and troubleshoot the behavior. It also outlines how to extend this to emotes/commands next.

## Overview

- SDK: We use Unitree SDK2 Python (`unitree_sdk2py`). LEDs are controlled via `AudioClient.LedControl(R,G,B)`.
- Transport: SDK talks over DDS. We must initialize a DDS domain on the NIC that routes to the robot.
- App behavior:
  - At startup, LEDs are set to solid pink and kept there with a keepalive.
  - While the bot is speaking, LEDs blink pink ↔ dim pink (fast).
  - After speaking, they return to solid pink.

Files of interest (pipecat-class):
- `src/voice_agent/processors/unitree_led.py` — LED control processor
- `src/voice_agent/pipeline/handlers.py` — detects speaking state (TTS/Bot frames + audio fallback)
- Test tools under `src/voice_agent/tools/`:
  - `unitree_led_test.py` — full color and blink sequence
  - `unitree_led_pink_test.py` — solid + blink pink test
  - `unitree_led_solid_pink_css.py` — solid pink CSS (#FFC0CB)
  - `unitree_led_solid_pink.py` — alternative pink (HotPink)
  - `unitree_led_solid_red.py` — solid red

## Choosing the network interface (macOS)

When connecting a Mac to the robot via Ethernet/USB adapter, macOS will assign an `enX` device. You must pass this NIC name to the SDK/DDS initializer.

Find the right device:

- If you know the robot IP:
  - `route get <ROBOT_IP> | awk '/interface:/{print $2}'` → e.g. `en13`
- Or list hardware ports:
  - `networksetup -listallhardwareports` → look for your USB/Thunderbolt adapter (Device: en13)
- Check it’s active and has an IP:
  - `ifconfig en13 | grep status`

Run with the interface:

- One-shot:
  - `UNITREE_INTERFACE=en13 uv run voice-agent`
- In test scripts:
  - `uv run python -m voice_agent.tools.unitree_led_test en13`

On the Unitree robot (native): usually leave `UNITREE_INTERFACE` unset, or set to `eth0` if needed.

## Installing the SDK in the uv environment

- If the SDK lives in the repo as `unitree_sdk2_python/`, install it into the same uv environment used by `voice-agent`:

```
uv pip install -e ./unitree_sdk2_python
uv run python -c "import unitree_sdk2py, sys; print(unitree_sdk2py.__file__)"
```

- If you see `No module named 'cyclonedds'` on macOS, install Cyclone DDS on your machine (or run on the robot where it’s already present). On macOS, a common approach is to install CycloneDDS via your system package manager, then re-run.

## Running the voice agent with LEDs

```
UNITREE_INTERFACE=en13 uv run voice-agent
```

You should see logs:
- `[UnitreeLED] Connected via interface 'en13' (...)`
- `[UnitreeLED] apply RGB (255,10,10)` at startup (solid pink)
- While the bot speaks, repeating `apply RGB (...)` as it blinks.

The web controller stays available at `http://127.0.0.1:8710` for mic on/off.

## LED behavior and tuning

Defaults (can be overridden by env vars):
- Pink color: `RGB(255,10,10)` (hardware-verified pink)
- Blink frequency: `UNITREE_LED_HZ=4.0` (Hz)
- Dim level (off-phase): `UNITREE_LED_DIM=0.15` (15%)
- Keepalive refresh: `UNITREE_LED_KEEPALIVE_SECS=2.0`

Example overrides:
- Faster blink: `UNITREE_LED_HZ=6.0 UNITREE_INTERFACE=en13 uv run voice-agent`
- Less dim (more visible off-phase): `UNITREE_LED_DIM=0.25 ...`
- Aggressive keepalive: `UNITREE_LED_KEEPALIVE_SECS=1.0 ...`

## Test tools

- Basic sequence test:
```
uv run python -m voice_agent.tools.unitree_led_test en13
```
- Pink-only blink test (uses explicit pink ↔ dim pink):
```
uv run python -m voice_agent.tools.unitree_led_pink_test en13 --hz 4 --dim 0.6
```
- Solid reference colors:
```
uv run python -m voice_agent.tools.unitree_led_solid_pink_css en13   # CSS Pink (255,192,203)
uv run python -m voice_agent.tools.unitree_led_solid_pink en13      # HotPink (255,105,180)
uv run python -m voice_agent.tools.unitree_led_solid_red en13       # Red (255,0,0)
```

## How speaking is detected (reliable across builds)

We flip a single state flag `state.tts_speaking` that drives the LED:
- Prefer transport/service frames: `TTSStartedFrame/TTSStoppedFrame`, or `BotStartedSpeakingFrame/BotStoppedSpeakingFrame`.
- Audio fallback: watch `OutputAudioRawFrame` and consider the bot “speaking” until 350 ms of silence.

This logic lives in `src/voice_agent/pipeline/handlers.py`. The LED processor subscribes to `SharedState` and blinks while `tts_speaking=True`.

## Implementation notes

- `UnitreeLEDProcessor` is a `FrameProcessor` that:
  - Initializes DDS/AudioClient on `StartFrame` using `UNITREE_INTERFACE`.
  - Applies baseline pink immediately and maintains it (keepalive) while not speaking.
  - While speaking, blinks between bright pink and dim pink at the configured frequency.
  - Always forwards frames; it never blocks the pipeline.

- We position the processor early so it receives Start/Stop lifecycle frames. It does not depend on transport output placement because speaking state comes from handlers.

## Troubleshooting

- `No module named 'unitree_sdk2py'`
  - Install the SDK into uv: `uv pip install -e ./unitree_sdk2_python`
  - Or export `PYTHONPATH=$PWD/unitree_sdk2_python:$PYTHONPATH`

- `No module named 'cyclonedds'`
  - Install Cyclone DDS on macOS (or run on the robot); then retry.

- LED flashes once then reverts
  - The keepalive periodically re-applies the color; adjust with `UNITREE_LED_KEEPALIVE_SECS=1.0`.

- Blink looks white/blue
  - Use explicit pink and increase `UNITREE_LED_DIM` (e.g., 0.25–0.6) if dimming shifts hue on your hardware.

- Port 8710 is in use
  - Run on another port: `uv run voice-agent --port 8711`

## Extending to emotes/commands

You can drive more than LEDs — e.g., emotes on chat commands — by adding a small command processor or API:

- Add a new FastAPI route `/api/robot/emote` that accepts `{ "type": "wave" }` etc.
- Or add a custom `FrameProcessor` that listens for a `TextFrame` pattern (e.g., `!wave`) and triggers an emote routine.
- Implement emote routines as small coroutines controlling LEDs (and later, other actuators) with precise timings/patterns.

Skeleton idea for an emote processor:

```python
class EmoteProcessor(FrameProcessor):
    def __init__(self, led_client):
        super().__init__()
        self._led = led_client

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            txt = getattr(frame, "text", "").strip().lower()
            if txt.startswith("!wave"):
                await self._wave()
        await self.push_frame(frame, direction)

    async def _wave(self):
        # example: quick pink pulses
        for _ in range(6):
            self._led.set_color(255,10,10)
            await asyncio.sleep(0.08)
            self._led.set_color(40,5,5)
            await asyncio.sleep(0.08)
```

This can coexist with the main LED processor (or you can factor LED access into a small service class both use).

---

With this setup, LEDs are deterministic and robust on macOS + Unitree. Use the test tools to dial in colors and speeds, then wire additional emotes using the same control path.

