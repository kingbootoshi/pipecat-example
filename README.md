Voice Agent Module (Pipecat)
===========================

Modular, standalone voice-to-voice agent built on Pipecat. This module can run on its own or be embedded into a larger robot "brain" system.

Highlights
- Modular architecture under `src/voice_agent/`
- Config-first (YAML) personality and service settings
- Optional webcam vision and conversation memory
- CLI entrypoint (`voice-agent`) and `python -m voice_agent`

Quick Start
- Ensure Python 3.10+
- Set environment variables (e.g. via `.env`):
  - `ASSEMBLYAI_API_KEY`
  - `OPENROUTER_API_KEY`
  - `ELEVENLABS_API_KEY`
- Configure `config.yml` at repo root

Run
- CLI (after install): `voice-agent --config ./config.yml`
- Module: `python -m voice_agent --config ./config.yml`
  - If running without install, add `src` to `PYTHONPATH`, e.g. `PYTHONPATH=src python -m voice_agent`

CLI Flags
- `--config PATH`            Path to YAML config (default: `./config.yml`)
- `--no-vision`              Disable webcam image capture
- `--no-memory`              Disable conversation memory persistence
- `--allow-interruptions`    Allow user to interrupt TTS
- `--log-level LEVEL`        Logging level (DEBUG/INFO/...)

Programmatic Usage
```python
from voice_agent import VoiceAgent
import asyncio

async def main():
    agent = VoiceAgent(config_path="./config.yml", vision=True, memory=True)
    await agent.start()

asyncio.run(main())
```

Project Structure
```
src/voice_agent/
  app.py                    # VoiceAgent class (public API)
  cli.py                    # CLI entrypoint
  config.py                 # YAML loader
  constants.py              # Paths (config, conversations)
  logging.py                # Loguru setup
  settings.py               # .env loading + key checks
  memory/
    conversation_store.py   # save/load conversation
  observers/
    whisker.py              # optional Whisker observer
  pipeline/
    builder.py              # Pipeline assembly
    params.py               # Pipeline params factory
  processors/
    transcription_logger.py
    llm_response_logger.py
    image_capture.py
    filters.py
  services/
    stt.py, llm.py, tts.py
  transports/
    local_audio.py
```

Notes
- `config.yml` and `conversations/` remain at repo root
- Use `--no-vision` if no webcam is available
- Memory is optional and stored at `./conversations/conversation.json`
