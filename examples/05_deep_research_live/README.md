# Deep Research Demo — Live Helix Visualisation

Watch Felix agents spiral from broad exploration to focused synthesis in real time.

```
       ╔═══════════════════════════════════════╗
       ║  F E L I X  Deep Research Demo        ║
       ║  Helical multi-agent orchestration    ║
       ╚═══════════════════════════════════════╝
```

## What This Shows

- **4 agents** (2 Research, 1 Analysis, 1 Critic) traverse a 3D helix
- Agents start **wide** (exploration: high temperature, broad search) and spiral **inward** (synthesis: low temperature, focused output)
- **Live ASCII visualisation** renders the helix cross-section with agent positions, confidence bars, and phase transitions
- **No API key needed** — ships with a rich mock provider that returns realistic multi-agent research output

## Quick Start

```bash
# From the repo root
pip install -e ".[dev]"

# Run with mock provider (no API key)
python examples/05_deep_research_live/run.py

# Skip animations for quick testing
python examples/05_deep_research_live/run.py --fast

# Custom topic
python examples/05_deep_research_live/run.py --topic "the future of quantum error correction"

# Use a real provider (requires API key)
python examples/05_deep_research_live/run.py --provider anthropic
python examples/05_deep_research_live/run.py --provider openai
python examples/05_deep_research_live/run.py --provider local  # LM Studio / Ollama
```

## What You'll See

```
  F E L I X  Deep Research Demo  │  Round 2/4  │  3.2s elapsed
  ═══════════════════════════════════════════════════════════════
  Phases: ~ EXPLORE  = ANALYSE  # SYNTHESISE

              ~                        │ Agents
        ~          [R]                 │ ---------------------
     ~                                 │ [R] research-001
  --------~---[R]------[A]----------- │   ████████████░░░░  52.3%
              =                        │   conf:0.48  temp:0.43
          =                            │
  ------------=------[C]------------- │ [C] critic-004
            #                          │   ██████████░░░░░░  41.0%
          #                            │   conf:0.55  temp:0.35
  ═══════════════════════════════════════════════════════════════
  Team Confidence: ██████████████░░░░░░░░ 52.3%
```

## Architecture

This demo manually orchestrates the same components that `FelixWorkflow.run()` uses internally, but hooks into each step to update the terminal visualisation:

1. **AgentFactory** creates a team from the workflow config
2. **CentralPost + SpokeManager** handle hub-spoke message routing
3. **CollaborativeContextBuilder** feeds previous agent outputs as context for the next round
4. **HelixVisualizer** renders agent positions on the helix cross-section each frame
5. **WorkflowSynthesizer** produces the final merged output

## Recording a GIF for Social Media

The easiest way to capture a shareable terminal GIF:

### Option A: VHS (recommended — deterministic, scriptable)

Install [VHS](https://github.com/charmbracelet/vhs), then create a `demo.tape` file:

```
# demo.tape
Output demo.gif
Set Width 1200
Set Height 800
Set FontSize 14
Set Theme "Dracula"
Set Padding 20

Type "python examples/05_deep_research_live/run.py"
Enter
Sleep 25s
```

Run it:

```bash
vhs demo.tape
```

### Option B: asciinema + agg (real recording)

```bash
# Record the session
asciinema rec felix-demo.cast -c "python examples/05_deep_research_live/run.py"

# Convert to GIF (install agg: https://github.com/asciinema/agg)
agg felix-demo.cast felix-demo.gif --cols 120 --rows 40
```

### Option C: ttygif (lightweight)

```bash
ttyrec -e "python examples/05_deep_research_live/run.py"
ttygif ttyrecord -f felix-demo.gif
```

### Tips for best results

- Use a dark terminal theme (Dracula, One Dark, or Catppuccin work well)
- Set terminal to at least 120 columns x 40 rows
- Use a monospace font with good Unicode support (JetBrains Mono, Fira Code)
- The full run with animations takes ~20 seconds — perfect length for Twitter/X

## Files

| File | Purpose |
|---|---|
| `run.py` | Main demo script — orchestration loop with visualisation hooks |
| `helix_visualizer.py` | ASCII helix renderer, phase transitions, intro/outro animations |
| `_mock_research.py` | Phase-aware mock provider with realistic research responses |
