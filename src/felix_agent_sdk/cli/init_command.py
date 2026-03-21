"""``felix init`` — scaffold a new Felix project."""

from __future__ import annotations

import sys
from pathlib import Path

_TEMPLATES = {
    "research": {
        "team": [
            {"type": "research"},
            {"type": "research"},
            {"type": "analysis"},
            {"type": "critic"},
        ],
        "helix": "research_heavy",
        "max_rounds": 4,
        "confidence_threshold": 0.75,
    },
    "analysis": {
        "team": [
            {"type": "research"},
            {"type": "analysis"},
            {"type": "analysis"},
            {"type": "critic"},
        ],
        "helix": "fast_convergence",
        "max_rounds": 3,
        "confidence_threshold": 0.85,
    },
    "review": {
        "team": [
            {"type": "research"},
            {"type": "analysis"},
            {"type": "critic"},
            {"type": "critic"},
        ],
        "helix": "default",
        "max_rounds": 3,
        "confidence_threshold": 0.80,
    },
}


def run_init(name: str, template: str) -> int:
    """Scaffold a new Felix project directory.

    Creates:
        name/
            felix.yaml
            main.py
            requirements.txt
            .env.example

    Returns:
        Exit code (0 = success, 1 = error).
    """
    target = Path(name)
    if target.exists():
        print(f"Error: directory '{name}' already exists.", file=sys.stderr)
        return 1

    if template not in _TEMPLATES:
        valid = ", ".join(_TEMPLATES)
        print(f"Error: unknown template '{template}'. Valid: {valid}", file=sys.stderr)
        return 1

    target.mkdir(parents=True)

    # felix.yaml
    cfg = _TEMPLATES[template]
    team_lines = "\n".join(f'  - type: {e["type"]}' for e in cfg["team"])
    yaml_content = f"""\
# Felix workflow configuration
# Run with: felix run felix.yaml

provider: openai
model: gpt-4o-mini

helix: {cfg["helix"]}
max_rounds: {cfg["max_rounds"]}
confidence_threshold: {cfg["confidence_threshold"]}

team:
{team_lines}

task: "Your research question or task description here"
"""
    (target / "felix.yaml").write_text(yaml_content)

    # main.py
    main_py = """\
#!/usr/bin/env python3
\"\"\"Run a Felix workflow programmatically.\"\"\"

from felix_agent_sdk import run_felix_workflow, WorkflowConfig
from felix_agent_sdk.providers import auto_detect_provider

provider = auto_detect_provider()
config = WorkflowConfig(max_rounds=3)

result = run_felix_workflow(config, provider, "Your task here")
print(result.synthesis)
"""
    (target / "main.py").write_text(main_py)

    # requirements.txt
    (target / "requirements.txt").write_text("felix-agent-sdk[openai]\n")

    # .env.example
    (target / ".env.example").write_text(
        "# Set your LLM provider API key\n"
        "# OPENAI_API_KEY=sk-...\n"
        "# ANTHROPIC_API_KEY=sk-ant-...\n"
        "FELIX_PROVIDER=openai\n"
        "FELIX_MODEL=gpt-4o-mini\n"
    )

    print(f"Created Felix project: {target}/")
    print(f"  Template: {template}")
    print(f"  Config:   {target}/felix.yaml")
    print(f"  Entry:    {target}/main.py")
    print()
    print("Next steps:")
    print(f"  cd {name}")
    print("  pip install -r requirements.txt")
    print("  # Copy .env.example to .env and add your API key")
    print("  felix run felix.yaml")
    return 0
