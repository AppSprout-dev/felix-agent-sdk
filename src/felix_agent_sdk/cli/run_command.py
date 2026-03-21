"""``felix run`` — execute a workflow from a YAML config file."""

from __future__ import annotations

import sys
from contextlib import nullcontext

from felix_agent_sdk.cli.yaml_loader import load_workflow_yaml
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.utils.logging import FelixLogConfig, configure_logging
from felix_agent_sdk.workflows.runner import run_felix_workflow


def run_workflow(
    config_path: str,
    provider_name: str | None,
    verbose: bool,
    visualize: bool = False,
) -> int:
    """Load a YAML config, resolve the provider, and run the workflow.

    Args:
        config_path: Path to the ``felix.yaml`` config file.
        provider_name: Override provider name (``None`` = use config).
        verbose: Enable debug logging.
        visualize: Show live helix visualization during the run.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    if verbose:
        configure_logging(FelixLogConfig(level="DEBUG"))
    else:
        configure_logging(FelixLogConfig(level="WARNING"))

    try:
        config, task, provider_info = load_workflow_yaml(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Resolve provider
    effective_provider = provider_name or provider_info.get("provider", "")
    model = provider_info.get("model", "")

    provider = _resolve_provider(effective_provider, model)
    if provider is None:
        print(
            f"Error: could not create provider '{effective_provider}'. "
            "Check your API key and provider dependencies.",
            file=sys.stderr,
        )
        return 1

    # Optional visualization
    event_bus = None
    viz = None

    if visualize:
        from felix_agent_sdk.events import EventBus
        from felix_agent_sdk.visualization import HelixVisualizer

        event_bus = EventBus()
        helix = config.helix_config.to_geometry()
        viz = HelixVisualizer(helix)
        viz.attach_event_bus(event_bus)

        def _on_round_complete(event):  # type: ignore[no-untyped-def]
            viz.render(tick=event.data.get("round", 0))

        event_bus.subscribe("workflow.round.completed", _on_round_complete)
    else:
        print("Running Felix workflow...")
        print(f"  Provider: {effective_provider or 'auto'}")
        print(f"  Task: {task[:80]}{'...' if len(task) > 80 else ''}")
        print(f"  Team: {len(config.team_composition)} agents, {config.max_rounds} rounds")
        print()

    ctx = viz.live() if viz else nullcontext()
    with ctx:
        try:
            result = run_felix_workflow(config, provider, task, event_bus=event_bus)
        except Exception as e:
            print(f"Error during workflow: {e}", file=sys.stderr)
            return 1
        if viz:
            viz.render()

    print("=== Result ===")
    print(f"Rounds: {result.total_rounds}")
    print(f"Confidence: {result.final_confidence:.3f}")
    print(f"Converged: {result.metadata.get('converged', False)}")
    print(f"Tokens: {result.metadata.get('total_tokens', 0)}")
    print(f"Time: {result.metadata.get('elapsed_seconds', 0)}s")
    print(f"\nSynthesis:\n{result.synthesis}")
    return 0


def _resolve_provider(name: str, model: str) -> BaseProvider | None:
    """Try to create a provider by name. Returns None on failure."""
    if not name or name == "auto":
        try:
            from felix_agent_sdk.providers import auto_detect_provider

            return auto_detect_provider()
        except Exception:
            return None

    try:
        if name == "openai":
            from felix_agent_sdk.providers import OpenAIProvider

            return OpenAIProvider(model=model or "gpt-4o-mini")
        elif name == "anthropic":
            from felix_agent_sdk.providers import AnthropicProvider

            return AnthropicProvider(model=model or "claude-sonnet-4-5")
        elif name == "local":
            from felix_agent_sdk.providers import LocalProvider

            return LocalProvider(model=model or "default")
    except Exception:
        pass
    return None
