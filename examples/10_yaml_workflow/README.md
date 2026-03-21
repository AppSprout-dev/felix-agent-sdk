# YAML Workflow Example

Run a Felix workflow from a YAML config file using the CLI.

## Prerequisites

```bash
pip install felix-agent-sdk[openai]
export OPENAI_API_KEY=sk-...
```

## Usage

```bash
# From the repo root
felix run examples/10_yaml_workflow/felix.yaml

# With verbose logging
felix run examples/10_yaml_workflow/felix.yaml --verbose

# Override provider
felix run examples/10_yaml_workflow/felix.yaml --provider anthropic
```

## Config Reference

See `felix.yaml` in this directory for all available fields.
