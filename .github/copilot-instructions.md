# Copilot Instructions

## Rules

1. **Do not generate unnecessary markdown files** - Only create markdown documentation files (README.md, guides, etc.) when explicitly requested by the user. Do not automatically create summary or documentation files after completing tasks unless asked.

2. **Examine existing patterns first** - When working with structured concepts (like `.github/skills/` files), always check existing examples in the codebase before implementing. This clarifies format, purpose, and intent. A single existing file is worth more than assumptions.

3. **Distinguish between instruction files and output files** - `.github/skills/SKILL.md` files are procedural guides for agents to follow, not documentation listing outputs. They should contain instructions and validation steps, not sample results or command references.

## Project Setup

This project uses **`uv`** for Python dependency management and running Python scripts. 

- **Running scripts**: Use `uv run <script>` from the project root to execute Python files
- **Dependencies**: Managed via `pyproject.toml` at the repository root
- **Exercise structure**: Each exercise folder (Exercise1.1, Exercise1.2, etc.) contains:
  - `src/` - Source code and Python scripts
  - `requirements.txt` - Exercise-specific dependencies (referenced in pyproject.toml)
  - `configs/` - Configuration files
  - `models/` - Model checkpoints/weights
  - `outputs/` - Generated outputs and results
  - `week_report/` - Weekly reports and documentation
