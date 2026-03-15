---
description: Run a Python script from an exercise using uv run
---

Run the specified script using `uv run` from the repository root.

1. If the user provides a script name or path, run it with: `uv run python <path>`
2. If the user provides just an exercise number (e.g. "2.3"), look in `ExerciseX.Y/src/` for scripts with `if __name__ == "__main__":` blocks and list them, then ask which to run.
3. If no argument is given, check the currently open file in the IDE and run that.

Always run from the repo root. Pass through any extra arguments the user provides.
