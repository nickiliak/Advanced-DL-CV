---
description: Generate src/README.md with run commands for an exercise
---

Generate and update the `src/README.md` file for the specified exercise directory.

1. Identify which Python scripts in `ExerciseX.Y/src/` have `if __name__ == "__main__":` blocks
2. Extract each script's purpose, parameters, and default values
3. Generate `ExerciseX.Y/src/README.md` with `uv run python ExerciseX.Y/src/script_name.py` commands
4. Include parameter documentation and example invocations

If no exercise is specified, ask the user which exercise to generate for.
