---
name: gen-src-readme
description: Generate and update src/README.md files for each exercise with commands to run Python entry point scripts using uv run from the repo root.
---

## Generate Source README Files

Guidelines for generating and updating the `src/README.md` file for each exercise.

### Purpose

The `src/README.md` file in each `ExerciseX.Y/src/` directory should contain instructions and commands to run the Python scripts in that exercise using `uv run` from the repository root.

### Generation Process

**1. Identify Runnable Scripts**

For each exercise directory, identify which Python scripts have `if __name__ == "__main__":` blocks (entry points).

These are the scripts that should be documented with run commands.

**2. Extract Script Information**

For each runnable script:
- Identify the script name and path
- Extract the `main()` function signature to get available parameters
- Document all parameter names, types, and default values
- Document what the script does (from docstrings or comments)

**3. Format Commands**

All commands must follow the format:
```bash
uv run python ExerciseX.Y/src/script_name.py [options]
```

Where:
- Commands are executed from the repository root
- Default run (no arguments) comes first
- Then show optional parameter examples
- Use `--parameter=value` format

**4. Update Strategy**

When updating an existing `src/README.md`:

- **Keep valid commands**: If a script and its parameters still exist, keep the command documentation
- **Update commands**: If script parameters changed, update the documented commands
- **Delete commands**: If a script no longer exists or the `if __name__` block was removed, delete that section
- **Add new commands**: If new runnable scripts appear, add them following the same format

**5. File Structure**

```markdown
# [Exercise Name] - Running the Code

## [Script Name]

Brief description of what the script does.

### Default run
\`\`\`bash
uv run python ExerciseX.Y/src/script_name.py
\`\`\`

### With custom parameters
\`\`\`bash
uv run python ExerciseX.Y/src/script_name.py --param1=value --param2=value
\`\`\`

**Available Parameters**:
- `--param1`: Description (default: value)
- `--param2`: Description (default: value)

---

## [Next Script Name]
...
```

### Validation

Before finalizing `src/README.md`:
1. Verify each command reflects actual script parameters
2. Test that all documented commands work with `uv run`
3. Remove any commands for scripts that no longer have entry points
4. Keep only the most practical example configurations
