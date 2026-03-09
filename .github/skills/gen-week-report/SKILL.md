---
name: gen-week-report
description: Generate and update week_report/README.md and a 1-page .tex file for each exercise by reading the corresponding PDF, analyzing the code, and compiling a concise report.
---

## Generate Weekly Report Files

Guidelines for generating and updating the `week_report/README.md` and a 1-page `.tex` file for each exercise.

### Purpose

The `week_report/README.md` in each `ExerciseX.Y/week_report/` directory should contain a concise summary and analysis of the exercise, including:
- Key concepts and methods used
- Results and findings
- Insights from the code
- Answers to exercise questions (if present in the PDF)

A corresponding 1-page `.tex` file should be generated, suitable for academic submission, summarizing the same content in LaTeX format.

### Generation Process

**1. Read Exercise PDF**
- Locate the PDF for the exercise (e.g., `ExerciseX.Y/ExerciseX.Y.pdf`)
- Extract questions, instructions, and relevant context

**2. Analyze Code**
- Review all scripts in the exercise's `src/` directory
- Identify main algorithms, models, and workflow
- Summarize implementation details and results

**3. Compile Report**
- Answer exercise questions from the PDF
- Summarize code logic, results, and insights
- Highlight any challenges or unique solutions

**4. Format Files**
- `week_report/README.md`: Markdown summary, organized by question and topic
- `.tex` file: 1-page academic summary, formatted for clarity and brevity

**5. Update Strategy**
- If the PDF or code changes, regenerate both files
- Keep only current exercise content; remove outdated sections
- Ensure the `.tex` file is exactly 1 page (use \newpage or adjust content as needed)

### File Structure

```markdown
# [Exercise Name] - Weekly Report

## Summary
Brief overview of the exercise and its goals.

## Key Questions & Answers
- Q1: ...
  - A1: ...
- Q2: ...
  - A2: ...

## Code Insights
- Main algorithms and models
- Results and findings
- Challenges and solutions
```

```latex
% [Exercise Name] - Weekly Report
\documentclass[12pt]{article}
\usepackage{amsmath,graphicx}
\begin{document}
\section*{Summary}
% Brief summary here

\section*{Key Questions & Answers}
% Q&A here

\section*{Code Insights}
% Algorithms, results, challenges

\end{document}
```

### Validation

Before finalizing `week_report/README.md` and `.tex`:
1. Verify all questions from the PDF are answered
2. Ensure code analysis is accurate and concise
3. Confirm the `.tex` file compiles to exactly 1 page
4. Remove any outdated or irrelevant content
