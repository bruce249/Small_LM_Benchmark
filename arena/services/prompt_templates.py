"""Prompt template library for different evaluation task types."""

from __future__ import annotations

from arena.schemas import TaskType

# ── Templates ─────────────────────────────────────────────────────────────────
# Each template receives an ``{input}`` placeholder that will be filled
# with the dataset sample.

PROMPT_TEMPLATES: dict[TaskType, str] = {
    TaskType.SUMMARIZATION: (
        "You are an expert summarizer. Provide a concise and accurate summary "
        "of the following text.\n\n"
        "Text:\n{input}\n\n"
        "Summary:"
    ),
    TaskType.QA: (
        "Answer the following question accurately and concisely.\n\n"
        "{input}\n\n"
        "Answer:"
    ),
    TaskType.CODING: (
        "Complete the following Python function. Return ONLY the function body "
        "without any explanation.\n\n"
        "{input}"
    ),
    TaskType.REASONING: (
        "Solve the following problem step by step. Show your work and provide "
        "the final numerical answer.\n\n"
        "Problem:\n{input}\n\n"
        "Solution:"
    ),
}


def build_prompt(task_type: TaskType, input_text: str) -> str:
    """Render a prompt for the given task type and input."""
    template = PROMPT_TEMPLATES.get(task_type, PROMPT_TEMPLATES[TaskType.SUMMARIZATION])
    return template.format(input=input_text)
