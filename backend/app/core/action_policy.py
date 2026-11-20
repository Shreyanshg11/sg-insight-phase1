"""
Single authoritative definition of which recommendation actions can never be
executed, and which cannot be executed *through Copilot*.

- REVIEW_ONLY_ACTIONS: advice only. No endpoint may execute these, whatever
  the caller sends (flag_for_review is what outlier recommendations carry).
- COPILOT_NON_EXECUTABLE_ACTIONS: actions that exist for the manual
  recommendation cards (where the user chooses the exact values) but must not
  be proposed by an LLM. fuzzy_merge needs user-chosen values/canonical and is
  only ever a review-only suggestion in Copilot.
"""

REVIEW_ONLY_ACTIONS = frozenset({"flag_for_review"})
COPILOT_NON_EXECUTABLE_ACTIONS = frozenset({"fuzzy_merge"})


def normalize_action(action) -> str:
    return action.strip().lower() if isinstance(action, str) else ""


def is_review_only(action) -> bool:
    return normalize_action(action) in REVIEW_ONLY_ACTIONS


def is_copilot_blocked(action) -> bool:
    return normalize_action(action) in COPILOT_NON_EXECUTABLE_ACTIONS


def is_copilot_executable(action) -> bool:
    return bool(normalize_action(action)) and not is_review_only(action) and not is_copilot_blocked(action)
