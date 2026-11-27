from __future__ import annotations

from xair.core.models import ActionIntent


class DistributedCoordinator:
    """Single-node conflict resolution for concurrent intents on same target."""

    def __init__(self) -> None:
        self._active_targets: dict[str, str] = {}

    def check_conflict(self, intent: ActionIntent) -> tuple[bool, str]:
        target = intent.payload.target_entity
        if target in self._active_targets:
            return True, f"target_busy:{target}"
        return False, ""

    def acquire(self, intent: ActionIntent) -> None:
        self._active_targets[intent.payload.target_entity] = intent.id

    def release(self, intent: ActionIntent) -> None:
        target = intent.payload.target_entity
        if self._active_targets.get(target) == intent.id:
            del self._active_targets[target]

    def resolve(
        self, intents: list[ActionIntent], winner_policy: str = "priority_then_xr"
    ) -> tuple[ActionIntent | None, list[ActionIntent]]:
        if not intents:
            return None, []
        if winner_policy == "priority_then_xr":
            sorted_intents = sorted(
                intents,
                key=lambda i: (i.priority, 1 if i.source == "xr" else 0),
                reverse=True,
            )
        else:
            sorted_intents = sorted(intents, key=lambda i: i.priority, reverse=True)
        winner = sorted_intents[0]
        losers = sorted_intents[1:]
        return winner, losers
