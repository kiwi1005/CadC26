from __future__ import annotations

from puzzleplace.actions.schema import ActionPrimitive, TypedAction


def generate_negative_actions(actions: list[TypedAction]) -> list[TypedAction]:
    negatives: list[TypedAction] = []
    for action in actions:
        if action.primitive is ActionPrimitive.PLACE_ABSOLUTE and action.x is not None and action.y is not None:
            negatives.append(
                TypedAction(
                    primitive=ActionPrimitive.PLACE_ABSOLUTE,
                    block_index=action.block_index,
                    x=action.x + 1.0,
                    y=action.y + 1.0,
                    w=action.w,
                    h=action.h,
                    metadata={"negative": True, "kind": "shifted_absolute"},
                )
            )
        elif action.primitive is ActionPrimitive.PLACE_RELATIVE and action.target_index is not None:
            negatives.append(
                TypedAction(
                    primitive=ActionPrimitive.PLACE_RELATIVE,
                    block_index=action.block_index,
                    target_index=action.target_index,
                    dx=(action.dx or 0.0) + 1.0,
                    dy=(action.dy or 0.0) + 1.0,
                    w=action.w,
                    h=action.h,
                    metadata={"negative": True, "kind": "shifted_relative"},
                )
            )
    return negatives
