from __future__ import annotations


def measure_intent_preservation(
    before: dict[int, tuple[float, float, float, float]],
    after: dict[int, tuple[float, float, float, float]],
) -> dict[str, float]:
    shared = sorted(set(before) & set(after))
    if not shared:
        return {
            "mean_displacement": 0.0,
            "max_displacement": 0.0,
            "preserved_x_order_fraction": 1.0,
        }

    displacements: list[float] = []
    preserved_pairs = 0
    total_pairs = 0
    for block_index in shared:
        bx, by, _bw, _bh = before[block_index]
        ax, ay, _aw, _ah = after[block_index]
        displacements.append(abs(ax - bx) + abs(ay - by))

    for left in range(len(shared)):
        for right in range(left + 1, len(shared)):
            li = shared[left]
            ri = shared[right]
            total_pairs += 1
            preserved_pairs += int(
                (before[li][0] <= before[ri][0]) == (after[li][0] <= after[ri][0])
            )

    return {
        "mean_displacement": sum(displacements) / max(len(displacements), 1),
        "max_displacement": max(displacements, default=0.0),
        "preserved_x_order_fraction": preserved_pairs / max(total_pairs, 1),
    }
