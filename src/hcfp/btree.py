"""Minimal B*-Tree parsing and deterministic contour packing."""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


def contact_aware_vertical_orders(
    base_order: Tensor,
    boundary_bits: Tensor,
    group_membership: Tensor,
) -> tuple[tuple[str, Tensor], ...]:
    """Return a few deterministic y orders that expose boundary/group contacts."""

    order = torch.as_tensor(base_order, dtype=torch.long, device="cpu").reshape(-1)
    n = int(order.numel())
    if sorted(order.tolist()) != list(range(n)):
        raise ValueError("base_order must be a permutation of block indices")
    bits = torch.as_tensor(boundary_bits, dtype=torch.bool, device="cpu")
    groups = torch.as_tensor(group_membership, dtype=torch.bool, device="cpu")
    if bits.shape != (n, 4) or groups.ndim != 2 or groups.shape[1] != n:
        raise ValueError("constraint tensors do not match base_order")

    rank = torch.empty(n, dtype=torch.long)
    rank[order] = torch.arange(n)
    group_id = torch.arange(n, dtype=torch.long) + int(groups.shape[0])
    group_rank = torch.arange(n, dtype=torch.float64)
    for index, row in enumerate(groups):
        members = torch.nonzero(row, as_tuple=False).reshape(-1)
        if members.numel() <= 1:
            continue
        group_id[members] = index
        group_rank[members] = float(rank[members].double().mean())

    def boundary_band(node: int) -> int:
        if bool(bits[node, 3]) and not bool(bits[node, 2]):
            return 0
        if bool(bits[node, 2]) and not bool(bits[node, 3]):
            return 2
        return 1

    nodes = list(range(n))
    variants = (
        (
            "boundary_band",
            sorted(nodes, key=lambda node: (boundary_band(node), int(rank[node]))),
        ),
        (
            "group_cluster",
            sorted(
                nodes,
                key=lambda node: (
                    float(group_rank[node]),
                    int(group_id[node]),
                    int(rank[node]),
                ),
            ),
        ),
        (
            "boundary_group",
            sorted(
                nodes,
                key=lambda node: (
                    boundary_band(node),
                    float(group_rank[node]),
                    int(group_id[node]),
                    int(rank[node]),
                ),
            ),
        ),
    )
    unique: list[tuple[str, Tensor]] = [("base", order.clone())]
    seen = {tuple(order.tolist())}
    for name, values in variants:
        key = tuple(values)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, torch.tensor(values, dtype=torch.long)))
    return tuple(unique)


def local_tree_variants(
    tree: "BStarTree",
    boundary_bits: Tensor,
    group_membership: Tensor,
    *,
    limit: int,
) -> tuple[tuple[str, "BStarTree"], ...]:
    """Generate bounded valid sibling flips and group-leaf reinsertion moves."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return ()
    n = tree.block_count
    bits = torch.as_tensor(boundary_bits, dtype=torch.bool, device="cpu")
    groups = torch.as_tensor(group_membership, dtype=torch.bool, device="cpu")
    if bits.shape != (n, 4) or groups.ndim != 2 or groups.shape[1] != n:
        raise ValueError("constraint tensors do not match B*-Tree")
    candidates: list[tuple[tuple[int, ...], str, BStarTree]] = []

    for parent, (left_child, right_child) in enumerate(
        zip(tree.left, tree.right, strict=True)
    ):
        if left_child < 0 or right_child < 0:
            continue
        left = list(tree.left)
        right = list(tree.right)
        left[parent], right[parent] = right_child, left_child
        priority = -int(bits[left_child].any()) - int(bits[right_child].any())
        candidates.append(
            (
                (0, priority, parent),
                f"sibling_flip:{parent}",
                _tree_from_children(left, right),
            )
        )

    parents = [-1] * n
    parent_sides = [-1] * n
    for parent, children in enumerate(zip(tree.left, tree.right, strict=True)):
        for side, child in enumerate(children):
            if child >= 0:
                parents[child] = parent
                parent_sides[child] = side
    for group_index, row in enumerate(groups):
        members = torch.nonzero(row, as_tuple=False).reshape(-1).tolist()
        if len(members) <= 1:
            continue
        for leaf in members:
            if leaf == tree.root or tree.left[leaf] >= 0 or tree.right[leaf] >= 0:
                continue
            for target in members:
                if target == leaf or target == parents[leaf]:
                    continue
                for side in (0, 1):
                    if (tree.left if side == 0 else tree.right)[target] >= 0:
                        continue
                    left = list(tree.left)
                    right = list(tree.right)
                    old_parent = parents[leaf]
                    (left if parent_sides[leaf] == 0 else right)[old_parent] = -1
                    (left if side == 0 else right)[target] = leaf
                    candidates.append(
                        (
                            (1, group_index, leaf, target, side),
                            f"group_reinsert:{group_index}:{leaf}:{target}:{side}",
                            _tree_from_children(left, right),
                        )
                    )

    selected: list[tuple[str, BStarTree]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for _, name, candidate in sorted(candidates, key=lambda item: item[0]):
        key = (candidate.left, candidate.right)
        if key in seen:
            continue
        seen.add(key)
        selected.append((name, candidate))
        if len(selected) == limit:
            break
    return tuple(selected)


def subtree_move_variants(
    tree: "BStarTree",
    *,
    limit: int,
) -> tuple[tuple[str, "BStarTree"], ...]:
    """Generate bounded recursive transpose and subtree reinsert variants.

    A ``subtree_transpose`` recursively exchanges left and right children for
    every node below the selected root.  A ``subtree_reinsert`` detaches one
    subtree and attaches it to an empty child slot outside that subtree.  Both
    operations preserve the B*-Tree node set and are rebuilt through
    :meth:`BStarTree.from_edges`, so malformed or cyclic variants cannot leak
    into callers.
    """

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0 or tree.block_count <= 1:
        return ()

    n = tree.block_count
    children = tuple(zip(tree.left, tree.right, strict=True))
    parents = [-1] * n
    parent_sides = [-1] * n
    for parent, (left_child, right_child) in enumerate(children):
        for side, child in enumerate((left_child, right_child)):
            if child >= 0:
                parents[child] = parent
                parent_sides[child] = side

    descendants: list[set[int]] = [set() for _ in range(n)]

    def collect(node: int) -> set[int]:
        members = descendants[node]
        if members:
            return members
        for child in children[node]:
            if child >= 0:
                members.add(child)
                members.update(collect(child))
        return members

    for node in range(n):
        collect(node)

    candidates: list[tuple[tuple[int, ...], str, BStarTree]] = []

    # Keep transpose variants before reinsertions, and enumerate roots by
    # block index for deterministic output independent of hash/set ordering.
    for node in range(n):
        if not descendants[node]:
            continue
        left = list(tree.left)
        right = list(tree.right)
        stack = [node]
        while stack:
            current = stack.pop()
            left[current], right[current] = right[current], left[current]
            if left[current] >= 0:
                stack.append(left[current])
            if right[current] >= 0:
                stack.append(right[current])
        candidates.append(
            ((0, node), f"subtree_transpose:{node}", _tree_from_children(left, right))
        )

    # A target slot is empty in the post-detach tree either when it was
    # already empty, or when it is the source slot just vacated.
    for node in range(n):
        if node == tree.root:
            continue
        old_parent = parents[node]
        old_side = parent_sides[node]
        if old_parent < 0:
            continue
        blocked = descendants[node] | {node}
        for target in range(n):
            if target in blocked:
                continue
            for side in (0, 1):
                current_child = children[target][side]
                if current_child >= 0 and not (
                    target == old_parent and side == old_side
                ):
                    continue
                if target == old_parent and side == old_side:
                    continue
                left = list(tree.left)
                right = list(tree.right)
                source_branch = left if old_side == 0 else right
                source_branch[old_parent] = -1
                target_branch = left if side == 0 else right
                target_branch[target] = node
                candidates.append(
                    (
                        (1, node, target, side),
                        f"subtree_reinsert:{node}:{target}:{side}",
                        _tree_from_children(left, right),
                    )
                )

    selected: list[tuple[str, BStarTree]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for _, name, candidate in sorted(candidates, key=lambda item: item[0]):
        key = (candidate.left, candidate.right)
        if key in seen:
            continue
        seen.add(key)
        selected.append((name, candidate))
        if len(selected) == limit:
            break
    return tuple(selected)


def _tree_from_children(left: list[int], right: list[int]) -> "BStarTree":
    edges = [
        (parent, child, side)
        for parent, children in enumerate(zip(left, right, strict=True))
        for side, child in enumerate(children)
        if child >= 0
    ]
    return BStarTree.from_edges(torch.tensor(edges, dtype=torch.long), len(left))


@dataclass(frozen=True)
class BStarTree:
    root: int
    left: tuple[int, ...]
    right: tuple[int, ...]

    @classmethod
    def from_edges(cls, edges: Tensor, block_count: int) -> "BStarTree":
        values = torch.as_tensor(edges, dtype=torch.float64, device="cpu")
        if values.shape != (block_count - 1, 3) or not bool(
            torch.isfinite(values).all()
        ):
            raise ValueError("B*-Tree edges must have shape [N-1,3] and be finite")
        rounded = values.round()
        if not torch.equal(values, rounded):
            raise ValueError("B*-Tree edges must be integer-valued")
        rows = rounded.to(dtype=torch.long)
        if bool((rows[:, :2] < 0).any()) or bool((rows[:, :2] >= block_count).any()):
            raise ValueError("B*-Tree node index is out of range")
        if not bool(((rows[:, 2] == 0) | (rows[:, 2] == 1)).all()):
            raise ValueError("B*-Tree side must be 0 or 1")
        left = [-1] * block_count
        right = [-1] * block_count
        parents = [-1] * block_count
        for parent, child, side in rows.tolist():
            branch = left if side == 0 else right
            if parent == child or parents[child] != -1 or branch[parent] != -1:
                raise ValueError("B*-Tree contains duplicate parent/branch assignments")
            branch[parent] = child
            parents[child] = parent
        roots = [index for index, parent in enumerate(parents) if parent == -1]
        if len(roots) != 1:
            raise ValueError("B*-Tree must have exactly one root")
        root = roots[0]
        visited: set[int] = set()
        active: set[int] = set()

        def visit(node: int) -> None:
            if node in active:
                raise ValueError("B*-Tree contains a cycle")
            if node in visited:
                return
            active.add(node)
            for child in (left[node], right[node]):
                if child >= 0:
                    visit(child)
            active.remove(node)
            visited.add(node)

        visit(root)
        if len(visited) != block_count:
            raise ValueError("B*-Tree is disconnected")
        return cls(root, tuple(left), tuple(right))

    @property
    def block_count(self) -> int:
        return len(self.left)

    def edges(self) -> Tensor:
        return torch.tensor(
            [
                (parent, child, side)
                for parent, children in enumerate(
                    zip(self.left, self.right, strict=True)
                )
                for side, child in enumerate(children)
                if child >= 0
            ],
            dtype=torch.long,
        )

    def pack(self, dimensions: Tensor) -> Tensor:
        dims = torch.as_tensor(dimensions, dtype=torch.float64, device="cpu")
        if dims.shape != (self.block_count, 2) or not bool(torch.isfinite(dims).all()):
            raise ValueError("dimensions must have shape [N,2] and be finite")
        if bool((dims <= 0.0).any()):
            raise ValueError("dimensions must be positive")
        boxes = torch.zeros((self.block_count, 4), dtype=torch.float64)
        contour: list[tuple[float, float, float]] = []

        def contour_y(x0: float, x1: float) -> float:
            return max(
                (height for start, end, height in contour if x0 < end and x1 > start),
                default=0.0,
            )

        def update(x0: float, x1: float, height: float) -> None:
            retained: list[tuple[float, float, float]] = []
            for start, end, old_height in contour:
                if end <= x0 or start >= x1:
                    retained.append((start, end, old_height))
                else:
                    if start < x0:
                        retained.append((start, x0, old_height))
                    if end > x1:
                        retained.append((x1, end, old_height))
            retained.append((x0, x1, height))
            retained.sort()
            contour[:] = retained

        def place(node: int, x: float) -> None:
            width, height = (float(value) for value in dims[node])
            y = contour_y(x, x + width)
            boxes[node] = torch.tensor((x, y, width, height), dtype=torch.float64)
            update(x, x + width, y + height)
            if self.left[node] >= 0:
                place(self.left[node], x + width)
            if self.right[node] >= 0:
                place(self.right[node], x)

        place(self.root, 0.0)
        return boxes

    def pack_with_preplaced(
        self,
        dimensions: Tensor,
        preplaced_mask: Tensor,
        preplaced_xywh: Tensor,
        *,
        origin: tuple[float, float] = (0.0, 0.0),
        gutter: float = 2.0e-6,
    ) -> Tensor:
        """Pack movable nodes above an anchor-aware contour, preserving preplaced boxes."""

        dims = torch.as_tensor(dimensions, dtype=torch.float64, device="cpu")
        mask = torch.as_tensor(preplaced_mask, dtype=torch.bool, device="cpu").reshape(
            -1
        )
        targets = torch.as_tensor(preplaced_xywh, dtype=torch.float64, device="cpu")
        if dims.shape != (self.block_count, 2) or mask.numel() != self.block_count:
            raise ValueError("anchor-aware B*-Tree inputs do not match block count")
        if targets.shape != (self.block_count, 4):
            raise ValueError("preplaced_xywh must have shape [N,4]")
        boxes = torch.zeros((self.block_count, 4), dtype=torch.float64)
        placed: list[tuple[float, float, float, float]] = []
        for index in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
            box = tuple(float(value) for value in targets[index])
            boxes[index] = targets[index]
            placed.append(box)

        def place(node: int, proposed_x: float) -> None:
            width, height = (float(value) for value in dims[node])
            if bool(mask[node]):
                x, y = (float(boxes[node, 0]), float(boxes[node, 1]))
                width, height = (float(boxes[node, 2]), float(boxes[node, 3]))
            else:
                x = proposed_x
                y = float(origin[1])
                for left, bottom, obstacle_width, obstacle_height in placed:
                    if x < left + obstacle_width and x + width > left:
                        y = max(y, bottom + obstacle_height + gutter)
                boxes[node] = torch.tensor((x, y, width, height), dtype=torch.float64)
                placed.append((x, y, width, height))
            if self.left[node] >= 0:
                place(self.left[node], x + width + gutter)
            if self.right[node] >= 0:
                place(self.right[node], x)

        place(self.root, float(origin[0]))
        return boxes

    def pack_x_compacted(
        self,
        dimensions: Tensor,
        vertical_order: Tensor,
        preplaced_mask: Tensor,
        preplaced_xywh: Tensor,
        *,
        origin: tuple[float, float] = (0.0, 0.0),
        gutter: float = 2.0e-6,
    ) -> Tensor:
        """Keep B*-Tree x relations and compact y by a supplied runtime order."""

        dims = torch.as_tensor(dimensions, dtype=torch.float64, device="cpu")
        order = torch.as_tensor(vertical_order, dtype=torch.long, device="cpu").reshape(
            -1
        )
        mask = torch.as_tensor(preplaced_mask, dtype=torch.bool, device="cpu").reshape(
            -1
        )
        targets = torch.as_tensor(preplaced_xywh, dtype=torch.float64, device="cpu")
        if dims.shape != (self.block_count, 2) or mask.numel() != self.block_count:
            raise ValueError("x-compacted B*-Tree inputs do not match block count")
        if sorted(order.tolist()) != list(range(self.block_count)):
            raise ValueError("vertical_order must be a permutation of block indices")
        boxes = torch.zeros((self.block_count, 4), dtype=torch.float64)

        def assign_x(node: int, proposed_x: float) -> None:
            width, height = (float(value) for value in dims[node])
            if bool(mask[node]):
                boxes[node] = targets[node]
                x, width = float(targets[node, 0]), float(targets[node, 2])
            else:
                x = proposed_x
                boxes[node] = torch.tensor((x, float(origin[1]), width, height))
            if self.left[node] >= 0:
                assign_x(self.left[node], x + width + gutter)
            if self.right[node] >= 0:
                assign_x(self.right[node], x)

        assign_x(self.root, float(origin[0]))
        placed = [
            tuple(float(value) for value in targets[index])
            for index in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
        ]
        for node in order.tolist():
            if bool(mask[node]):
                continue
            x, _, width, height = (float(value) for value in boxes[node])
            y = float(origin[1])
            while True:
                blockers = [
                    obstacle
                    for obstacle in placed
                    if x < obstacle[0] + obstacle[2]
                    and x + width > obstacle[0]
                    and y < obstacle[1] + obstacle[3] + gutter
                    and y + height > obstacle[1]
                ]
                if not blockers:
                    break
                y = max(obstacle[1] + obstacle[3] + gutter for obstacle in blockers)
            boxes[node, 1] = y
            placed.append((x, y, width, height))
        return boxes

    def pack_y_compacted(
        self,
        dimensions: Tensor,
        horizontal_order: Tensor,
        preplaced_mask: Tensor,
        preplaced_xywh: Tensor,
        *,
        origin: tuple[float, float] = (0.0, 0.0),
        gutter: float = 2.0e-6,
    ) -> Tensor:
        """Keep B*-Tree y relations and compact x by a runtime order."""

        dims = torch.as_tensor(dimensions, dtype=torch.float64, device="cpu")
        targets = torch.as_tensor(preplaced_xywh, dtype=torch.float64, device="cpu")
        if dims.shape != (self.block_count, 2):
            raise ValueError("y-compacted B*-Tree inputs do not match block count")
        if targets.shape != (self.block_count, 4):
            raise ValueError("preplaced_xywh must have shape [N,4]")

        # Transpose the geometry and swap branches so the original right-child
        # (y) relation becomes the x relation consumed by pack_x_compacted.
        transposed_tree = BStarTree(self.root, self.right, self.left)
        transposed = transposed_tree.pack_x_compacted(
            dims[:, [1, 0]],
            horizontal_order,
            preplaced_mask,
            targets[:, [1, 0, 3, 2]],
            origin=(origin[1], origin[0]),
            gutter=gutter,
        )
        return transposed[:, [1, 0, 3, 2]]


def btree_dimension_variants(
    dimensions: Tensor,
    *,
    fixed_mask: Tensor | None = None,
    preplaced_mask: Tensor | None = None,
    mib_membership: Tensor | None = None,
    weighted_degree: Tensor | None = None,
    high_degree_threshold: float | None = None,
    low_degree_threshold: float | None = None,
    areas: Tensor | None = None,
) -> dict[str, Tensor]:
    """Build small aspect-ratio challenger families for B*-Tree packing.

    The returned tensors use ``[width, height]`` rows and preserve the input
    order.  ``unlimited`` is an exact clone of ``dimensions``.  ``ar64`` and
    ``ar32`` cap the aspect ratio of movable, non-MIB members while retaining
    each row's area.  ``net_aware`` uses weighted degree to apply the tighter
    32 cap to high-degree blocks, the 64 cap to ordinary blocks, and leaves
    low-degree blocks unchanged.

    Fixed, preplaced, and every member of a MIB group are protected: their
    dimensions are copied verbatim into every returned family.  Thresholds
    for ``net_aware`` default to the 75th/25th degree quantiles over eligible
    blocks and may be supplied explicitly for deterministic sweeps.
    """

    source = torch.as_tensor(dimensions)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("dimensions must have shape [N,2]")
    if not source.is_floating_point() and not source.is_complex():
        source = source.to(dtype=torch.float64)
    else:
        source = source.to(dtype=torch.float64)
    if not bool(torch.isfinite(source).all()) or bool((source <= 0.0).any()):
        raise ValueError("dimensions must be finite and positive")
    n = int(source.shape[0])

    def mask_or_zero(value: Tensor | None, name: str) -> Tensor:
        if value is None:
            return torch.zeros(n, dtype=torch.bool, device=source.device)
        mask = torch.as_tensor(value, dtype=torch.bool, device=source.device).reshape(
            -1
        )
        if mask.numel() != n:
            raise ValueError(f"{name} must have shape [N]")
        return mask

    protected = mask_or_zero(fixed_mask, "fixed_mask") | mask_or_zero(
        preplaced_mask, "preplaced_mask"
    )
    if mib_membership is not None:
        mib = torch.as_tensor(mib_membership, dtype=torch.bool, device=source.device)
        if mib.ndim == 1:
            if mib.numel() != n:
                raise ValueError("mib_membership must have shape [N] or [G,N]")
            protected |= mib.reshape(-1)
        elif mib.ndim == 2 and mib.shape[1] == n:
            protected |= mib.any(dim=0)
        else:
            raise ValueError("mib_membership must have shape [N] or [G,N]")

    if areas is None:
        target_area = source[:, 0] * source[:, 1]
    else:
        target_area = torch.as_tensor(
            areas, dtype=torch.float64, device=source.device
        ).reshape(-1)
        if target_area.numel() != n:
            raise ValueError("areas must have shape [N]")
        if not bool(torch.isfinite(target_area).all()) or bool(
            (target_area <= 0.0).any()
        ):
            raise ValueError("areas must be finite and positive")
    eligible = ~protected

    def capped(cap: float, selected: Tensor) -> Tensor:
        if cap <= 0.0:
            raise ValueError("aspect-ratio cap must be positive")
        result = source.clone()
        width, height = source[:, 0], source[:, 1]
        ratio = torch.maximum(width / height, height / width)
        change = selected & (ratio > cap)
        wide = width >= height
        capped_width = torch.where(
            wide,
            torch.sqrt(target_area * cap),
            torch.sqrt(target_area / cap),
        )
        # Derive the second side from the target area instead of a second
        # square root so the area invariant is preserved as tightly as the
        # working dtype allows.
        capped_height = target_area / capped_width
        result[:, 0] = torch.where(change, capped_width, width)
        result[:, 1] = torch.where(change, capped_height, height)
        return result

    result: dict[str, Tensor] = {
        "unlimited": source.clone(),
        "ar64": capped(64.0, eligible),
        "ar32": capped(32.0, eligible),
    }

    if weighted_degree is None:
        result["net_aware"] = result["ar64"].clone()
    else:
        degree = torch.as_tensor(
            weighted_degree, dtype=torch.float64, device=source.device
        ).reshape(-1)
        if degree.numel() != n:
            raise ValueError("weighted_degree must have shape [N]")
        if not bool(torch.isfinite(degree).all()):
            raise ValueError("weighted_degree must be finite")
        eligible_degree = degree[eligible]
        if eligible_degree.numel() == 0:
            result["net_aware"] = source.clone()
        else:
            high = (
                float(torch.quantile(eligible_degree, 0.75))
                if high_degree_threshold is None
                else float(high_degree_threshold)
            )
            low = (
                float(torch.quantile(eligible_degree, 0.25))
                if low_degree_threshold is None
                else float(low_degree_threshold)
            )
            if high < low:
                raise ValueError(
                    "high_degree_threshold must be >= low_degree_threshold"
                )
            high_mask = eligible & (degree >= high)
            low_mask = eligible & (degree <= low) & ~high_mask
            ordinary_mask = eligible & ~high_mask & ~low_mask
            aware = capped(64.0, ordinary_mask)
            aware[high_mask] = capped(32.0, high_mask)[high_mask]
            result["net_aware"] = aware
    return result


def decode_btree_logits(root_logits: Tensor, edge_logits: Tensor) -> BStarTree:
    """Greedily decode a valid rooted binary tree from root and edge scores."""

    root_scores = (
        torch.as_tensor(root_logits)
        .detach()
        .to(dtype=torch.float64, device="cpu")
        .reshape(-1)
    )
    edges = torch.as_tensor(edge_logits).detach().to(dtype=torch.float64, device="cpu")
    n = int(root_scores.numel())
    if n <= 0 or edges.shape != (n, n, 2):
        raise ValueError("B*-Tree logits must have shapes [N] and [N,N,2]")
    if not bool(torch.isfinite(root_scores).all()) or not bool(
        torch.isfinite(edges).all()
    ):
        raise ValueError("B*-Tree logits must be finite")
    root = int(root_scores.argmax())
    connected = {root}
    remaining = set(range(n)) - connected
    left = [-1] * n
    right = [-1] * n
    rows = []
    while remaining:
        choices = []
        for parent in sorted(connected):
            for side, branch in enumerate((left, right)):
                if branch[parent] >= 0:
                    continue
                for child in sorted(remaining):
                    choices.append(
                        (float(edges[child, parent, side]), -parent, -child, -side)
                    )
        if not choices:
            raise RuntimeError("B*-Tree decoder exhausted branch capacity")
        _, parent_neg, child_neg, side_neg = max(choices)
        parent, child, side = -parent_neg, -child_neg, -side_neg
        (left if side == 0 else right)[parent] = child
        rows.append((parent, child, side))
        connected.add(child)
        remaining.remove(child)
    return BStarTree.from_edges(torch.tensor(rows, dtype=torch.long), n)
