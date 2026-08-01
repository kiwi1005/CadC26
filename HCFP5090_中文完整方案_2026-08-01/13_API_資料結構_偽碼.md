# 13　API、資料結構與核心偽碼

## 13.1 主要資料結構

```python
@dataclass
class FloorplanCase:
    n: int
    area: Tensor              # [N], fp32
    b2b_weight: Tensor        # [N,N], fp32
    p2b_edges: Tensor         # [Ep,3]
    pins: Tensor              # [P,2]
    constraints: Tensor       # [N,5]
    target: Tensor            # [N,4]
    block_mask: Tensor        # [N]
    fixed_mask: Tensor        # [N]
    preplaced_mask: Tensor    # [N]
    group_membership: Tensor  # [G,N]
    mib_membership: Tensor    # [M,N]
    boundary_bits: Tensor     # [N,4]
    scale: float
    origin: Tensor            # [2]
```

```python
@dataclass
class PopulationState:
    center: Tensor            # [K,N,2], fp32
    log_aspect: Tensor        # [K,N], fp32
    velocity: Tensor          # [K,N,3], fp32
    latent: Tensor            # [K,N,H], bf16
    region_prob: Tensor       # [K,N,R], bf16/fp32
    latch_edge: Tensor        # fixed padded representation
    energy_history: Tensor    # [K,Hs,C]
    alive_mask: Tensor        # [K]
```

```python
@dataclass
class ProjectionBatch:
    z0: Tensor
    constraint_kind: Tensor
    idx_a: Tensor
    idx_b: Tensor
    coeff_a: Tensor
    coeff_b: Tensor
    rhs: Tensor
    hard_mask: Tensor
    active_mask: Tensor
    direction_code: Tensor
```

## 13.2 Module contracts

```text
SceneEncoder(case_static) -> SceneEmbedding
PopulationInitializer(scene, case, K) -> PopulationState
CollectiveDynamics.step(scene, case, state, t) -> state, diagnostics
EventController(diagnostics, state) -> EventPlan
PopulationManager.apply(state, event_plan) -> state
Ranker(scene, case, state) -> CandidatePredictions
DirectionSelector(scene, case, candidate, active_pairs) -> logits
BDP(project_batch) -> projected, residuals
FastVerifier(case, projected) -> masks/metrics
ExactVerifier(case, candidate) -> exact metrics
SafeFallback(case) -> positions
```

## 13.3 Dynamics step 偽碼

```python
def collective_step(case, scene, state, t):
    dims = exact_shape_projection(case, state.log_aspect)
    pair = rectangle_pair_features(state.center, dims)

    f_net = smooth_net_force(case, state.center)
    f_overlap = rectangle_overlap_force(pair)
    f_boundary = boundary_force(case, state.center, dims)
    f_group, latch_update = contact_latch_force(case, state, dims)
    f_mib = mib_shape_force(case, state.log_aspect)
    f_grid = global_density_field(case, state.center, dims, scene)

    learned = interaction_network(
        scene.block, state.latent, pair,
        analytic_channels=[f_net, f_overlap, f_boundary,
                           f_group, f_mib, f_grid],
        time=t,
    )
    rho, precond, delta_latent = controller(learned, scene, state)
    force = gated_sum(learned, analytic_channels)
    state.velocity = rho * state.velocity + precond * force
    state.center, state.log_aspect = bounded_update(state, t)
    state = local_hard_project(case, state)
    state.latch_edge = hysteresis_update(state.latch_edge, latch_update)
    state.latent += delta_latent
    return state, compute_diagnostics(case, state)
```

## 13.4 ETR 偽碼

```python
def maybe_reconfigure(state, diagnostics, budget):
    trigger = stagnation_detector(diagnostics)
    conflict = build_conflict_graph(diagnostics)
    components = gpu_components(conflict)
    plan = event_value_network(state, diagnostics, components)

    # 固定 shape 執行，無 Python candidate loop
    branched = apply_masked_events(state, plan, budget)
    return population_reservoir_select(branched)
```

## 13.5 BDP 偽碼

```python
def bdp_project(case, candidate, scene, beam=8, outer=3):
    states = make_direction_beam(case, candidate, scene, beam)
    for _ in range(outer):
        states = shape_consensus_project(case, states)
        constraints = build_padded_linear_constraints(case, states)
        states.z, residual = pdhg_fixed_iters(
            states.z, constraints, iters=48
        )
        states = exact_local_finalize(case, states)
        states.active_pairs = rebuild_active_pairs(case, states)
    feasible, metrics = fast_verify(case, states)
    return select_feasible(states, metrics, feasible)
```

## 13.6 Safe fallback 偽碼

```python
def safe_shelf(case):
    pos = exact_hard_shapes_and_preplaced(case)
    gap = 1e-4 * max(case.scale, 1.0)
    x = max_right_of_preplaced(pos) + gap
    y = choose_shelf_y_outside_preplaced_vertical_span(pos)
    for i in movable_blocks_sorted_deterministically(case):
        w, h = valid_shape(case, i)
        pos[i] = (x, y, w, h)
        x += w + gap
    while exact_overlap(pos):
        shift_movable_shelf_right(pos, case, 2 * total_width(pos) + gap)
    return pos
```

## 13.7 Unit tests

- `test_area_parameterization_exact`；
- `test_preplaced_roundtrip`；
- `test_edge_touch_legal`；
- `test_boundary_bitmask_all_codes`；
- `test_group_union_components`；
- `test_mib_compatible_and_incompatible`；
- `test_bdp_each_direction`；
- `test_direction_cycle_recovery`；
- `test_fallback_with_negative_preplaced`；
- `test_permutation_equivariance`；
- `test_d4_transform_inverse`；
- `test_cuda_graph_bucket_replay`；
- `test_official_score_parity`。
