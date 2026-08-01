"""Standalone HCFP-5090 inference pseudocode."""
import time


def solve(case, system, budget):
    deadline = time.perf_counter() + budget.seconds
    fallback = safe_shelf(case)
    best_exact = fallback
    best_cost = float("inf")

    scene = system.scene_encoder(case)
    state = system.initializer(scene, budget.population, case.block_mask)

    for segment in range(budget.segments):
        state, diagnostics = system.run_dynamics_segment(case, scene, state)
        if system.event_controller.should_trigger(diagnostics):
            plan = system.event_controller(scene, state, diagnostics, budget)
            state = system.population_manager.apply_events(state, plan)
        quick = system.fast_verifier(case, state)
        if quick.any_feasible:
            for candidate in quick.top_feasible(limit=2):
                exact = system.exact_verifier(case, candidate)
                if exact.feasible and exact.cost_proxy < best_cost:
                    best_exact, best_cost = candidate, exact.cost_proxy
        if time.perf_counter() >= deadline:
            return denormalize(best_exact, case)

    ranked = system.ranker(scene, case, state)
    projected = system.bdp(case, state.select(ranked.top_m), scene)
    for candidate in projected.top_candidates():
        exact = system.exact_verifier(case, candidate)
        if exact.feasible and exact.cost_proxy < best_cost:
            best_exact, best_cost = candidate, exact.cost_proxy
        if time.perf_counter() >= deadline:
            break
    return denormalize(best_exact, case)
