"""Training curriculum pseudocode for HCFP-5090."""
from typing import Iterable
import torch


def train_initializer(model, loader, optimizer):
    for case, gold in loader:
        scene = model.scene(case)
        pop = model.init(scene, k=16, block_mask=case.block_mask)
        loss_best = best_of_k_geometry_loss(pop, gold, case)
        loss_energy = smooth_floorplan_energy(pop, case, gold.metrics)
        loss_div = population_diversity_loss(pop)
        loss = loss_best + 0.2 * loss_energy + 0.05 * loss_div
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def train_recovery(model, loader, optimizer, unroll_steps: int):
    for case, gold in loader:
        corrupted = online_corrupt(gold, case)
        state = corrupted
        total = 0.0
        for t in range(unroll_steps):
            state, diag = model.collective_step(case, state, t)
            total = total + discounted_state_loss(state, gold, case, diag, t)
            if should_teacher_force(t):
                state = mix_with_teacher_state(state, gold)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def self_improvement_round(system, cases: Iterable, replay):
    with torch.no_grad():
        for case in cases:
            trajectories = system.rollout_population(case)
            projected = system.bdp(trajectories)
            exact = system.exact_like_score(projected)
            replay.add(select_elites_failures_and_stagnation(trajectories, projected, exact))
    # Retrain with a mixture of gold corruptions, analytic teachers and replay.
