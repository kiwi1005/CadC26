"""Batched Disjunctive Projection (BDP) pseudocode."""
import torch
from torch import Tensor


def pdhg(z0: Tensor, constraints, linear_grad: Tensor,
         tau: float, sigma: float, theta: float, iters: int) -> tuple[Tensor, Tensor]:
    """Solve one direction-conditioned convex proximal projection.

    constraints.az(z) and constraints.aty(y) must be fused GPU operations.
    Inequalities use A z <= b.
    """
    z = z0.clone()
    z_prev = z.clone()
    y = torch.zeros_like(constraints.rhs)
    mass = constraints.diag_mass
    for _ in range(iters):
        z_bar = z + theta * (z - z_prev)
        y = torch.clamp_min(y + sigma * (constraints.az(z_bar) - constraints.rhs), 0.0)
        z_prev = z
        u = z - tau * (constraints.aty(y) + linear_grad)
        z = (u + tau * mass * z0) / (1.0 + tau * mass)
        z = constraints.overwrite_hard_equalities(z)
    residual = torch.clamp_min(constraints.az(z) - constraints.rhs, 0.0)
    return z, residual


def bdp(candidate, case, direction_net, beam: int = 8):
    active_pairs = build_active_pairs(candidate, case)
    logits = direction_net(candidate, case, active_pairs)
    assignments = build_direction_beam(logits, candidate, beam)
    states = expand_candidate(candidate, assignments)
    for _ in range(3):
        states = exact_shape_consensus(case, states)
        constraints = build_linear_constraints(case, states, assignments)
        grad = linearized_quality_gradient(case, states)
        states.z, residual = pdhg(states.z, constraints, grad,
                                  tau=0.2, sigma=0.2, theta=1.0, iters=48)
        states = exact_finalize(case, states)
        active_pairs = build_active_pairs(states, case)
    return fast_verify_and_select(case, states, residual)
