"""Baseline policies and evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


PolicyFn = Callable[[object, np.ndarray], int]


def no_rebalancing_policy(env: object, obs: np.ndarray) -> int:
    """Always choose no rebalancing."""
    return 0


def random_policy(env: object, obs: np.ndarray) -> int:
    """Sample a random action from the environment action space."""
    return int(env.action_space.sample())


def greedy_policy(env: object, obs: np.ndarray) -> int:
    """Move from the top surplus candidate to the top shortage candidate."""
    return 1


def rule_based_policy(env: object, obs: np.ndarray) -> int:
    """Use action 1 only when inventory imbalance crosses simple thresholds."""
    ratio = env.inventory / env.capacity
    if float(ratio.min()) < 0.25 and float(ratio.max()) > 0.75:
        return 1
    return 0


def random_district_policy(env: object, obs: np.ndarray) -> int:
    """Sample a district action, including no-op."""
    return int(env.action_space.sample())


def greedy_district_policy(env: object, obs: np.ndarray) -> int:
    """Select the district with the largest simple shortage pressure."""
    metrics = env.district_metrics()
    station_ratio = env.inventory / env.capacity
    shortage_counts = np.array(
        [float((station_ratio[idx] < 0.2).sum()) for idx in env.district_station_indices],
        dtype=np.float32,
    )
    score = shortage_counts + metrics["recent_rental_norm"] - metrics["recent_return_norm"]
    return int(np.argmax(score)) + 1


def rule_based_district_policy(env: object, obs: np.ndarray) -> int:
    """Choose one stressed district, otherwise no-op."""
    metrics = env.district_metrics()
    score = metrics["shortage_ratio"] + metrics["recent_rental_norm"] - metrics["recent_return_norm"]
    eligible = (metrics["inventory_ratio"] < 0.35) | (metrics["shortage_ratio"] > 0.3)
    if not bool(np.any(eligible)):
        return 0
    masked_score = np.where(eligible, score, -np.inf)
    return int(np.argmax(masked_score)) + 1


def evaluate_policy(
    env: object,
    policy_fn: PolicyFn,
    n_episodes: int = 10,
    policy_name: str | None = None,
    seed: int | None = 42,
) -> dict[str, float | str]:
    """Evaluate a policy function and return aggregate metrics."""
    totals = {
        "total_reward": 0.0,
        "total_served_rentals": 0.0,
        "total_lost_rentals": 0.0,
        "total_accepted_returns": 0.0,
        "total_overflow_returns": 0.0,
        "total_reposition_distance_km": 0.0,
        "total_moved_bikes": 0.0,
        "local_dqn_used_count": 0.0,
        "local_greedy_fallback_count": 0.0,
    }

    for episode in range(n_episodes):
        reset_seed = None if seed is None else seed + episode
        obs, _ = env.reset(seed=reset_seed)
        done = False
        while not done:
            action = int(policy_fn(env, obs))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            totals["total_reward"] += float(reward)
            totals["total_served_rentals"] += float(info.get("total_served_rentals", info.get("served_rentals", 0.0)))
            totals["total_lost_rentals"] += float(info.get("total_lost_rentals", info.get("lost_rentals", 0.0)))
            totals["total_accepted_returns"] += float(info.get("total_accepted_returns", info.get("accepted_returns", 0.0)))
            totals["total_overflow_returns"] += float(info.get("total_overflow_returns", info.get("overflow_returns", 0.0)))
            totals["total_reposition_distance_km"] += float(
                info.get("total_reposition_distance_km", info.get("reposition_distance_km", 0.0))
            )
            totals["total_moved_bikes"] += float(info.get("total_moved_bikes", info.get("moved_bikes", 0.0)))
            totals["local_dqn_used_count"] += float(info.get("local_dqn_used_count", 0.0))
            totals["local_greedy_fallback_count"] += float(info.get("local_greedy_fallback_count", 0.0))

    served = totals["total_served_rentals"]
    lost = totals["total_lost_rentals"]
    accepted = totals["total_accepted_returns"]
    overflow = totals["total_overflow_returns"]
    totals["rental_satisfaction_rate"] = served / max(1.0, served + lost)
    totals["return_acceptance_rate"] = accepted / max(1.0, accepted + overflow)
    if policy_name:
        return {"policy": policy_name, **totals}
    return totals
