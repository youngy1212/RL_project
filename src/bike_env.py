"""Gymnasium environment for public bike rebalancing."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from local_district_env import build_local_observation, execute_greedy_local_action, execute_local_action
from local_policy_manager import LocalPolicyManager


class BikeRebalancingEnv(gym.Env):
    """Inventory simulation environment for station-level bike rebalancing.

    The environment uses observed rentals and returns as exogenous demand
    proxies. If simulated inventory cannot satisfy that demand, the unmet
    amount is counted as lost rental demand or overflow return demand.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rental_demand: np.ndarray,
        return_demand: np.ndarray,
        capacity: np.ndarray,
        distance_matrix: np.ndarray,
        time_features: np.ndarray | None = None,
        episode_length: int = 48,
        move_qty: int = 3,
        k_candidates: int = 3,
        random_start: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.rental_demand = np.asarray(rental_demand, dtype=np.float32)
        self.return_demand = np.asarray(return_demand, dtype=np.float32)
        self.capacity = np.asarray(capacity, dtype=np.float32)
        self.distance_matrix = np.asarray(distance_matrix, dtype=np.float32)
        self.time_features = None if time_features is None else np.asarray(time_features, dtype=np.float32)
        self.episode_length = int(episode_length)
        self.move_qty = float(move_qty)
        self.k_candidates = int(k_candidates)
        self.random_start = bool(random_start)

        if self.rental_demand.shape != self.return_demand.shape:
            raise ValueError("rental_demand와 return_demand의 shape가 같아야 합니다.")
        if self.rental_demand.ndim != 2:
            raise ValueError("demand 배열은 shape (T, N)의 2차원 배열이어야 합니다.")

        self.t_steps, self.n_stations = self.rental_demand.shape
        if self.t_steps == 0 or self.n_stations == 0:
            raise ValueError("demand 배열이 비어 있습니다.")
        if self.capacity.shape != (self.n_stations,):
            raise ValueError("capacity shape는 (N,)이어야 합니다.")
        if self.distance_matrix.shape != (self.n_stations, self.n_stations):
            raise ValueError("distance_matrix shape는 (N, N)이어야 합니다.")
        if self.time_features is not None and len(self.time_features) != self.t_steps:
            raise ValueError("time_features 길이는 demand time step 수와 같아야 합니다.")

        self.capacity = np.maximum(self.capacity, 1.0)
        self.rental_scale = np.percentile(self.rental_demand, 95, axis=0).astype(np.float32)
        self.return_scale = np.percentile(self.return_demand, 95, axis=0).astype(np.float32)
        self.rental_scale[self.rental_scale <= 0] = 1.0
        self.return_scale[self.return_scale <= 0] = 1.0

        obs_dim = 4 + 3 * self.n_stations
        low = np.concatenate(
            [
                np.full(4, -1.0, dtype=np.float32),
                np.zeros(3 * self.n_stations, dtype=np.float32),
            ]
        )
        high = np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(1 + self.k_candidates * self.k_candidates)
        self.action_space.seed(seed)

        self.inventory = np.zeros(self.n_stations, dtype=np.float32)
        self.prev_rental = np.zeros(self.n_stations, dtype=np.float32)
        self.prev_return = np.zeros(self.n_stations, dtype=np.float32)
        self.start_step = 0
        self.current_step = 0
        self.steps_in_episode = 0
        self.np_random = np.random.default_rng(seed)

    def _time_obs(self) -> np.ndarray:
        """Return cyclic time features for the current step."""
        step = min(self.current_step, self.t_steps - 1)
        if self.time_features is not None:
            return self.time_features[step].astype(np.float32)

        hour = (step % 48) / 2.0
        day = (step // 48) % 7
        hour_angle = 2.0 * np.pi * hour / 24.0
        day_angle = 2.0 * np.pi * day / 7.0
        return np.array(
            [np.sin(hour_angle), np.cos(hour_angle), np.sin(day_angle), np.cos(day_angle)],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        """Build the normalized observation vector."""
        inventory_ratio = np.clip(self.inventory / self.capacity, 0.0, 1.0)
        prev_rental_norm = np.clip(self.prev_rental / self.rental_scale, 0.0, 1.0)
        prev_return_norm = np.clip(self.prev_return / self.return_scale, 0.0, 1.0)
        obs = np.concatenate([self._time_obs(), inventory_ratio, prev_rental_norm, prev_return_norm])
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _candidate_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return surplus and shortage candidate station indices."""
        ratio = self.inventory / self.capacity
        surplus = np.argsort(-ratio)[: min(self.k_candidates, self.n_stations)]
        shortage = np.argsort(ratio)[: min(self.k_candidates, self.n_stations)]
        return surplus, shortage

    def _apply_action(self, action: int) -> tuple[float, float, float]:
        """Apply a discrete rebalancing action and return move metrics."""
        if action <= 0:
            return 0.0, 0.0, 0.0

        idx = int(action) - 1
        source_rank = idx // self.k_candidates
        target_rank = idx % self.k_candidates
        surplus, shortage = self._candidate_indices()
        if source_rank >= len(surplus) or target_rank >= len(shortage):
            return 0.0, 0.0, 0.0

        source = int(surplus[source_rank])
        target = int(shortage[target_rank])
        if source == target:
            return 0.0, 0.0, 0.0

        movable = min(self.move_qty, float(self.inventory[source]), float(self.capacity[target] - self.inventory[target]))
        moved = max(0.0, movable)
        if moved <= 0:
            return 0.0, 0.0, 0.0

        self.inventory[source] -= moved
        self.inventory[target] += moved
        distance = float(self.distance_matrix[source, target])
        cost = distance * moved
        return moved, distance, cost

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode state."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.action_space.seed(seed)

        max_start = max(0, self.t_steps - max(1, self.episode_length))
        if self.random_start and max_start > 0:
            self.start_step = int(self.np_random.integers(0, max_start + 1))
        else:
            self.start_step = 0

        self.current_step = self.start_step
        self.steps_in_episode = 0
        self.inventory = (self.capacity * 0.5).astype(np.float32)
        self.prev_rental = np.zeros(self.n_stations, dtype=np.float32)
        self.prev_return = np.zeros(self.n_stations, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Run one simulation step."""
        moved_bikes, reposition_distance_km, reposition_cost = self._apply_action(int(action))

        rental_t = self.rental_demand[self.current_step]
        return_t = self.return_demand[self.current_step]

        served_rentals = np.minimum(self.inventory, rental_t)
        lost_rentals = np.maximum(rental_t - served_rentals, 0.0)
        self.inventory -= served_rentals

        available_docks = np.maximum(self.capacity - self.inventory, 0.0)
        accepted_returns = np.minimum(available_docks, return_t)
        overflow_returns = np.maximum(return_t - accepted_returns, 0.0)
        self.inventory += accepted_returns
        self.inventory = np.clip(self.inventory, 0.0, self.capacity)

        raw_penalty = float(lost_rentals.sum() + 0.5 * overflow_returns.sum() + 0.05 * reposition_cost)
        demand_total = float(rental_t.sum() + return_t.sum())
        reward = -raw_penalty / max(1.0, demand_total)

        self.prev_rental = rental_t.astype(np.float32)
        self.prev_return = return_t.astype(np.float32)
        self.current_step += 1
        self.steps_in_episode += 1

        terminated = self.steps_in_episode >= self.episode_length or self.current_step >= self.t_steps
        truncated = False
        info = {
            "served_rentals": float(served_rentals.sum()),
            "lost_rentals": float(lost_rentals.sum()),
            "accepted_returns": float(accepted_returns.sum()),
            "overflow_returns": float(overflow_returns.sum()),
            "moved_bikes": float(moved_bikes),
            "reposition_distance_km": float(reposition_distance_km),
            "reposition_cost": float(reposition_cost),
            "reward_raw_penalty": float(raw_penalty),
            "inventory_mean": float(self.inventory.mean()),
            "inventory_min": float(self.inventory.min()),
            "inventory_max": float(self.inventory.max()),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


class DistrictBikeRebalancingEnv(gym.Env):
    """District-action environment with greedy or frozen-Local-DQN station moves."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        rental_demand: np.ndarray,
        return_demand: np.ndarray,
        capacity: np.ndarray,
        distance_matrix: np.ndarray,
        station_district: np.ndarray,
        district_names: list[str],
        time_features: np.ndarray | None = None,
        episode_length: int = 96,
        trucks: int = 10,
        truck_capacity: int = 10,
        random_start: bool = True,
        seed: int | None = None,
        lost_weight: float = 1.0,
        overflow_weight: float = 0.5,
        distance_weight: float = 0.05,
        imbalance_weight: float = 0.1,
        local_policy_mode: str = "greedy",
        local_model_dir: str = "models/local_dqn",
        local_policy_manager: LocalPolicyManager | None = None,
        local_decisions_per_global_step: int | None = None,
    ) -> None:
        super().__init__()
        self.rental_demand = np.asarray(rental_demand, dtype=np.float32)
        self.return_demand = np.asarray(return_demand, dtype=np.float32)
        self.capacity = np.maximum(np.asarray(capacity, dtype=np.float32), 1.0)
        self.distance_matrix = np.asarray(distance_matrix, dtype=np.float32)
        self.station_district = np.asarray(station_district, dtype=np.int64)
        self.district_names = list(district_names)
        self.time_features = None if time_features is None else np.asarray(time_features, dtype=np.float32)
        self.episode_length = int(episode_length)
        self.trucks = int(trucks)
        self.truck_capacity = float(truck_capacity)
        self.random_start = bool(random_start)
        self.lost_weight = float(lost_weight)
        self.overflow_weight = float(overflow_weight)
        self.distance_weight = float(distance_weight)
        self.imbalance_weight = float(imbalance_weight)
        self.local_policy_mode = str(local_policy_mode)
        if self.local_policy_mode not in {"greedy", "dqn"}:
            raise ValueError("local_policy_mode은 'greedy' 또는 'dqn'이어야 합니다.")
        self.local_model_dir = local_model_dir
        self.local_policy_manager = local_policy_manager
        if self.local_policy_mode == "dqn" and self.local_policy_manager is None:
            self.local_policy_manager = LocalPolicyManager(local_model_dir, deterministic=True, preload=False)
        self.local_decisions_per_global_step = (
            int(local_decisions_per_global_step) if local_decisions_per_global_step is not None else int(trucks)
        )

        if self.rental_demand.shape != self.return_demand.shape:
            raise ValueError("rental_demand와 return_demand의 shape가 같아야 합니다.")
        if self.rental_demand.ndim != 2:
            raise ValueError("demand 배열은 shape (T, N)의 2차원 배열이어야 합니다.")
        self.t_steps, self.n_stations = self.rental_demand.shape
        self.n_districts = len(self.district_names)
        if self.capacity.shape != (self.n_stations,):
            raise ValueError("capacity shape는 (N,)이어야 합니다.")
        if self.distance_matrix.shape != (self.n_stations, self.n_stations):
            raise ValueError("distance_matrix shape는 (N, N)이어야 합니다.")
        if self.station_district.shape != (self.n_stations,):
            raise ValueError("station_district shape는 (N,)이어야 합니다.")
        if self.station_district.min(initial=0) < 0 or self.station_district.max(initial=-1) >= self.n_districts:
            raise ValueError("station_district 값은 district_names 범위 안이어야 합니다.")
        if self.time_features is not None and len(self.time_features) != self.t_steps:
            raise ValueError("time_features 길이는 demand time step 수와 같아야 합니다.")

        self.district_station_indices = [np.where(self.station_district == i)[0] for i in range(self.n_districts)]
        self.district_capacity = np.array(
            [self.capacity[idx].sum() if len(idx) else 1.0 for idx in self.district_station_indices],
            dtype=np.float32,
        )
        district_rental = np.stack([self.rental_demand[:, idx].sum(axis=1) for idx in self.district_station_indices], axis=1)
        district_return = np.stack([self.return_demand[:, idx].sum(axis=1) for idx in self.district_station_indices], axis=1)
        self.rental_scale = np.percentile(district_rental, 95, axis=0).astype(np.float32)
        self.return_scale = np.percentile(district_return, 95, axis=0).astype(np.float32)
        self.rental_scale[self.rental_scale <= 0] = 1.0
        self.return_scale[self.return_scale <= 0] = 1.0
        self.station_rental_scale = np.percentile(self.rental_demand, 95, axis=0).astype(np.float32)
        self.station_return_scale = np.percentile(self.return_demand, 95, axis=0).astype(np.float32)
        self.station_rental_scale[self.station_rental_scale <= 0] = 1.0
        self.station_return_scale[self.station_return_scale <= 0] = 1.0

        obs_dim = 4 + 6 * self.n_districts
        low = np.concatenate([np.full(4, -1.0, dtype=np.float32), np.zeros(6 * self.n_districts, dtype=np.float32)])
        high = np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(1 + self.n_districts)
        self.action_space.seed(seed)

        self.inventory = np.zeros(self.n_stations, dtype=np.float32)
        self.prev_rental = np.zeros(self.n_stations, dtype=np.float32)
        self.prev_return = np.zeros(self.n_stations, dtype=np.float32)
        self.start_step = 0
        self.current_step = 0
        self.steps_in_episode = 0
        self.np_random = np.random.default_rng(seed)
        self.local_dqn_used_count = 0
        self.local_greedy_fallback_count = 0

    def _time_obs(self) -> np.ndarray:
        step = min(self.current_step, self.t_steps - 1)
        if self.time_features is not None:
            return self.time_features[step].astype(np.float32)
        hour = (step % 48) / 2.0
        day = (step // 48) % 7
        hour_angle = 2.0 * np.pi * hour / 24.0
        day_angle = 2.0 * np.pi * day / 7.0
        return np.array([np.sin(hour_angle), np.cos(hour_angle), np.sin(day_angle), np.cos(day_angle)], dtype=np.float32)

    def district_metrics(self) -> dict[str, np.ndarray]:
        """Return district-level metrics used by observations and baselines."""
        station_ratio = np.clip(self.inventory / self.capacity, 0.0, 1.0)
        inventory_ratio = np.zeros(self.n_districts, dtype=np.float32)
        recent_rental = np.zeros(self.n_districts, dtype=np.float32)
        recent_return = np.zeros(self.n_districts, dtype=np.float32)
        shortage_ratio = np.zeros(self.n_districts, dtype=np.float32)
        overflow_ratio = np.zeros(self.n_districts, dtype=np.float32)
        for district_id, idx in enumerate(self.district_station_indices):
            if len(idx) == 0:
                continue
            inventory_ratio[district_id] = float(self.inventory[idx].sum() / self.district_capacity[district_id])
            recent_rental[district_id] = float(self.prev_rental[idx].sum())
            recent_return[district_id] = float(self.prev_return[idx].sum())
            shortage_ratio[district_id] = float(np.mean(station_ratio[idx] < 0.2))
            overflow_ratio[district_id] = float(np.mean(station_ratio[idx] > 0.8))
        recent_rental_norm = np.clip(recent_rental / self.rental_scale, 0.0, 1.0)
        recent_return_norm = np.clip(recent_return / self.return_scale, 0.0, 1.0)
        net_demand_norm = np.clip((recent_rental - recent_return) / np.maximum(self.rental_scale + self.return_scale, 1.0), -1.0, 1.0)
        net_demand_norm = (net_demand_norm + 1.0) * 0.5
        return {
            "inventory_ratio": inventory_ratio,
            "recent_rental_norm": recent_rental_norm.astype(np.float32),
            "recent_return_norm": recent_return_norm.astype(np.float32),
            "shortage_ratio": shortage_ratio,
            "overflow_ratio": overflow_ratio,
            "net_demand_norm": net_demand_norm.astype(np.float32),
        }

    def _get_obs(self) -> np.ndarray:
        metrics = self.district_metrics()
        obs = np.concatenate(
            [
                self._time_obs(),
                metrics["inventory_ratio"],
                metrics["recent_rental_norm"],
                metrics["recent_return_norm"],
                metrics["shortage_ratio"],
                metrics["overflow_ratio"],
                metrics["net_demand_norm"],
            ]
        )
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _get_district_local_obs(self, district_id: int) -> np.ndarray:
        """Build the LocalDistrictBikeEnv-compatible observation for a district."""
        idx = self.district_station_indices[int(district_id)]
        return build_local_observation(
            self.current_step,
            self.t_steps,
            self.inventory[idx],
            self.capacity[idx],
            self.prev_rental[idx],
            self.prev_return[idx],
            self.rental_demand[:, idx],
            self.return_demand[:, idx],
            self.station_rental_scale[idx],
            self.station_return_scale[idx],
            self.time_features,
        )

    def _execute_local_action(self, district_id: int, local_action: int, k_candidates: int) -> tuple[float, float]:
        """Execute one local action inside a selected district."""
        idx = self.district_station_indices[int(district_id)]
        return execute_local_action(
            self.inventory,
            self.capacity,
            self.distance_matrix,
            int(local_action),
            self.truck_capacity,
            int(k_candidates),
            station_indices=idx,
        )

    def _execute_greedy_local_rebalancing(self, district_id: int) -> tuple[float, float]:
        """Execute repeated greedy local decisions inside one district."""
        moved_total = 0.0
        distance_total = 0.0
        idx = self.district_station_indices[int(district_id)]
        for _ in range(max(0, self.local_decisions_per_global_step)):
            moved, distance = execute_greedy_local_action(
                self.inventory,
                self.capacity,
                self.distance_matrix,
                self.truck_capacity,
                idx,
            )
            if moved <= 0:
                break
            moved_total += moved
            distance_total += distance
        return moved_total, distance_total

    def _execute_dqn_local_rebalancing(self, district_id: int) -> tuple[float, float, int, int]:
        """Execute repeated frozen Local DQN decisions inside one district."""
        if self.local_policy_manager is None or not self.local_policy_manager.has_model(int(district_id)):
            self.local_greedy_fallback_count += 1
            moved, distance = self._execute_greedy_local_rebalancing(district_id)
            return moved, distance, 0, 1

        moved_total = 0.0
        distance_total = 0.0
        used_count = 0
        fallback_count = 0
        for _ in range(max(0, self.local_decisions_per_global_step)):
            obs = self._get_district_local_obs(district_id)
            try:
                local_action = self.local_policy_manager.predict(int(district_id), obs)
                local_k = self.local_policy_manager.k_candidates(int(district_id))
            except Exception as exc:
                print(f"[WARN] Local DQN fallback: district_id={district_id}, reason={exc}")
                fallback_count += 1
                self.local_greedy_fallback_count += 1
                moved, distance = self._execute_greedy_local_rebalancing(district_id)
                moved_total += moved
                distance_total += distance
                break
            if local_action is None or local_k is None:
                fallback_count += 1
                self.local_greedy_fallback_count += 1
                moved, distance = self._execute_greedy_local_rebalancing(district_id)
            else:
                moved, distance = self._execute_local_action(district_id, local_action, local_k)
                used_count += 1
                self.local_dqn_used_count += 1
            moved_total += moved
            distance_total += distance
            if moved <= 0:
                break
        return moved_total, distance_total, used_count, fallback_count

    def _apply_district_action(self, action: int) -> tuple[int | None, str | None, float, float, int, int]:
        if action <= 0:
            return None, None, 0.0, 0.0, 0, 0
        district_id = int(action) - 1
        if district_id < 0 or district_id >= self.n_districts:
            return None, None, 0.0, 0.0, 0, 0
        if self.local_policy_mode == "dqn":
            moved_total, distance_total, used_count, fallback_count = self._execute_dqn_local_rebalancing(district_id)
        else:
            moved_total, distance_total = self._execute_greedy_local_rebalancing(district_id)
            used_count, fallback_count = 0, 0
        return district_id, self.district_names[district_id], moved_total, distance_total, used_count, fallback_count

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.action_space.seed(seed)
        max_start = max(0, self.t_steps - max(1, self.episode_length))
        if self.random_start and max_start > 0:
            self.start_step = int(self.np_random.integers(0, max_start + 1))
        else:
            self.start_step = 0
        self.current_step = self.start_step
        self.steps_in_episode = 0
        self.inventory = (self.capacity * 0.5).astype(np.float32)
        self.prev_rental = np.zeros(self.n_stations, dtype=np.float32)
        self.prev_return = np.zeros(self.n_stations, dtype=np.float32)
        self.local_dqn_used_count = 0
        self.local_greedy_fallback_count = 0
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        district_id, district_name, moved_bikes, reposition_distance_km, local_used, local_fallback = self._apply_district_action(
            int(action)
        )

        rental_t = self.rental_demand[self.current_step]
        return_t = self.return_demand[self.current_step]
        served_rentals = np.minimum(self.inventory, rental_t)
        lost_rentals = np.maximum(rental_t - served_rentals, 0.0)
        self.inventory -= served_rentals

        available_docks = np.maximum(self.capacity - self.inventory, 0.0)
        accepted_returns = np.minimum(available_docks, return_t)
        overflow_returns = np.maximum(return_t - accepted_returns, 0.0)
        self.inventory += accepted_returns
        self.inventory = np.clip(self.inventory, 0.0, self.capacity)

        station_ratio = np.clip(self.inventory / self.capacity, 0.0, 1.0)
        shortage_risk = np.maximum(0.0, 0.2 - station_ratio).sum()
        overflow_risk = np.maximum(0.0, station_ratio - 0.8).sum()
        raw_penalty = float(
            self.lost_weight * lost_rentals.sum()
            + self.overflow_weight * overflow_returns.sum()
            + self.distance_weight * reposition_distance_km
            + self.imbalance_weight * (shortage_risk + overflow_risk)
        )
        demand_total = float(rental_t.sum() + return_t.sum())
        reward = -raw_penalty / max(1.0, demand_total)

        self.prev_rental = rental_t.astype(np.float32)
        self.prev_return = return_t.astype(np.float32)
        metrics = self.district_metrics()
        self.current_step += 1
        self.steps_in_episode += 1
        terminated = self.steps_in_episode >= self.episode_length or self.current_step >= self.t_steps
        truncated = False
        info = {
            "selected_district_id": -1 if district_id is None else int(district_id),
            "selected_district_name": "" if district_name is None else district_name,
            "local_policy_mode": self.local_policy_mode,
            "local_dqn_used_count": int(local_used),
            "local_greedy_fallback_count": int(local_fallback),
            "total_served_rentals": float(served_rentals.sum()),
            "total_lost_rentals": float(lost_rentals.sum()),
            "total_accepted_returns": float(accepted_returns.sum()),
            "total_overflow_returns": float(overflow_returns.sum()),
            "total_moved_bikes": float(moved_bikes),
            "total_reposition_distance_km": float(reposition_distance_km),
            "served_rentals": float(served_rentals.sum()),
            "lost_rentals": float(lost_rentals.sum()),
            "accepted_returns": float(accepted_returns.sum()),
            "overflow_returns": float(overflow_returns.sum()),
            "moved_bikes": float(moved_bikes),
            "reposition_distance_km": float(reposition_distance_km),
            "reward_raw_penalty": float(raw_penalty),
            "district_inventory_ratio_mean": float(metrics["inventory_ratio"].mean()),
            "district_inventory_ratio_min": float(metrics["inventory_ratio"].min()),
            "district_inventory_ratio_max": float(metrics["inventory_ratio"].max()),
            "shortage_station_count": int((station_ratio < 0.2).sum()),
            "overflow_station_count": int((station_ratio > 0.8).sum()),
        }
        return self._get_obs(), float(reward), terminated, truncated, info
