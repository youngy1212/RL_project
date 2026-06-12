"""Local district-level Gymnasium environment for station-to-station rebalancing."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


def require_district_columns(station_meta: pd.DataFrame) -> None:
    """Validate district columns required by local and hierarchical training."""
    missing = [col for col in ["district_id", "district_name"] if col not in station_meta.columns]
    if missing:
        raise ValueError(
            f"station_meta.csv에 district 컬럼이 없습니다: missing={missing}. "
            "preprocess.py --zone-mode district --use-all-stations를 먼저 실행하세요."
        )


def district_station_indices(station_meta: pd.DataFrame, target_district_id: int) -> np.ndarray:
    """Return global station indices for one district from station_meta order."""
    require_district_columns(station_meta)
    mask = station_meta["district_id"].astype(int).eq(int(target_district_id))
    return np.flatnonzero(mask.to_numpy())


def district_name_for_id(station_meta: pd.DataFrame, target_district_id: int) -> str:
    """Return a district name for a district id."""
    require_district_columns(station_meta)
    rows = station_meta.loc[station_meta["district_id"].astype(int).eq(int(target_district_id)), "district_name"]
    if rows.empty:
        return str(target_district_id)
    return str(rows.iloc[0])


def time_features_for_step(step: int, t_steps: int, time_features: np.ndarray | None = None) -> np.ndarray:
    """Return cyclic time features for a timestep."""
    safe_step = min(max(0, int(step)), max(0, int(t_steps) - 1))
    if time_features is not None:
        return np.asarray(time_features[safe_step], dtype=np.float32)
    hour = (safe_step % 48) / 2.0
    day = (safe_step // 48) % 7
    hour_angle = 2.0 * np.pi * hour / 24.0
    day_angle = 2.0 * np.pi * day / 7.0
    return np.array([np.sin(hour_angle), np.cos(hour_angle), np.sin(day_angle), np.cos(day_angle)], dtype=np.float32)


def rolling_mean_before(demand: np.ndarray, current_step: int, window: int = 3) -> np.ndarray:
    """Return shifted rolling mean using only timesteps before current_step."""
    if current_step <= 0:
        return np.zeros(demand.shape[1], dtype=np.float32)
    start = max(0, int(current_step) - int(window))
    return demand[start:int(current_step)].mean(axis=0).astype(np.float32)


def build_local_observation(
    current_step: int,
    t_steps: int,
    inventory: np.ndarray,
    capacity: np.ndarray,
    prev_rental: np.ndarray,
    prev_return: np.ndarray,
    rental_demand: np.ndarray,
    return_demand: np.ndarray,
    rental_scale: np.ndarray,
    return_scale: np.ndarray,
    time_features: np.ndarray | None = None,
) -> np.ndarray:
    """Build the LocalDistrictBikeEnv observation vector."""
    capacity = np.maximum(np.asarray(capacity, dtype=np.float32), 1.0)
    inventory_ratio = np.clip(np.asarray(inventory, dtype=np.float32) / capacity, 0.0, 1.0)
    prev_rental_norm = np.clip(np.asarray(prev_rental, dtype=np.float32) / rental_scale, 0.0, 1.0)
    prev_return_norm = np.clip(np.asarray(prev_return, dtype=np.float32) / return_scale, 0.0, 1.0)
    rolling_rental = rolling_mean_before(rental_demand, current_step)
    rolling_return = rolling_mean_before(return_demand, current_step)
    rolling_rental_norm = np.clip(rolling_rental / rental_scale, 0.0, 1.0)
    rolling_return_norm = np.clip(rolling_return / return_scale, 0.0, 1.0)
    obs = np.concatenate(
        [
            time_features_for_step(current_step, t_steps, time_features),
            inventory_ratio,
            prev_rental_norm,
            prev_return_norm,
            rolling_rental_norm,
            rolling_return_norm,
        ]
    )
    return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def execute_local_action(
    inventory: np.ndarray,
    capacity: np.ndarray,
    distance_matrix: np.ndarray,
    action: int,
    move_qty: float,
    k_candidates: int,
    station_indices: np.ndarray | None = None,
) -> tuple[float, float]:
    """Apply a local DQN action to local or global inventory and return moved bikes and km."""
    if action <= 0:
        return 0.0, 0.0
    if station_indices is None:
        idx = np.arange(len(inventory))
    else:
        idx = np.asarray(station_indices, dtype=int)
    if len(idx) < 2:
        return 0.0, 0.0

    ratio = np.asarray(inventory[idx], dtype=np.float32) / np.maximum(np.asarray(capacity[idx], dtype=np.float32), 1.0)
    k = min(int(k_candidates), len(idx))
    source_rank = (int(action) - 1) // int(k_candidates)
    target_rank = (int(action) - 1) % int(k_candidates)
    surplus = idx[np.argsort(-ratio)[:k]]
    shortage = idx[np.argsort(ratio)[:k]]
    if source_rank >= len(surplus) or target_rank >= len(shortage):
        return 0.0, 0.0

    source = int(surplus[source_rank])
    target = int(shortage[target_rank])
    if source == target:
        return 0.0, 0.0
    source_ratio = float(inventory[source] / max(1.0, capacity[source]))
    target_ratio = float(inventory[target] / max(1.0, capacity[target]))
    if source_ratio <= 0.5 or target_ratio >= 0.8:
        return 0.0, 0.0

    movable_from_source = float(inventory[source] - 0.5 * capacity[source])
    movable_to_target = float(0.8 * capacity[target] - inventory[target])
    moved = float(np.floor(min(float(move_qty), movable_from_source, movable_to_target)))
    if moved < 1.0:
        return 0.0, 0.0

    inventory[source] -= moved
    inventory[target] += moved
    inventory[:] = np.clip(inventory, 0.0, capacity)
    return moved, float(distance_matrix[source, target] * moved)


def execute_greedy_local_action(
    inventory: np.ndarray,
    capacity: np.ndarray,
    distance_matrix: np.ndarray,
    move_qty: float,
    station_indices: np.ndarray,
) -> tuple[float, float]:
    """Move from the highest-ratio station to the lowest-ratio station."""
    if len(station_indices) < 2:
        return 0.0, 0.0
    ratio = inventory[station_indices] / np.maximum(capacity[station_indices], 1.0)
    source = int(station_indices[int(np.argmax(ratio))])
    target = int(station_indices[int(np.argmin(ratio))])
    if source == target:
        return 0.0, 0.0
    source_ratio = float(inventory[source] / max(1.0, capacity[source]))
    target_ratio = float(inventory[target] / max(1.0, capacity[target]))
    if source_ratio <= 0.5 or target_ratio >= 0.8:
        return 0.0, 0.0
    movable_from_source = float(inventory[source] - 0.5 * capacity[source])
    movable_to_target = float(0.8 * capacity[target] - inventory[target])
    moved = float(np.floor(min(float(move_qty), movable_from_source, movable_to_target)))
    if moved < 1.0:
        return 0.0, 0.0
    inventory[source] -= moved
    inventory[target] += moved
    inventory[:] = np.clip(inventory, 0.0, capacity)
    return moved, float(distance_matrix[source, target] * moved)


class LocalDistrictBikeEnv(gym.Env):
    """One-district station-to-station rebalancing environment for Local DQN."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        rental_demand: np.ndarray,
        return_demand: np.ndarray,
        capacity: np.ndarray,
        distance_matrix: np.ndarray,
        station_meta: pd.DataFrame | np.ndarray,
        target_district_id: int,
        time_features: np.ndarray | None = None,
        episode_length: int = 96,
        move_qty: int = 10,
        k_candidates: int = 5,
        random_start: bool = True,
        seed: int | None = None,
        lost_weight: float = 1.0,
        overflow_weight: float = 0.5,
        distance_weight: float = 0.05,
        imbalance_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.target_district_id = int(target_district_id)
        if isinstance(station_meta, pd.DataFrame):
            self.global_station_indices = district_station_indices(station_meta, self.target_district_id)
            self.district_name = district_name_for_id(station_meta, self.target_district_id)
        else:
            station_district = np.asarray(station_meta, dtype=np.int64)
            self.global_station_indices = np.flatnonzero(station_district == self.target_district_id)
            self.district_name = str(self.target_district_id)
        if len(self.global_station_indices) < 2:
            raise ValueError(
                f"LocalDistrictBikeEnv 생성 실패: district_id={target_district_id}의 station 수가 2개 미만입니다."
            )

        rental_demand = np.asarray(rental_demand, dtype=np.float32)
        return_demand = np.asarray(return_demand, dtype=np.float32)
        capacity = np.maximum(np.asarray(capacity, dtype=np.float32), 1.0)
        distance_matrix = np.asarray(distance_matrix, dtype=np.float32)
        self.rental_demand = rental_demand[:, self.global_station_indices].astype(np.float32)
        self.return_demand = return_demand[:, self.global_station_indices].astype(np.float32)
        self.capacity = capacity[self.global_station_indices].astype(np.float32)
        self.distance_matrix = distance_matrix[np.ix_(self.global_station_indices, self.global_station_indices)].astype(np.float32)
        self.time_features = None if time_features is None else np.asarray(time_features, dtype=np.float32)
        self.episode_length = int(episode_length)
        self.move_qty = float(move_qty)
        self.k_candidates = int(k_candidates)
        self.random_start = bool(random_start)
        self.lost_weight = float(lost_weight)
        self.overflow_weight = float(overflow_weight)
        self.distance_weight = float(distance_weight)
        self.imbalance_weight = float(imbalance_weight)

        if self.rental_demand.shape != self.return_demand.shape:
            raise ValueError("rental_demand와 return_demand의 shape가 같아야 합니다.")
        self.t_steps, self.n_stations = self.rental_demand.shape
        self.rental_scale = np.percentile(self.rental_demand, 95, axis=0).astype(np.float32)
        self.return_scale = np.percentile(self.return_demand, 95, axis=0).astype(np.float32)
        self.rental_scale[self.rental_scale <= 0] = 1.0
        self.return_scale[self.return_scale <= 0] = 1.0

        obs_dim = 4 + 5 * self.n_stations
        self.observation_space = spaces.Box(
            low=np.concatenate([np.full(4, -1.0, dtype=np.float32), np.zeros(5 * self.n_stations, dtype=np.float32)]),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(1 + self.k_candidates * self.k_candidates)
        self.action_space.seed(seed)
        self.np_random = np.random.default_rng(seed)
        self.inventory = np.zeros(self.n_stations, dtype=np.float32)
        self.prev_rental = np.zeros(self.n_stations, dtype=np.float32)
        self.prev_return = np.zeros(self.n_stations, dtype=np.float32)
        self.start_step = 0
        self.current_step = 0
        self.steps_in_episode = 0

    def _get_obs(self) -> np.ndarray:
        """Return current local observation."""
        obs = build_local_observation(
            self.current_step,
            self.t_steps,
            self.inventory,
            self.capacity,
            self.prev_rental,
            self.prev_return,
            self.rental_demand,
            self.return_demand,
            self.rental_scale,
            self.return_scale,
            self.time_features,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset inventory and episode position."""
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
        self.inventory = np.floor(self.capacity * 0.5).astype(np.float32)
        self.prev_rental = np.zeros(self.n_stations, dtype=np.float32)
        self.prev_return = np.zeros(self.n_stations, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Run one local district simulation step."""
        moved_bikes, reposition_distance_km = execute_local_action(
            self.inventory,
            self.capacity,
            self.distance_matrix,
            int(action),
            self.move_qty,
            self.k_candidates,
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
        self.inventory = np.clip(np.floor(self.inventory), 0.0, self.capacity)

        station_ratio = np.clip(self.inventory / self.capacity, 0.0, 1.0)
        shortage_risk = np.maximum(0.0, 0.2 - station_ratio).sum()
        overflow_risk = np.maximum(0.0, station_ratio - 0.8).sum()
        imbalance_penalty = shortage_risk + overflow_risk
        raw_penalty = float(
            self.lost_weight * lost_rentals.sum()
            + self.overflow_weight * overflow_returns.sum()
            + self.distance_weight * reposition_distance_km
            + self.imbalance_weight * imbalance_penalty
        )
        demand_total = float(rental_t.sum() + return_t.sum())
        reward = -raw_penalty / max(1.0, demand_total)

        self.prev_rental = rental_t.astype(np.float32)
        self.prev_return = return_t.astype(np.float32)
        self.current_step += 1
        self.steps_in_episode += 1
        terminated = self.steps_in_episode >= self.episode_length or self.current_step >= self.t_steps
        truncated = False
        info = {
            "district_id": int(self.target_district_id),
            "district_name": self.district_name,
            "total_served_rentals": float(served_rentals.sum()),
            "total_lost_rentals": float(lost_rentals.sum()),
            "total_accepted_returns": float(accepted_returns.sum()),
            "total_overflow_returns": float(overflow_returns.sum()),
            "total_moved_bikes": float(moved_bikes),
            "total_reposition_distance_km": float(reposition_distance_km),
            "reward_raw_penalty": float(raw_penalty),
            "inventory_mean": float(self.inventory.mean()),
            "inventory_min": float(self.inventory.min()),
            "inventory_max": float(self.inventory.max()),
        }
        return self._get_obs(), float(reward), terminated, truncated, info
