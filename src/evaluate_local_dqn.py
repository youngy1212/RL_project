"""Evaluate frozen Local DQN policies against local greedy baselines."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DEFAULT_PROCESSED_DIR, DEFAULT_RESULTS_DIR
from local_district_env import LocalDistrictBikeEnv, require_district_columns
from plot_utils import configure_korean_font
from train_local_dqn import district_demand_table, select_districts
from utils import ensure_dir, load_time_features, set_global_seed

configure_korean_font()


def load_processed_test(processed_dir: str) -> dict[str, object]:
    """Load arrays and metadata required for local evaluation."""
    base = Path(processed_dir)
    required = ["rental_test.npy", "return_test.npy", "capacity.npy", "distance_matrix.npy", "station_meta.csv"]
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Local DQN 평가에 필요한 processed 파일이 없습니다: {missing}")
    station_meta = pd.read_csv(base / "station_meta.csv", encoding="utf-8-sig")
    require_district_columns(station_meta)
    return {
        "rental": np.load(base / "rental_test.npy"),
        "returns": np.load(base / "return_test.npy"),
        "capacity": np.load(base / "capacity.npy"),
        "distance": np.load(base / "distance_matrix.npy"),
        "station_meta": station_meta,
        "time_features": load_time_features(base / "time_index_test.csv"),
    }


def local_no_rebalancing(env: LocalDistrictBikeEnv, obs: np.ndarray) -> int:
    """Local no-op policy."""
    return 0


def local_greedy(env: LocalDistrictBikeEnv, obs: np.ndarray) -> int:
    """Local greedy action: top surplus to top shortage."""
    return 1


def evaluate_env(env: LocalDistrictBikeEnv, policy_fn, policy_name: str, seed: int) -> dict[str, float | str]:
    """Evaluate a local policy for one deterministic episode."""
    totals = {
        "policy": policy_name,
        "total_reward": 0.0,
        "total_served_rentals": 0.0,
        "total_lost_rentals": 0.0,
        "total_accepted_returns": 0.0,
        "total_overflow_returns": 0.0,
        "total_reposition_distance_km": 0.0,
        "total_moved_bikes": 0.0,
    }
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = int(policy_fn(env, obs))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        totals["total_reward"] += float(reward)
        for key in [
            "total_served_rentals",
            "total_lost_rentals",
            "total_accepted_returns",
            "total_overflow_returns",
            "total_reposition_distance_km",
            "total_moved_bikes",
        ]:
            totals[key] += float(info.get(key, 0.0))
    served = totals["total_served_rentals"]
    lost = totals["total_lost_rentals"]
    accepted = totals["total_accepted_returns"]
    overflow = totals["total_overflow_returns"]
    totals["rental_satisfaction_rate"] = served / max(1.0, served + lost)
    totals["return_acceptance_rate"] = accepted / max(1.0, accepted + overflow)
    return totals


def add_relative_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add no-rebalancing-relative metrics per district."""
    result = frame.copy()
    for district_id, group in result.groupby("district_id"):
        no = group.loc[group["policy"].eq("No Rebalancing Local")]
        if no.empty:
            continue
        no_lost = float(no.iloc[0]["total_lost_rentals"])
        idx = group.index
        saved = no_lost - pd.to_numeric(result.loc[idx, "total_lost_rentals"], errors="coerce")
        result.loc[idx, "saved_rentals_vs_no_rebalancing"] = saved
        result.loc[idx, "lost_rentals_improvement_vs_no_rebalancing_pct"] = saved / max(1.0, no_lost) * 100.0
        result.loc[idx, "distance_per_saved_rental"] = pd.to_numeric(
            result.loc[idx, "total_reposition_distance_km"], errors="coerce"
        ) / np.maximum(1.0, saved)
    return result


def aggregate_policy_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate evaluated rows across all districts by policy."""
    evaluated = frame.loc[frame.get("status", "evaluated").eq("evaluated")].copy()
    if evaluated.empty:
        return pd.DataFrame()
    summed_cols = [
        "total_reward",
        "total_served_rentals",
        "total_lost_rentals",
        "total_accepted_returns",
        "total_overflow_returns",
        "total_reposition_distance_km",
        "total_moved_bikes",
    ]
    grouped = evaluated.groupby("policy", as_index=False)[summed_cols].sum()
    grouped.insert(0, "district_name", "ALL")
    grouped.insert(0, "district_id", -1)
    served = grouped["total_served_rentals"]
    lost = grouped["total_lost_rentals"]
    accepted = grouped["total_accepted_returns"]
    overflow = grouped["total_overflow_returns"]
    grouped["rental_satisfaction_rate"] = served / np.maximum(1.0, served + lost)
    grouped["return_acceptance_rate"] = accepted / np.maximum(1.0, accepted + overflow)
    grouped["status"] = "aggregate"
    grouped["reason"] = ""
    return add_relative_metrics(grouped)


def plot_grouped(frame: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    """Save a compact grouped bar chart by district and policy."""
    data = frame.loc[frame["district_id"].ne(-1) & frame["status"].eq("evaluated")]
    if data.empty or metric not in data.columns:
        return
    pivot = data.pivot(index="district_name", columns="policy", values=metric).fillna(0.0)
    pivot.plot(kind="bar", figsize=(12, 5))
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for local evaluation."""
    parser = argparse.ArgumentParser(description="자치구별 Local DQN을 Local Greedy와 비교 평가합니다.")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model-dir", default="models/local_dqn")
    parser.add_argument("--district-ids", default=None)
    parser.add_argument("--all-districts", action="store_true")
    parser.add_argument("--top-districts", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=999_999)
    parser.add_argument("--move-qty", type=int, default=10)
    parser.add_argument("--k-candidates", type=int, default=5)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Evaluate local policies and save district plus aggregate results."""
    args = parse_args(argv)
    set_global_seed(args.seed)
    from stable_baselines3 import DQN

    data = load_processed_test(args.processed_dir)
    demand_table = district_demand_table(data["station_meta"], data["rental"], data["returns"])
    target_ids = select_districts(args, demand_table)
    rows: list[dict[str, object]] = []

    for district_id in target_ids:
        row = demand_table.loc[demand_table["district_id"].astype(int).eq(int(district_id))]
        district_name = str(row["district_name"].iloc[0]) if not row.empty else str(district_id)
        try:
            base_env_kwargs = dict(
                rental_demand=data["rental"],
                return_demand=data["returns"],
                capacity=data["capacity"],
                distance_matrix=data["distance"],
                station_meta=data["station_meta"],
                target_district_id=int(district_id),
                time_features=data["time_features"],
                episode_length=args.episode_length,
                move_qty=args.move_qty,
                k_candidates=args.k_candidates,
                random_start=False,
                seed=args.seed,
            )
            for policy_name, policy_fn in [
                ("No Rebalancing Local", local_no_rebalancing),
                ("Local Greedy", local_greedy),
            ]:
                env = LocalDistrictBikeEnv(**base_env_kwargs)
                result = evaluate_env(env, policy_fn, policy_name, args.seed)
                rows.append({**result, "district_id": int(district_id), "district_name": district_name, "status": "evaluated", "reason": ""})

            model_path = Path(args.model_dir) / f"local_dqn_district_{int(district_id)}.zip"
            if not model_path.exists():
                rows.append(
                    {
                        "district_id": int(district_id),
                        "district_name": district_name,
                        "policy": "Local DQN",
                        "status": "skipped",
                        "reason": f"model not found: {model_path}",
                    }
                )
                continue
            model = DQN.load(model_path, device="cpu")

            def dqn_policy(env: LocalDistrictBikeEnv, obs: np.ndarray) -> int:
                action, _ = model.predict(obs, deterministic=True)
                return int(action)

            env = LocalDistrictBikeEnv(**base_env_kwargs)
            result = evaluate_env(env, dqn_policy, "Local DQN", args.seed)
            rows.append({**result, "district_id": int(district_id), "district_name": district_name, "status": "evaluated", "reason": ""})
        except Exception as exc:
            rows.append(
                {
                    "district_id": int(district_id),
                    "district_name": district_name,
                    "policy": "Local DQN",
                    "status": "failed",
                    "reason": repr(exc),
                }
            )

    result = add_relative_metrics(pd.DataFrame(rows))
    aggregate = aggregate_policy_rows(result)
    if not aggregate.empty:
        result = pd.concat([result, aggregate], ignore_index=True)
    tables_dir = ensure_dir(Path(args.results_dir) / "tables")
    figures_dir = ensure_dir(Path(args.results_dir) / "figures")
    output_path = tables_dir / "local_dqn_evaluation.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    plot_grouped(result, "total_lost_rentals", figures_dir / "local_lost_rentals_by_district.png", "Local Lost Rentals by District")
    plot_grouped(
        result,
        "rental_satisfaction_rate",
        figures_dir / "local_satisfaction_by_district.png",
        "Local Rental Satisfaction by District",
    )
    plot_grouped(
        result,
        "total_reposition_distance_km",
        figures_dir / "local_distance_by_district.png",
        "Local Reposition Distance by District",
    )
    plot_grouped(
        result,
        "saved_rentals_vs_no_rebalancing",
        figures_dir / "local_dqn_improvement_by_district.png",
        "Local Saved Rentals vs No Rebalancing",
    )
    print(f"[Local Eval] 결과 저장 완료: {output_path}")
    return result


if __name__ == "__main__":
    main()
