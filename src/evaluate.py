"""Evaluate DQN and baseline rebalancing policies on the test set."""

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

from baselines import (
    evaluate_policy,
    greedy_district_policy,
    greedy_policy,
    no_rebalancing_policy,
    random_policy,
    random_district_policy,
    rule_based_district_policy,
    rule_based_policy,
)
from bike_env import BikeRebalancingEnv, DistrictBikeRebalancingEnv
from config import DEFAULT_MODEL_PATH, DEFAULT_PROCESSED_DIR, DEFAULT_RESULTS_DIR
from plot_utils import configure_korean_font
from utils import ensure_dir, load_time_features, set_global_seed

configure_korean_font()


def load_test_data(processed_dir: str) -> dict[str, np.ndarray]:
    """Load processed test arrays."""
    base = Path(processed_dir)
    required = [
        "rental_test.npy",
        "return_test.npy",
        "capacity.npy",
        "distance_matrix.npy",
    ]
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"processed 데이터가 부족합니다: {missing}\n"
            "먼저 python src/preprocess.py 를 실행해 data/processed/ 산출물을 생성해주세요."
        )
    data = {
        "rental": np.load(base / "rental_test.npy"),
        "returns": np.load(base / "return_test.npy"),
        "capacity": np.load(base / "capacity.npy"),
        "distance": np.load(base / "distance_matrix.npy"),
        "time_features": load_time_features(base / "time_index_test.csv"),
    }
    if (base / "station_district.npy").exists():
        data["station_district"] = np.load(base / "station_district.npy")
    if (base / "district_meta.csv").exists():
        data["district_names"] = pd.read_csv(base / "district_meta.csv")["district_name"].astype(str).tolist()
    return data


def make_env(
    data: dict[str, np.ndarray],
    env_type: str,
    episode_length: int,
    move_qty: int,
    k_candidates: int,
    seed: int,
    trucks: int,
    truck_capacity: int,
    local_policy_mode: str = "greedy",
    local_model_dir: str = "models/local_dqn",
    local_decisions_per_global_step: int | None = None,
) -> BikeRebalancingEnv | DistrictBikeRebalancingEnv:
    """Create a deterministic test environment."""
    if env_type == "district":
        if "station_district" not in data or "district_names" not in data:
            raise FileNotFoundError(
                "district 환경에는 station_district.npy와 district_meta.csv가 필요합니다. "
                "preprocess.py --zone-mode district --use-all-stations를 먼저 실행하세요."
            )
        return DistrictBikeRebalancingEnv(
            data["rental"],
            data["returns"],
            data["capacity"],
            data["distance"],
            data["station_district"],
            data["district_names"],
            time_features=data["time_features"],
            episode_length=episode_length,
            trucks=trucks,
            truck_capacity=truck_capacity,
            random_start=False,
            seed=seed,
            local_policy_mode=local_policy_mode,
            local_model_dir=local_model_dir,
            local_decisions_per_global_step=local_decisions_per_global_step,
        )
    return BikeRebalancingEnv(
        data["rental"],
        data["returns"],
        data["capacity"],
        data["distance"],
        time_features=data["time_features"],
        episode_length=episode_length,
        move_qty=move_qty,
        k_candidates=k_candidates,
        random_start=False,
        seed=seed,
    )


def add_comparison_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add no-rebalancing-relative comparison columns."""
    result = frame.copy()
    no_rows = result.loc[result["policy"] == "No Rebalancing"]
    if no_rows.empty:
        return result
    baseline = no_rows.iloc[0]
    no_lost = float(baseline["total_lost_rentals"])
    no_reward = float(baseline["total_reward"])
    saved = no_lost - result["total_lost_rentals"].astype(float)
    result["saved_rentals_vs_no_rebalancing"] = saved
    result["lost_rentals_improvement_vs_no_rebalancing_pct"] = saved / max(1.0, no_lost) * 100.0
    result["reward_improvement_vs_no_rebalancing_pct"] = (
        (result["total_reward"].astype(float) - no_reward) / max(1.0, abs(no_reward)) * 100.0
    )
    result["distance_per_saved_rental"] = result["total_reposition_distance_km"].astype(float) / np.maximum(1.0, saved)
    return result


def plot_metric(frame: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: Path) -> None:
    """Save a simple bar chart for one evaluation metric."""
    values = pd.to_numeric(frame[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    plt.figure(figsize=(8, 4.5))
    plt.bar(frame["policy"], values, color="#3A7CA5")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


BASE_PLOTS = [
    ("total_reward", "Total reward", "Policy Reward Comparison", "reward_comparison.png"),
    ("total_lost_rentals", "Lost rentals", "Lost Rentals Comparison", "lost_rentals_comparison.png"),
    ("total_reposition_distance_km", "Reposition distance (km)", "Reposition Distance Comparison", "reposition_distance_comparison.png"),
    ("rental_satisfaction_rate", "Rental satisfaction rate", "Rental Satisfaction Comparison", "rental_satisfaction_comparison.png"),
    ("saved_rentals_vs_no_rebalancing", "Saved rentals", "Lost Rentals Delta vs No Rebalancing", "lost_rentals_delta_vs_no_rebalancing.png"),
    ("distance_per_saved_rental", "km per saved rental", "Distance per Saved Rental", "distance_per_saved_rental.png"),
]

EXPERIMENT_PLOTS = [
    ("total_reward", "Total reward", "Reward Comparison", "reward_comparison.png"),
    ("total_lost_rentals", "Lost rentals", "Lost Rentals Comparison", "lost_rentals_comparison.png"),
    ("saved_rentals_vs_no_rebalancing", "Saved rentals", "Lost Rentals Delta", "lost_rentals_delta.png"),
    ("rental_satisfaction_rate", "Rental satisfaction rate", "Satisfaction Comparison", "satisfaction_comparison.png"),
    ("total_reposition_distance_km", "Reposition distance (km)", "Reposition Distance Comparison", "reposition_distance_comparison.png"),
    ("distance_per_saved_rental", "km per saved rental", "Distance per Saved Rental", "distance_per_saved_rental.png"),
]


def save_evaluation_plots(frame: pd.DataFrame, figures_dir: Path, experiment_prefix: str = "") -> None:
    """Save common and experiment-specific evaluation plots."""
    for metric, ylabel, title, filename in BASE_PLOTS:
        plot_metric(frame, metric, ylabel, title, figures_dir / filename)
    plot_metric(frame, "total_reward", "Total reward", "Policy Reward Comparison", figures_dir / "baseline_comparison_reward.png")

    if not experiment_prefix:
        return

    for metric, ylabel, title, filename in EXPERIMENT_PLOTS:
        specific_name = f"{experiment_prefix}_{filename}"
        specific_title = f"{experiment_prefix} {title}"
        plot_metric(frame, metric, ylabel, specific_title, figures_dir / specific_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="test set에서 DQN과 baseline 정책을 평가합니다.")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--move-qty", type=int, default=3)
    parser.add_argument("--k-candidates", type=int, default=3)
    parser.add_argument("--env-type", choices=["station", "district"], default="station")
    parser.add_argument("--local-policy-mode", choices=["greedy", "dqn"], default="greedy")
    parser.add_argument("--local-model-dir", default="models/local_dqn")
    parser.add_argument("--local-decisions-per-global-step", type=int, default=None)
    parser.add_argument("--compare-hierarchical", action="store_true")
    parser.add_argument("--experiment-name", default="default")
    parser.add_argument("--trucks", type=int, default=10)
    parser.add_argument("--truck-capacity", type=int, default=10)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _dqn_policy_from_path(model_path: Path):
    """Load a DQN model and return a policy function."""
    from stable_baselines3 import DQN

    model = DQN.load(model_path, device="cpu")

    def dqn_policy(env: object, obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return dqn_policy


def resolve_dqn_model_path(args: argparse.Namespace) -> Path | None:
    """Resolve a DQN model path with district-level compatibility fallback."""
    candidates: list[Path] = []
    if args.model_path:
        candidates.append(Path(args.model_path))
    elif args.env_type == "district" and args.local_policy_mode == "greedy":
        candidates.append(Path("models/global_dqn_district_local_greedy.zip"))
    else:
        candidates.append(Path(DEFAULT_MODEL_PATH))

    fallback = Path(DEFAULT_MODEL_PATH)
    if fallback not in candidates:
        candidates.append(fallback)

    for path in candidates:
        if path.exists():
            if args.model_path and path != Path(args.model_path):
                print(f"[Evaluate] 지정 모델이 없어 fallback 모델을 사용합니다: {path}")
            return path
    return None


def experiment_output_paths(results_dir: Path, experiment_name: str) -> tuple[Path, str]:
    """Return the experiment-specific table path and figure prefix."""
    tables_dir = ensure_dir(results_dir / "tables")
    if experiment_name == "district_only":
        return tables_dir / "district_only_baseline_comparison.csv", "district_only"
    if experiment_name == "hierarchical":
        return tables_dir / "hierarchical_baseline_comparison.csv", "hierarchical"
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in experiment_name.strip())
    if safe_name and safe_name != "default":
        return tables_dir / f"{safe_name}_baseline_comparison.csv", safe_name
    return tables_dir / "baseline_comparison.csv", ""


def evaluate_hierarchical(args: argparse.Namespace, data: dict[str, np.ndarray], episode_length: int) -> pd.DataFrame:
    """Evaluate hierarchical global/local policy combinations."""
    specs = [
        {
            "name": "No Rebalancing",
            "local_policy_mode": "greedy",
            "policy_fn": no_rebalancing_policy,
            "model_path": None,
        },
        {
            "name": "Greedy District + Greedy Local",
            "local_policy_mode": "greedy",
            "policy_fn": greedy_district_policy,
            "model_path": None,
        },
        {
            "name": "Global DQN + Greedy Local",
            "local_policy_mode": "greedy",
            "policy_fn": None,
            "model_path": Path(args.model_path) if args.model_path else Path("models/global_dqn_district_local_greedy.zip"),
        },
        {
            "name": "Greedy District + Local DQN",
            "local_policy_mode": "dqn",
            "policy_fn": greedy_district_policy,
            "model_path": None,
        },
        {
            "name": "Global DQN + Local DQN",
            "local_policy_mode": "dqn",
            "policy_fn": None,
            "model_path": Path("models/global_dqn_district_local_dqn.zip"),
        },
    ]
    rows: list[dict] = []
    for spec in specs:
        policy_fn = spec["policy_fn"]
        model_path = spec["model_path"]
        if policy_fn is None:
            if model_path is None or not model_path.exists():
                print(f"[Evaluate] 정책을 건너뜁니다. model not found: {spec['name']} path={model_path}")
                continue
            try:
                policy_fn = _dqn_policy_from_path(model_path)
            except Exception as exc:
                print(f"[Evaluate] 정책을 건너뜁니다. model load failed: {spec['name']} reason={exc}")
                continue
        env = make_env(
            data,
            "district",
            episode_length,
            args.move_qty,
            args.k_candidates,
            args.seed,
            args.trucks,
            args.truck_capacity,
            local_policy_mode=spec["local_policy_mode"],
            local_model_dir=args.local_model_dir,
            local_decisions_per_global_step=args.local_decisions_per_global_step,
        )
        try:
            rows.append(evaluate_policy(env, policy_fn, n_episodes=args.n_episodes, policy_name=spec["name"], seed=args.seed))
        except Exception as exc:
            print(f"[Evaluate] 정책 평가를 건너뜁니다: {spec['name']} reason={exc}")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Evaluate baseline policies and, when available, a trained DQN model."""
    args = parse_args(argv)
    set_global_seed(args.seed)
    data = load_test_data(args.processed_dir)
    episode_length = args.episode_length or len(data["rental"])

    if args.compare_hierarchical:
        if args.env_type != "district":
            raise ValueError("--compare-hierarchical는 --env-type district에서만 사용할 수 있습니다.")
        result = evaluate_hierarchical(args, data, episode_length)
    elif args.env_type == "district":
        policy_map = {
            "No Rebalancing": no_rebalancing_policy,
            "Random District": random_district_policy,
            "Greedy District": greedy_district_policy,
            "Rule-Based District": rule_based_district_policy,
        }
        rows: list[dict] = []
        for name, policy_fn in policy_map.items():
            env = make_env(
                data,
                args.env_type,
                episode_length,
                args.move_qty,
                args.k_candidates,
                args.seed,
                args.trucks,
                args.truck_capacity,
                local_policy_mode=args.local_policy_mode,
                local_model_dir=args.local_model_dir,
                local_decisions_per_global_step=args.local_decisions_per_global_step,
            )
            rows.append(evaluate_policy(env, policy_fn, n_episodes=args.n_episodes, policy_name=name, seed=args.seed))
    else:
        policy_map = {
            "No Rebalancing": no_rebalancing_policy,
            "Random": random_policy,
            "Greedy": greedy_policy,
            "Rule-Based": rule_based_policy,
        }
        rows = []
        for name, policy_fn in policy_map.items():
            env = make_env(
                data,
                args.env_type,
                episode_length,
                args.move_qty,
                args.k_candidates,
                args.seed,
                args.trucks,
                args.truck_capacity,
            )
            rows.append(evaluate_policy(env, policy_fn, n_episodes=args.n_episodes, policy_name=name, seed=args.seed))

    if not args.compare_hierarchical:
        model_path = resolve_dqn_model_path(args)
        if model_path is not None:
            try:
                dqn_policy = _dqn_policy_from_path(model_path)
                env = make_env(
                    data,
                    args.env_type,
                    episode_length,
                    args.move_qty,
                    args.k_candidates,
                    args.seed,
                    args.trucks,
                    args.truck_capacity,
                    local_policy_mode=args.local_policy_mode,
                    local_model_dir=args.local_model_dir,
                    local_decisions_per_global_step=args.local_decisions_per_global_step,
                )
                dqn_name = "Global DQN + Greedy Local" if args.env_type == "district" else "DQN"
                rows.append(evaluate_policy(env, dqn_policy, n_episodes=args.n_episodes, policy_name=dqn_name, seed=args.seed))
            except Exception as exc:
                print(f"[Evaluate] DQN 모델 평가를 건너뜁니다. env/model 호환성을 확인하세요: {exc}")
        else:
            print("[Evaluate] DQN 모델이 없어 baseline만 평가합니다.")

        result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("평가할 정책 결과가 없습니다. 모델 경로와 env 설정을 확인하세요.")

    result = add_comparison_metrics(result)
    ordered_columns = [
        "policy",
        "total_reward",
        "total_served_rentals",
        "total_lost_rentals",
        "total_accepted_returns",
        "total_overflow_returns",
        "rental_satisfaction_rate",
        "return_acceptance_rate",
        "total_reposition_distance_km",
        "total_moved_bikes",
        "saved_rentals_vs_no_rebalancing",
        "lost_rentals_improvement_vs_no_rebalancing_pct",
        "reward_improvement_vs_no_rebalancing_pct",
        "distance_per_saved_rental",
        "local_dqn_used_count",
        "local_greedy_fallback_count",
    ]
    result = result[[col for col in ordered_columns if col in result.columns]]
    results_dir = Path(args.results_dir)
    tables_dir = ensure_dir(results_dir / "tables")
    figures_dir = ensure_dir(Path(args.results_dir) / "figures")
    experiment_table, figure_prefix = experiment_output_paths(results_dir, args.experiment_name)

    result.to_csv(experiment_table, index=False, encoding="utf-8-sig")
    latest_table = tables_dir / "baseline_comparison.csv"
    result.to_csv(latest_table, index=False, encoding="utf-8-sig")

    save_evaluation_plots(result, figures_dir, figure_prefix)

    print(f"[Evaluate] 평가 결과 저장 완료: {experiment_table}")
    print(f"[Evaluate] 최신 결과 저장 완료: {latest_table}")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    main()
