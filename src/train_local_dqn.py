"""Train frozen Local DQN policies for each Seoul district."""

from __future__ import annotations

import argparse
import json
import os
import traceback
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
from utils import ensure_dir, load_time_features, set_global_seed

configure_korean_font()


def parse_net_arch(value: str) -> list[int]:
    """Parse a comma-separated network architecture string."""
    return [int(part.strip()) for part in value.split(",") if part.strip()]


class RewardLoggerCallback:
    """Stable-Baselines3 callback collecting completed episode rewards."""

    def __init__(self) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.rows: list[dict[str, float]] = []

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    if "episode" in info:
                        episode = info["episode"]
                        self.rows.append(
                            {
                                "timesteps": float(self.num_timesteps),
                                "episode_reward": float(episode["r"]),
                                "episode_length": float(episode["l"]),
                            }
                        )
                return True

        self.callback = _Callback()


def load_processed_train(processed_dir: str) -> dict[str, object]:
    """Load arrays and station metadata required for local training."""
    base = Path(processed_dir)
    required = ["rental_train.npy", "return_train.npy", "capacity.npy", "distance_matrix.npy", "station_meta.csv"]
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Local DQN 학습에 필요한 processed 파일이 없습니다: {missing}")
    station_meta = pd.read_csv(base / "station_meta.csv", encoding="utf-8-sig")
    require_district_columns(station_meta)
    return {
        "rental": np.load(base / "rental_train.npy"),
        "returns": np.load(base / "return_train.npy"),
        "capacity": np.load(base / "capacity.npy"),
        "distance": np.load(base / "distance_matrix.npy"),
        "station_meta": station_meta,
        "time_features": load_time_features(base / "time_index_train.csv"),
    }


def district_demand_table(station_meta: pd.DataFrame, rental: np.ndarray, returns: np.ndarray) -> pd.DataFrame:
    """Build district-level train demand and station count table."""
    frame = station_meta[["district_id", "district_name"]].copy()
    frame["train_total_rental_demand"] = rental.sum(axis=0)
    frame["train_total_return_demand"] = returns.sum(axis=0)
    grouped = (
        frame.groupby(["district_id", "district_name"], as_index=False)
        .agg(
            station_count=("district_id", "size"),
            train_total_rental_demand=("train_total_rental_demand", "sum"),
            train_total_return_demand=("train_total_return_demand", "sum"),
        )
        .sort_values("district_id")
        .reset_index(drop=True)
    )
    grouped["total_demand"] = grouped["train_total_rental_demand"] + grouped["train_total_return_demand"]
    return grouped


def select_districts(args: argparse.Namespace, table: pd.DataFrame) -> list[int]:
    """Select target district IDs from CLI options."""
    if args.district_ids:
        return [int(part.strip()) for part in args.district_ids.split(",") if part.strip()]
    if args.all_districts or args.top_districts is None:
        return table["district_id"].astype(int).tolist()
    return table.sort_values("total_demand", ascending=False).head(int(args.top_districts))["district_id"].astype(int).tolist()


def save_reward_plot(rows: list[dict[str, float]], output_path: Path, title: str) -> None:
    """Save a reward curve for one local model."""
    log = pd.DataFrame(rows)
    plt.figure(figsize=(8, 4.5))
    if not log.empty:
        plt.plot(log["timesteps"], log["episode_reward"], alpha=0.55, label="Episode reward")
        if len(log) >= 5:
            plt.plot(log["timesteps"], log["episode_reward"].rolling(5, min_periods=1).mean(), linewidth=2, label="Rolling mean")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No completed episodes logged", ha="center", va="center")
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for local DQN training."""
    parser = argparse.ArgumentParser(description="자치구별 Local DQN을 독립적으로 학습합니다.")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--district-ids", default=None)
    parser.add_argument("--all-districts", action="store_true")
    parser.add_argument("--top-districts", type=int, default=None)
    parser.add_argument("--min-stations", type=int, default=2)
    parser.add_argument("--min-demand", type=float, default=1.0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--episode-length", type=int, default=96)
    parser.add_argument("--move-qty", type=int, default=10)
    parser.add_argument("--k-candidates", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--exploration-fraction", type=float, default=0.4)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--net-arch", default="128,128")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", default="models/local_dqn")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Train Local DQN models for selected districts and save a summary."""
    args = parse_args(argv)
    set_global_seed(args.seed)
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor

    data = load_processed_train(args.processed_dir)
    rental = data["rental"]
    returns = data["returns"]
    station_meta = data["station_meta"]
    demand_table = district_demand_table(station_meta, rental, returns)
    target_ids = select_districts(args, demand_table)

    model_dir = ensure_dir(args.model_dir)
    tables_dir = ensure_dir(Path(args.results_dir) / "tables")
    figures_dir = ensure_dir(Path(args.results_dir) / "figures")
    summary_rows: list[dict[str, object]] = []

    for district_id in target_ids:
        row = demand_table.loc[demand_table["district_id"].astype(int).eq(int(district_id))]
        district_name = str(row["district_name"].iloc[0]) if not row.empty else str(district_id)
        station_count = int(row["station_count"].iloc[0]) if not row.empty else 0
        total_demand = float(row["total_demand"].iloc[0]) if not row.empty else 0.0
        print(f"[Local Train] district_id={district_id}, district_name={district_name}, stations={station_count}")
        base_summary = {
            "district_id": int(district_id),
            "district_name": district_name,
            "station_count": station_count,
            "total_demand": total_demand,
            "model_path": "",
            "final_mean_reward": np.nan,
            "total_timesteps": int(args.total_timesteps),
        }
        if station_count < args.min_stations:
            summary_rows.append({**base_summary, "status": "skipped", "reason": "station_count < min_stations"})
            continue
        if total_demand < args.min_demand:
            summary_rows.append({**base_summary, "status": "skipped", "reason": "total_demand < min_demand"})
            continue
        try:
            env = LocalDistrictBikeEnv(
                rental,
                returns,
                data["capacity"],
                data["distance"],
                station_meta,
                int(district_id),
                time_features=data["time_features"],
                episode_length=args.episode_length,
                move_qty=args.move_qty,
                k_candidates=args.k_candidates,
                random_start=True,
                seed=args.seed + int(district_id),
            )
            env = Monitor(env)
            model = DQN(
                policy="MlpPolicy",
                env=env,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                gamma=args.gamma,
                exploration_fraction=args.exploration_fraction,
                exploration_final_eps=args.exploration_final_eps,
                target_update_interval=1000,
                train_freq=4,
                gradient_steps=1,
                policy_kwargs={"net_arch": parse_net_arch(args.net_arch)},
                verbose=1,
                seed=args.seed + int(district_id),
                device="cpu",
            )
            logger = RewardLoggerCallback()
            model.learn(total_timesteps=args.total_timesteps, callback=logger.callback)
            model_path = model_dir / f"local_dqn_district_{int(district_id)}.zip"
            model.save(model_path)

            log = pd.DataFrame(logger.callback.rows)
            log_path = tables_dir / f"local_training_log_district_{int(district_id)}.csv"
            log.to_csv(log_path, index=False, encoding="utf-8-sig")
            save_reward_plot(
                logger.callback.rows,
                figures_dir / f"local_dqn_training_reward_district_{int(district_id)}.png",
                f"Local DQN Training Reward - {district_name}",
            )
            final_mean = float(log["episode_reward"].tail(10).mean()) if not log.empty else np.nan
            metadata = {
                "district_id": int(district_id),
                "district_name": district_name,
                "station_count": station_count,
                "train_total_rental_demand": float(row["train_total_rental_demand"].iloc[0]),
                "train_total_return_demand": float(row["train_total_return_demand"].iloc[0]),
                "total_timesteps": int(args.total_timesteps),
                "episode_length": int(args.episode_length),
                "move_qty": int(args.move_qty),
                "k_candidates": int(args.k_candidates),
                "observation_dim": int(env.observation_space.shape[0]),
                "action_dim": int(env.action_space.n),
                "seed": int(args.seed),
            }
            with (model_dir / f"local_dqn_district_{int(district_id)}.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            summary_rows.append(
                {
                    **base_summary,
                    "status": "trained",
                    "reason": "",
                    "model_path": str(model_path),
                    "final_mean_reward": final_mean,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            summary_rows.append({**base_summary, "status": "failed", "reason": repr(exc)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "local_dqn_training_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[Local Train] summary 저장 완료: {tables_dir / 'local_dqn_training_summary.csv'}")
    return summary


if __name__ == "__main__":
    main()
