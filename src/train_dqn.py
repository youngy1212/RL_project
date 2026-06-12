"""Train a Stable-Baselines3 DQN agent for bike rebalancing."""

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

from bike_env import BikeRebalancingEnv, DistrictBikeRebalancingEnv
from config import DEFAULT_MODEL_DIR, DEFAULT_PROCESSED_DIR, DEFAULT_RESULTS_DIR
from plot_utils import configure_korean_font
from utils import ensure_dir, load_time_features, set_global_seed

configure_korean_font()


def load_train_data(processed_dir: str) -> dict[str, np.ndarray]:
    """Load processed training arrays."""
    base = Path(processed_dir)
    required = [
        "rental_train.npy",
        "return_train.npy",
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
        "rental": np.load(base / "rental_train.npy"),
        "returns": np.load(base / "return_train.npy"),
        "capacity": np.load(base / "capacity.npy"),
        "distance": np.load(base / "distance_matrix.npy"),
        "time_features": load_time_features(base / "time_index_train.csv"),
    }
    if (base / "station_district.npy").exists():
        data["station_district"] = np.load(base / "station_district.npy")
    if (base / "district_meta.csv").exists():
        data["district_names"] = pd.read_csv(base / "district_meta.csv")["district_name"].astype(str).tolist()
    return data


def parse_net_arch(value: str) -> list[int]:
    """Parse comma-separated hidden layer sizes."""
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"--net-arch 형식이 올바르지 않습니다: {value}") from exc


class RewardLoggerCallback:
    """Small callback adapter that stores Monitor episode rewards."""

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


def save_training_outputs(
    rows: list[dict[str, float]],
    results_dir: str,
    log_filename: str = "training_log.csv",
    figure_filename: str = "dqn_training_reward.png",
    title: str = "DQN Training Reward",
) -> None:
    """Save training log CSV and reward curve figure."""
    tables_dir = ensure_dir(Path(results_dir) / "tables")
    figures_dir = ensure_dir(Path(results_dir) / "figures")
    log = pd.DataFrame(rows)
    log.to_csv(tables_dir / log_filename, index=False, encoding="utf-8-sig")
    if log_filename != "training_log.csv":
        log.to_csv(tables_dir / "training_log.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 4.5))
    if not log.empty:
        plt.plot(log["timesteps"], log["episode_reward"], label="Episode reward", alpha=0.55)
        if len(log) >= 5:
            rolling = log["episode_reward"].rolling(5, min_periods=1).mean()
            plt.plot(log["timesteps"], rolling, label="Rolling mean (5)", linewidth=2)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No completed episodes logged", ha="center", va="center")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figures_dir / figure_filename, dpi=150)
    if figure_filename != "dqn_training_reward.png":
        plt.savefig(figures_dir / "dqn_training_reward.png", dpi=150)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Stable-Baselines3 DQN으로 따릉이 재배치 정책을 학습합니다.")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--episode-length", type=int, default=48)
    parser.add_argument("--move-qty", type=int, default=3)
    parser.add_argument("--k-candidates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-type", choices=["station", "district"], default="station")
    parser.add_argument("--local-policy-mode", choices=["greedy", "dqn"], default="greedy")
    parser.add_argument("--local-model-dir", default="models/local_dqn")
    parser.add_argument("--local-decisions-per-global-step", type=int, default=None)
    parser.add_argument("--trucks", type=int, default=10)
    parser.add_argument("--truck-capacity", type=int, default=10)
    parser.add_argument("--lost-weight", type=float, default=1.0)
    parser.add_argument("--overflow-weight", type=float, default=0.5)
    parser.add_argument("--distance-weight", type=float, default=0.05)
    parser.add_argument("--imbalance-weight", type=float, default=0.1)
    parser.add_argument("--net-arch", default="256,256")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Train DQN and save the model and learning curve."""
    args = parse_args(argv)
    set_global_seed(args.seed)

    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor

    data = load_train_data(args.processed_dir)
    if args.env_type == "district":
        if "station_district" not in data or "district_names" not in data:
            raise FileNotFoundError(
                "district 환경에는 station_district.npy와 district_meta.csv가 필요합니다. "
                "preprocess.py --zone-mode district --use-all-stations를 먼저 실행하세요."
            )
        env = DistrictBikeRebalancingEnv(
            data["rental"],
            data["returns"],
            data["capacity"],
            data["distance"],
            data["station_district"],
            data["district_names"],
            time_features=data["time_features"],
            episode_length=args.episode_length,
            trucks=args.trucks,
            truck_capacity=args.truck_capacity,
            random_start=True,
            seed=args.seed,
            lost_weight=args.lost_weight,
            overflow_weight=args.overflow_weight,
            distance_weight=args.distance_weight,
            imbalance_weight=args.imbalance_weight,
            local_policy_mode=args.local_policy_mode,
            local_model_dir=args.local_model_dir,
            local_decisions_per_global_step=args.local_decisions_per_global_step,
        )
    else:
        env = BikeRebalancingEnv(
            data["rental"],
            data["returns"],
            data["capacity"],
            data["distance"],
            time_features=data["time_features"],
            episode_length=args.episode_length,
            move_qty=args.move_qty,
            k_candidates=args.k_candidates,
            random_start=True,
            seed=args.seed,
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
        policy_kwargs={"net_arch": parse_net_arch(args.net_arch)},
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        verbose=1,
        seed=args.seed,
        device="cpu",
    )
    reward_logger = RewardLoggerCallback()
    model.learn(total_timesteps=args.total_timesteps, callback=reward_logger.callback)

    model_dir = ensure_dir(args.model_dir)
    if args.env_type == "district":
        suffix = "local_dqn" if args.local_policy_mode == "dqn" else "local_greedy"
        model_path = model_dir / f"global_dqn_district_{suffix}.zip"
        log_filename = f"global_training_log_{suffix}.csv"
        figure_filename = f"global_dqn_training_reward_{suffix}.png"
        title = f"Global DQN Training Reward ({suffix})"
    else:
        model_path = model_dir / "dqn_bike_rebalancing.zip"
        log_filename = "training_log.csv"
        figure_filename = "dqn_training_reward.png"
        title = "DQN Training Reward"
    model.save(model_path)
    if model_path.name != "dqn_bike_rebalancing.zip":
        model.save(model_dir / "dqn_bike_rebalancing.zip")
    save_training_outputs(reward_logger.callback.rows, args.results_dir, log_filename, figure_filename, title)
    print(f"[Train] 모델 저장 완료: {model_path}")
    return model_path


if __name__ == "__main__":
    main()
