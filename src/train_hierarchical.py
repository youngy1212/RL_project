"""Run the hierarchical bike rebalancing training pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_stage(name: str, cmd: list[str]) -> None:
    """Run one subprocess stage with clear logging."""
    # 각 단계를 subprocess로 실행해 단독 스크립트 CLI와 전체 파이프라인 CLI를 동일하게 유지한다.
    print(f"\n[Hierarchical] START: {name}")
    print("[Hierarchical] CMD:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"계층형 파이프라인 실패 단계: {name}, returncode={exc.returncode}") from exc
    print(f"[Hierarchical] DONE: {name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Local DQN과 Global DQN을 순차 학습하는 계층형 파이프라인입니다.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--local-total-timesteps", type=int, default=100_000)
    parser.add_argument("--global-total-timesteps", type=int, default=500_000)
    parser.add_argument("--episode-length", type=int, default=96)
    parser.add_argument("--move-qty", type=int, default=10)
    parser.add_argument("--k-candidates", type=int, default=5)
    parser.add_argument("--trucks", type=int, default=10)
    parser.add_argument("--truck-capacity", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-local-training", action="store_true")
    parser.add_argument("--skip-global-greedy", action="store_true")
    parser.add_argument("--skip-global-local-dqn", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full hierarchical pipeline after preprocessing."""
    args = parse_args(argv)
    python = sys.executable
    model_dir = Path(args.model_dir)
    local_model_dir = model_dir / "local_dqn"

    if not args.skip_local_training:
        # 1단계: 자치구별 Local DQN을 먼저 학습한다. 이후 Global 학습에서는 frozen policy로만 사용한다.
        run_stage(
            "Local DQN 전체 학습",
            [
                python,
                "src/train_local_dqn.py",
                "--processed-dir",
                args.processed_dir,
                "--all-districts",
                "--total-timesteps",
                str(args.local_total_timesteps),
                "--episode-length",
                str(args.episode_length),
                "--move-qty",
                str(args.move_qty),
                "--k-candidates",
                str(args.k_candidates),
                "--model-dir",
                str(local_model_dir),
                "--results-dir",
                args.results_dir,
                "--seed",
                str(args.seed),
            ],
        )

    # 2단계: Local DQN이 Local Greedy 대비 어떤 성능인지 자치구별로 평가한다.
    run_stage(
        "Local DQN 평가",
        [
            python,
            "src/evaluate_local_dqn.py",
            "--processed-dir",
            args.processed_dir,
            "--all-districts",
            "--move-qty",
            str(args.move_qty),
            "--k-candidates",
            str(args.k_candidates),
            "--model-dir",
            str(local_model_dir),
            "--results-dir",
            args.results_dir,
            "--seed",
            str(args.seed),
        ],
    )

    if not args.skip_global_greedy:
        # 3단계: 1차 구조인 Global DQN + Greedy Local을 학습한다.
        run_stage(
            "Global DQN + Local Greedy 학습",
            [
                python,
                "src/train_dqn.py",
                "--processed-dir",
                args.processed_dir,
                "--model-dir",
                args.model_dir,
                "--results-dir",
                args.results_dir,
                "--env-type",
                "district",
                "--local-policy-mode",
                "greedy",
                "--total-timesteps",
                str(args.global_total_timesteps),
                "--episode-length",
                str(args.episode_length),
                "--trucks",
                str(args.trucks),
                "--truck-capacity",
                str(args.truck_capacity),
                "--seed",
                str(args.seed),
            ],
        )

    if not args.skip_global_local_dqn:
        # 4단계: Frozen Local DQN이 포함된 환경에서 Global DQN을 학습한다.
        run_stage(
            "Global DQN + Frozen Local DQN 학습",
            [
                python,
                "src/train_dqn.py",
                "--processed-dir",
                args.processed_dir,
                "--model-dir",
                args.model_dir,
                "--results-dir",
                args.results_dir,
                "--env-type",
                "district",
                "--local-policy-mode",
                "dqn",
                "--local-model-dir",
                str(local_model_dir),
                "--total-timesteps",
                str(args.global_total_timesteps),
                "--episode-length",
                str(args.episode_length),
                "--trucks",
                str(args.trucks),
                "--truck-capacity",
                str(args.truck_capacity),
                "--seed",
                str(args.seed),
            ],
        )

    # 5단계: greedy 조합과 DQN 조합을 한 번에 비교해 최종 결과표를 만든다.
    run_stage(
        "Hierarchical 정책 평가",
        [
            python,
            "src/evaluate.py",
            "--processed-dir",
            args.processed_dir,
            "--env-type",
            "district",
            "--compare-hierarchical",
            "--local-model-dir",
            str(local_model_dir),
            "--episode-length",
            "999999",
            "--trucks",
            str(args.trucks),
            "--truck-capacity",
            str(args.truck_capacity),
            "--results-dir",
            args.results_dir,
            "--experiment-name",
            "hierarchical",
            "--seed",
            str(args.seed),
        ],
    )


if __name__ == "__main__":
    main()
