"""Frozen Local DQN policy loader for hierarchical global environments."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LocalPolicyManager:
    """Lazy-loading cache for per-district Local DQN models."""

    def __init__(self, model_dir: str | Path, deterministic: bool = True, preload: bool = False) -> None:
        self.model_dir = Path(model_dir)
        self.deterministic = bool(deterministic)
        # Global env가 매 step 모델을 다시 로드하면 매우 느리므로 한 번 로드한 모델은 캐시한다.
        self._models: dict[int, object] = {}
        if preload:
            # preload=True이면 시작 시 모든 자치구 모델을 미리 메모리에 올린다.
            for path in self.model_dir.glob("local_dqn_district_*.zip"):
                district_id = self._district_id_from_path(path)
                if district_id is not None:
                    self._models[district_id] = self._load_model(path)

    def _district_id_from_path(self, path: Path) -> int | None:
        stem = path.stem
        prefix = "local_dqn_district_"
        if not stem.startswith(prefix):
            return None
        try:
            return int(stem[len(prefix) :])
        except ValueError:
            return None

    def _path_for(self, district_id: int) -> Path:
        return self.model_dir / f"local_dqn_district_{int(district_id)}.zip"

    def _load_model(self, path: Path) -> object:
        from stable_baselines3 import DQN

        # Local DQN은 Global 학습 중 frozen policy로만 쓰며, CPU에서 predict만 수행한다.
        return DQN.load(path, device="cpu")

    def has_model(self, district_id: int) -> bool:
        """Return whether a local model exists or is already cached."""
        return int(district_id) in self._models or self._path_for(int(district_id)).exists()

    def predict(self, district_id: int, obs: np.ndarray) -> int | None:
        """Return a local action from a frozen model, or None when unavailable."""
        district_id = int(district_id)
        if not self.has_model(district_id):
            return None
        if district_id not in self._models:
            # lazy loading: 실제로 해당 district가 선택될 때 처음 모델을 로드한다.
            self._models[district_id] = self._load_model(self._path_for(district_id))
        model = self._models[district_id]
        # 여기서는 learn()을 절대 호출하지 않고 deterministic predict만 사용한다.
        action, _ = model.predict(obs, deterministic=self.deterministic)
        return int(action)

    def k_candidates(self, district_id: int) -> int | None:
        """Infer local k_candidates from the model action space."""
        district_id = int(district_id)
        if not self.has_model(district_id):
            return None
        if district_id not in self._models:
            self._models[district_id] = self._load_model(self._path_for(district_id))
        # Local action space는 1 + k*k 구조이므로 action 개수에서 k_candidates를 역산한다.
        action_n = int(getattr(self._models[district_id].action_space, "n", 1))
        k = int(round(np.sqrt(max(0, action_n - 1))))
        if 1 + k * k != action_n:
            return None
        return k
