"""主升概率预测模型（技术方案 Phase 4 MVP / §10 模型 E 基线）。

从日线库构建「特征 → 前瞻 20 日收益是否达到主升阈值」的监督数据集，
优先用 scikit-learn HistGradientBoostingClassifier（GBDT 基线），环境无
sklearn 时回退到内置 numpy L2 逻辑回归，保证本地与容器均可运行。
模型持久化到 ``data/models/main_rally.pkl``，盘前报告的
P(未来20交易日进入主升阶段) 由该模型输出，替代评分近似映射。
"""

from __future__ import annotations

import logging
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .database import Database
from .strategies.factors import compute_factor

LOG = logging.getLogger(__name__)

MODEL_FILE = "main_rally.pkl"

# 特征集：全部来自日线 OHLCV + 慢变宏观分，实盘候选与训练数据共用同一口径
FEATURE_NAMES = [
    "momentum", "reversal", "volatility", "liquidity", "rsi", "macd",
    "ma_trend", "high_52w", "atr", "amihud", "vwap_dev", "amount_z", "vol_ratio",
]

# 模型注册表：训练/预测共用的因子参数（键 = {因子名}_{参数名}）
FACTOR_PARAMS: dict[str, Any] = {
    "momentum_window": 60, "reversal_window": 5, "volatility_window": 20,
    "liquidity_window": 20, "rsi_window": 14, "macd_fast": 12, "macd_slow": 26,
    "macd_signal": 9, "high_52w_window": 250, "atr_window": 14, "amihud_window": 20,
}
_MODEL_FACTORS = ["momentum", "reversal", "volatility", "liquidity", "rsi",
                  "macd", "ma_trend", "high_52w", "atr", "amihud"]


def feature_vector_from_frame(frame: pd.DataFrame) -> dict[str, float] | None:
    """从一段日线数据（最后一行为最新交易日）计算模型特征向量。"""
    if frame is None or frame.empty or len(frame) < 61:
        return None
    features: dict[str, float] = {}
    for name in _MODEL_FACTORS:
        value = compute_factor(name, frame, FACTOR_PARAMS)
        if value is None or not math.isfinite(value):
            return None
        features[name] = float(value)
    close = frame["close"].astype(float)
    amount = frame["amount"].astype(float)
    volume = frame["volume"].astype(float)
    tail = min(20, len(frame))
    vwap = float(amount.tail(tail).sum() / max(volume.tail(tail).sum(), 1e-9))
    features["vwap_dev"] = float(close.iloc[-1]) / vwap - 1.0 if vwap > 0 else 0.0
    amounts = amount.tail(tail)
    std = float(amounts.std(ddof=0))
    mean = float(amounts.mean())
    features["amount_z"] = (float(amounts.iloc[-1]) - mean) / std if std > 0 else 0.0
    avg_volume = float(volume.tail(tail).mean())
    features["vol_ratio"] = float(volume.iloc[-1]) / avg_volume if avg_volume > 0 else 0.0
    return features


def build_dataset(
    database: Database, data_service: Any, sample_symbols: int = 300,
    forward_days: int = 20, rally_threshold: float = 0.10, step: int = 5,
) -> tuple[list[list[float]], list[int], list[str], int]:
    """构建训练数据集：周度网格采样 (code, date) → (特征, 是否进入主升)。

    返回 (X, y, dates, used_symbols)。进入主升的 V1 定义：未来 forward_days
    个交易日收盘涨幅 ≥ rally_threshold（可按样本外回测调整）。
    """
    universe = data_service.eligible_universe()
    codes = sorted(universe["code"].tolist())
    if not codes:
        raise RuntimeError("候选池为空，无法构建训练数据")
    step_codes = max(1, len(codes) // max(1, sample_symbols))
    sampled = codes[::step_codes][:sample_symbols]
    start = str(database.query_one("SELECT MIN(trade_date) AS d FROM daily_bars")["d"] or "2020-01-01")
    bars_by_code = data_service.load_bars_many(sampled, start_date=start)
    X: list[list[float]] = []
    y: list[int] = []
    dates: list[str] = []
    for frame in bars_by_code.values():
        n = len(frame)
        if n < 130 + forward_days:
            continue
        for i in range(130, n - forward_days, step):
            window = frame.iloc[max(0, i - 300):i + 1]
            features = feature_vector_from_frame(window)
            if features is None:
                continue
            base = float(frame["close"].iloc[i])
            future = float(frame["close"].iloc[i + forward_days])
            if base <= 0:
                continue
            X.append([features[name] for name in FEATURE_NAMES])
            y.append(1 if future / base - 1.0 >= rally_threshold else 0)
            dates.append(str(frame["trade_date"].iloc[i])[:10])
    return X, y, dates, len(bars_by_code)


def _auc(y_true: list[int], proba: np.ndarray) -> float:
    series = pd.Series(np.asarray(proba, dtype=float))
    ranks = series.rank(method="average").to_numpy()
    pos = np.array(y_true, dtype=bool)
    n_pos = int(pos.sum())
    n_neg = int(len(pos) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _train_sklearn(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, min_samples_leaf=50,
        l2_regularization=1.0, random_state=7,
    )
    model.fit(X_train, y_train)
    return {"type": "sklearn", "model": model}


def _train_numpy(X_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    """内置 L2 逻辑回归（全批量梯度下降），sklearn 不可用时的兜底。"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    Z = (X_train - mean) / std
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.zeros(Z.shape[1])
    target = y_train.astype(float)
    lr, l2 = 0.1, 1e-3
    for _ in range(1500):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        gradient = Z.T @ (p - target) / len(Z) + l2 * w
        w -= lr * gradient
    return {"type": "numpy", "weights": w[:-1].tolist(), "bias": float(w[-1]),
            "mean": mean.tolist(), "std": std.tolist()}


def _predict_proba(bundle: dict[str, Any], X: np.ndarray) -> np.ndarray:
    if bundle["type"] == "sklearn":
        return bundle["model"].predict_proba(X)[:, 1]
    mean = np.asarray(bundle["mean"])
    std = np.asarray(bundle["std"])
    std[std == 0] = 1.0
    Z = (X - mean) / std
    w = np.asarray(bundle["weights"])
    return 1.0 / (1.0 + np.exp(-np.clip(Z @ w + bundle["bias"], -30, 30)))


def _model_path(settings: Any) -> Path:
    return Path(settings.db_path).parent / "models" / MODEL_FILE


def train_and_save(
    database: Database, X: list[list[float]], y: list[int], dates: list[str], settings: Any,
) -> dict[str, Any]:
    """按时间切分训练并持久化模型，返回训练指标。"""
    if len(X) < 500 or len(set(y)) < 2:
        raise RuntimeError(f"训练样本不足或只有单一类别（样本 {len(X)}），请先补齐日线数据")
    order = np.argsort(np.asarray(dates))
    X_arr = np.asarray(X, dtype=float)[order]
    y_arr = np.asarray(y, dtype=int)[order]
    split = int(len(X_arr) * 0.8)
    X_train, y_train = X_arr[:split], y_arr[:split]
    X_valid, y_valid = X_arr[split:], y_arr[split:]
    try:
        bundle = _train_sklearn(X_train, y_train)
        backend = "sklearn-HistGB"
    except ImportError:
        bundle = _train_numpy(X_train, y_train)
        backend = "numpy-logistic"
    proba = _predict_proba(bundle, X_valid) if len(X_valid) else np.array([])
    auc = _auc(y_valid.tolist(), proba) if len(X_valid) else 0.5
    accuracy = float(((proba >= 0.5) == (y_valid == 1)).mean()) if len(X_valid) else 0.0
    bundle.update({
        "feature_names": FEATURE_NAMES, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": {"samples": len(X_arr), "positives": int(y_arr.sum()),
                    "validation_auc": round(auc, 4), "validation_accuracy": round(accuracy, 4),
                    "backend": backend},
    })
    path = _model_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    database.execute(
        "INSERT INTO model_runs(trained_at,samples,positive_rate,validation_auc,validation_accuracy,backend,parameters_json) "
        "VALUES(?,?,?,?,?,?,?)",
        (bundle["trained_at"], len(X_arr), round(float(y_arr.mean()), 4), round(auc, 4),
         round(accuracy, 4), backend, "{}"),
    )
    LOG.info("主升概率模型训练完成：backend=%s 样本=%d AUC=%.4f", backend, len(X_arr), auc)
    return bundle["metrics"]


def load_model(settings: Any) -> dict[str, Any] | None:
    path = _model_path(settings)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as error:
        LOG.warning("主升概率模型加载失败：%s", error)
        return None


def model_info(settings: Any) -> str:
    bundle = load_model(settings)
    if not bundle:
        return "未训练（使用评分近似）"
    metrics = bundle.get("metrics", {})
    return (f"{metrics.get('backend', bundle['type'])} · 验证AUC {metrics.get('validation_auc', '-')} · "
            f"样本 {metrics.get('samples', '-')} · 训练于 {str(bundle.get('trained_at', ''))[:10]}")


def predict_main_rally_probability(settings: Any, features: dict[str, float] | None) -> float | None:
    """用已训练模型输出 P(未来20交易日进入主升)，无模型或特征缺失返回 None。"""
    if not features:
        return None
    bundle = load_model(settings)
    if not bundle:
        return None
    vector = []
    for name in bundle.get("feature_names", FEATURE_NAMES):
        value = features.get(name)
        if value is None or not math.isfinite(value):
            return None
        vector.append(float(value))
    proba = _predict_proba(bundle, np.asarray([vector]))
    return float(np.clip(proba[0], 0.0, 1.0))


def retrain(database: Database, data_service: Any, settings: Any) -> dict[str, Any]:
    """收盘后重训入口：构建数据集 → 训练 → 落库。"""
    sample = int(settings.raw.get("model", {}).get("retrain_sample_symbols", 300) or 300)
    X, y, dates, used = build_dataset(database, data_service, sample_symbols=sample)
    metrics = train_and_save(database, X, y, dates, settings)
    return {**metrics, "symbols": used, "sample_symbols": sample}
