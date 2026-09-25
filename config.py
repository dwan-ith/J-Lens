"""
Central config for J-Space safety study.

Single source of truth for model, lens, experiment params.
No hardcoding elsewhere — all modules import from here.
Env var overrides + YAML + CLI supported for Thinking Machines Inkling/Tinker portability.

Usage:
  from config import get_config, MODEL_DIR, ANALYSIS_LAYERS
  cfg = get_config()  # or get_config(model_id="Qwen/Qwen3-8B")
  # or: cfg = ExperimentConfig.from_yaml("configs/inkling.yaml")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any

# ─── Root ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
MODEL_DIR_ENV = os.environ.get("JSPACE_MODEL_DIR")
LENS_DIR_ENV = os.environ.get("JSPACE_LENS_DIR")

DEFAULT_MODEL_DIR = Path(MODEL_DIR_ENV) if MODEL_DIR_ENV else (ROOT / "model")
DEFAULT_LENS_DIR = Path(LENS_DIR_ENV) if LENS_DIR_ENV else (ROOT / "lens_weights" / "qwen3-1.7b" / "jlens" / "Salesforce-wikitext")
DEFAULT_LENS_FILE = DEFAULT_LENS_DIR / "Qwen3-1.7B_jacobian_lens.pt"

# ─── Model registry (Thinking Machines + open) ───────────────────────────────

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen3-1.7b": {
        "hf_id": "Qwen/Qwen3-1.7B",
        "local_dir": str(DEFAULT_MODEL_DIR),
        "lens": str(DEFAULT_LENS_FILE),
        "n_layers": 28, "d_model": 2048, "vocab_size": 151936,
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "local_dir": None,
        "lens_repo": "neuronpedia/jacobian-lens",
        "lens_file": "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
        "n_layers": 36, "d_model": 4096, "vocab_size": 151936,
    },
    "llama-3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B",
        "local_dir": None,
        "lens_repo": "neuronpedia/jacobian-lens",
        "lens_file": "llama-3.2-3b/jlens/Salesforce-wikitext/Llama-3.2-3B_jacobian_lens.pt",
        "n_layers": 28, "d_model": 3072, "vocab_size": 128256,
    },
    "inkling": {
        "hf_id": os.environ.get("INKLING_MODEL_ID", "thinking-machines/inkling"),
        "local_dir": os.environ.get("INKLING_MODEL_DIR"),
        "lens": None,
        "n_layers": None, "d_model": None,  # auto-detect via model.config
        "api_base": os.environ.get("INKLING_API_BASE"),
        "api_key_env": "INKLING_API_KEY",
    },
    "inkling-small": {
        "hf_id": os.environ.get("INKLING_SMALL_MODEL_ID", "thinking-machines/inkling-small"),
        "local_dir": None,
        "lens": None,
        "n_layers": None, "d_model": None,
        "api_base": os.environ.get("INKLING_API_BASE"),
        "api_key_env": "INKLING_API_KEY",
    },
    "tinker": {
        "hf_id": os.environ.get("TINKER_MODEL_ID", "thinking-machines/tinker"),
        "local_dir": os.environ.get("TINKER_MODEL_DIR"),
        "lens": None,
        "n_layers": None, "d_model": None,
    },
}

# ─── Thresholds & statistical defaults ───────────────────────────────────────

@dataclass
class ThresholdConfig:
    """No magic numbers in analysis code — all thresholds here, with rationale."""
    # Heuristic deception thresholds (legacy, now data-driven via percentile; kept for ablation)
    # Defaults set to inf to disable heuristic classifier by default — use classify_heuristic_datadriven instead
    divergence_area: float = float("inf")
    mid_late_gap: float = float("inf")
    trajectory_var: float = float("inf")
    final_surprise: float = float("inf")
    suspicious_frac: float = 0.5  # need 2/4 signals
    uncertain_frac: float = 0.25  # need 1/4
    # Artifact criteria (proximity artifact vanishes if late < peak * ratio and peak > min)
    artifact_peak_min: float = 0.1
    artifact_late_ratio: float = 0.3
    artifact_advantage_floor: float = 0.1  # for crossover plot
    # Causal patching
    patch_alphas: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    high_divergence_cutoff: float = 0.5

@dataclass
class StatsConfig:
    n_boot: int = 5000
    n_perm: int = 5000
    ci_level: float = 0.95
    random_seeds: int = 3
    seed: int = 0

@dataclass
class PlotConfig:
    colors: Dict[str, str] = field(default_factory=lambda: {
        "in_dist": "#4CAF50", "safety": "#F44336", "adversarial": "#FF9800",
        "primary": "#2196F3", "secondary": "#FF9800", "accent": "#9C27B0",
    })
    figsize_single: tuple[int,int] = (10, 6)
    figsize_wide: tuple[int,int] = (14, 6)
    dpi: int = 150
    cmap_heatmap: str = "RdYlGn"
    cmap_cka: str = "viridis"
    cmap_overlap: str = "magma"

# ─── Experiment defaults ─────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    model_id: str = "qwen3-1.7b"
    max_seq_len: int = 96
    n_prompts_per_cat: int = 2
    n_rand_seeds: int = 3
    n_boot: int = 5000
    n_perm: int = 5000
    layer_fractions: list[float] = field(default_factory=lambda: [0.15, 0.30, 0.50, 0.70, 0.90])
    mid_layer_fraction: float = 0.50
    dtype: str = "float16"
    device: str = "cpu"
    seed: int = 0
    output_root: Path = ROOT / "results"
    local_dir: str | None = None
    lens: str | None = None
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    top_k_overlap: int = 32
    vocab_top_k: int = 15
    skip_first_positions: int = 16  # for Jacobian fitting, as in jlens.fitting.SKIP_FIRST_N_POSITIONS

    def analysis_layers(self, n_layers: int) -> list[int]:
        if n_layers <= 0:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if n_layers == 1:
            return [0]
        layers = sorted({min(n_layers - 1, max(0, int(round(n_layers * f)))) for f in self.layer_fractions})
        if len(layers) < 2:
            layers = [n_layers // 3, 2 * n_layers // 3]
        return layers

    def mid_layer_for(self, n_layers: int) -> int:
        if n_layers <= 0:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        return max(0, min(n_layers - 1, int(round(n_layers * self.mid_layer_fraction))))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        import yaml
        data = yaml.safe_load(Path(path).read_text())
        if not data:
            raise ValueError(f"empty config file: {path}")
        if "output_root" in data and isinstance(data["output_root"], str):
            data["output_root"] = Path(data["output_root"])
        # flatten nested
        if "thresholds" in data and isinstance(data["thresholds"], dict):
            data["thresholds"] = ThresholdConfig(**data["thresholds"])
        if "stats" in data and isinstance(data["stats"], dict):
            data["stats"] = StatsConfig(**data["stats"])
        if "plots" in data and isinstance(data["plots"], dict):
            data["plots"] = PlotConfig(**data["plots"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str | Path):
        import yaml
        data = asdict(self)
        # Path -> str for yaml (walk top level; nested dataclasses have no Paths)
        for k, v in data.items():
            if isinstance(v, Path):
                data[k] = str(v)
        Path(path).write_text(yaml.safe_dump(data))

# Global default
DEFAULT_CONFIG = ExperimentConfig()

def get_config(**overrides) -> ExperimentConfig:
    import copy
    import warnings
    # JSPACE_MODEL_ID wins over kwarg/default — resolve BEFORE registry lookup
    # so local_dir/lens belong to the right model
    if "JSPACE_MODEL_ID" in os.environ:
        overrides["model_id"] = os.environ["JSPACE_MODEL_ID"]
    # reject unknown kwargs early (typos would otherwise TypeError deep in the ctor)
    unknown = [k for k in overrides if k not in ExperimentConfig.__dataclass_fields__]
    if unknown:
        raise TypeError(f"get_config() got unexpected keyword(s): {unknown}")
    # Auto-populate local_dir and lens from MODEL_REGISTRY if not overridden
    model_id = overrides.get("model_id", "qwen3-1.7b")
    info = MODEL_REGISTRY.get(model_id, {})
    if model_id not in MODEL_REGISTRY:
        warnings.warn(f"unknown model_id {model_id!r} — local_dir/lens will be None")
    if "local_dir" not in overrides:
        overrides["local_dir"] = info.get("local_dir")
    if "lens" not in overrides:
        overrides["lens"] = info.get("lens")
    # deepcopy shared nested defaults — mutating one config must not leak into DEFAULT_CONFIG
    cfg = ExperimentConfig(
        **{**asdict(DEFAULT_CONFIG), **overrides,
           "thresholds": copy.deepcopy(overrides.get("thresholds", DEFAULT_CONFIG.thresholds)),
           "stats": copy.deepcopy(overrides.get("stats", DEFAULT_CONFIG.stats)),
           "plots": copy.deepcopy(overrides.get("plots", DEFAULT_CONFIG.plots))}
    )
    # handle nested overrides for thresholds/stats if passed as dict
    if isinstance(cfg.thresholds, dict):
        cfg.thresholds = ThresholdConfig(**cfg.thresholds)
    if isinstance(cfg.stats, dict):
        cfg.stats = StatsConfig(**cfg.stats)
    if isinstance(cfg.plots, dict):
        cfg.plots = PlotConfig(**cfg.plots)
    for attr in ("thresholds", "stats", "plots"):
        if getattr(cfg, attr) is None:
            raise TypeError(f"get_config() {attr}=None is not allowed — omit it for defaults")
    if isinstance(cfg.output_root, str):
        cfg.output_root = Path(cfg.output_root)
    # env overrides (guarded int parsing)
    def _env_int(name, default):
        if name in os.environ:
            try:
                return int(os.environ[name])
            except ValueError:
                warnings.warn(f"{name}={os.environ[name]!r} is not an int — keeping {default}")
        return default
    cfg.max_seq_len = _env_int("JSPACE_MAX_SEQ_LEN", cfg.max_seq_len)
    cfg.n_prompts_per_cat = _env_int("JSPACE_N_PROMPTS", cfg.n_prompts_per_cat)
    cfg.seed = _env_int("JSPACE_SEED", cfg.seed)
    if "JSPACE_DTYPE" in os.environ:
        cfg.dtype = os.environ["JSPACE_DTYPE"]
    return cfg

# ─── Convenience exports (for backward compat) ───────────────────────────────

MODEL_DIR = DEFAULT_MODEL_DIR
LENS_PATH = DEFAULT_LENS_FILE
ANALYSIS_LAYERS = DEFAULT_CONFIG.analysis_layers(28)  # valid for default qwen3-1.7b only
MAX_SEQ_LEN = DEFAULT_CONFIG.max_seq_len
N_PROMPTS_PER_CAT = DEFAULT_CONFIG.n_prompts_per_cat

def get_analysis_layers(n_layers: int = 28) -> list[int]:
    """Use this instead of ANALYSIS_LAYERS constant for non-default models."""
    return DEFAULT_CONFIG.analysis_layers(n_layers)
