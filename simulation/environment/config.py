import pathlib

import yaml

CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yml"


class Config(dict):
  # get and set attr to make config like namespace, instead of config["sim"]["num"] , config.sim.num is possible
  def __getattr__(self, key):
    try:
      return self[key]
    except KeyError as e:
      raise AttributeError(key) from e

  def __setattr__(self, key, value):
    self[key] = value

  # can initialize a nested class and so used for nested configs
  @classmethod
  def nested(cls, obj):
    if isinstance(obj, dict):
      return cls({k: cls.nested(v) for k, v in obj.items()})
    if isinstance(obj, list):
      return [cls.nested(v) for v in obj]
    return obj


# loads default.yml , else can also take path and update, or a dict of overrides and update
def load_config(path=None, overrides=None):
  with open(DEFAULT_CONFIG) as f:
    raw = yaml.safe_load(f) or {}
  if path:
    with open(path) as f:
      _update_yaml(raw, yaml.safe_load(f) or {})
  if overrides:
    _update_yaml(raw, overrides)
  cfg = Config.nested(raw)
  _validate(cfg)
  return cfg


# Present so it updates only the yaml block that we pass
def _update_yaml(base, override):
  for k, v in override.items():
    if isinstance(v, dict) and isinstance(base.get(k), dict):
      _update_yaml(base[k], v)
    else:
      base[k] = v


# Validate the config, the rates have to divide each other cleanly
def _validate(cfg):
  sim_hz = 1.0 / cfg.sim.timestep
  ik_hz = cfg.control.ik.hz
  if abs(sim_hz / ik_hz - round(sim_hz / ik_hz)) > 1e-9:
    raise ValueError(f"control.ik.hz={ik_hz} must divide sim rate {sim_hz}")
  if ik_hz % cfg.control.hz != 0:
    raise ValueError(f"control.hz={cfg.control.hz} must divide control.ik.hz={ik_hz}")
  if cfg.control.hz % cfg.cameras.record_hz != 0:
    raise ValueError(f"cameras.record_hz={cfg.cameras.record_hz} must divide control.hz={cfg.control.hz}")
  unknown = set(cfg.task.obstacles.kinds) - {"wall", "box", "cylinder"}
  if unknown:
    raise ValueError(f"unknown task.obstacles.kinds: {sorted(unknown)}")
