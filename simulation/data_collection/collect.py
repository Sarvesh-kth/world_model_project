import argparse
import multiprocessing as mp
import time
import numpy as np
from environment import PickPlaceEnv, load_config
from .scripted_policy import make_policy
from .writer import EpisodeWriter, append_index, next_episode_id

# Run one episode with the policy, recording frames into the writer if given
# stops when the env says so or when the policy has nothing left to do
# the physics runs much faster than real time, realtime=True sleeps so each control step takes 1/control.hz seconds
def run_episode(env, policy, writer=None, realtime=False):
  policy.reset()
  success = False
  collisions = 0
  t = 0
  step_time = 1.0 / env.cfg.control.hz
  while True:
    t0 = time.time()
    action = policy.act()
    obs, reward, terminated, truncated, info = env.step(action, give_up=policy.done)
    if writer is not None and t % env.frames_every == 0:
      pcs = None
      if env.cfg.cameras.pointcloud.enabled:
        pcs = env.point_clouds(frame=env.cfg.cameras.pointcloud.frame)
      writer.add_record(t, action, obs, reward, info, env.camera_poses(), env.render_all(), pcs)
    success |= info["success"]
    collisions += int(info["obstacle_contact"])
    t += 1
    if terminated or truncated:
      return {"steps": t, "success": success, "collisions": collisions}
    if realtime:
      time.sleep(max(0.0, step_time - (time.time() - t0)))

# Turn data.mix fractions into a shuffled list of policy kinds, one per episode
def allocate_kinds(mix, episodes, rng):
  kinds = sorted(mix)
  fracs = np.array([mix[k] for k in kinds], dtype=float)
  fracs /= fracs.sum()
  counts = np.floor(fracs * episodes).astype(int)
  # Hand the leftover episodes to the kinds that were rounded down the most
  for i in np.argsort(-(fracs * episodes - counts)):
    if counts.sum() >= episodes:
      break
    counts[i] += 1
  out = [k for k, c in zip(kinds, counts) for _ in range(int(c))]
  rng.shuffle(out)
  return out

# Collect one episode, success episodes get retried on fresh layouts until they actually succeed
def collect_one(env, writer, cfg, episode_id, seed, kind, holdout=False, realtime=False):
  attempts = cfg.data.success_attempts if kind == "success" else 1
  for attempt in range(attempts):
    ep_seed = seed + 10007 * attempt
    obs, _ = env.reset(seed=ep_seed, holdout=holdout)
    policy = make_policy(kind, env, np.random.default_rng(ep_seed), cfg)
    meta = policy.describe()
    meta.update(kind=kind, attempt=attempt + 1)
    writer.start_episode(env.layout, kind, ep_seed, obs["goal"], meta)
    result = run_episode(env, policy, writer, realtime=realtime)
    if kind != "success" or result["success"]:
      break
  _, row = writer.finish_episode(episode_id)
  return row

# Each worker process owns one env and one writer
_worker = {}

def _init_worker(cfg, out_dir, holdout, realtime):
  _worker["env"] = PickPlaceEnv(cfg)
  _worker["writer"] = EpisodeWriter(out_dir, cfg, write_index=False)
  _worker["cfg"] = cfg
  _worker["holdout"] = holdout
  _worker["realtime"] = realtime

def _run_task(task):
  episode_id, seed, kind = task
  return collect_one(_worker["env"], _worker["writer"], _worker["cfg"],
                     episode_id, seed, kind, _worker["holdout"], _worker["realtime"])

# Collect a dataset, appends to out_dir if it already has episodes
def collect(cfg, episodes, out_dir, seed=None, workers=None, holdout=False, realtime=False):
  workers = cfg.data.workers if workers is None else workers
  start_id = next_episode_id(out_dir)
  base_seed = (cfg.seed if seed is None else seed) + 977 * start_id
  rng = np.random.default_rng(base_seed)
  kinds = allocate_kinds(cfg.data.mix, episodes, rng)
  tasks = [(start_id + i, int(rng.integers(2**30)), kinds[i]) for i in range(episodes)]

  start = time.time()
  rows = []

  def report(row):
    rows.append(row)
    print(f"[{len(rows)}/{episodes}] ep {row['episode']:06d} {row['policy']:9s} "
          f"{row['object']:9s} {row['pick_end']}->{row['place_end']} "
          f"steps={row['steps']} success={row['success']} "
          f"collisions={row['collisions']}", flush=True)

  if workers <= 1:
    _init_worker(cfg, out_dir, holdout, realtime)
    try:
      for task in tasks:
        report(_run_task(task))
    finally:
      _worker["env"].close()
      _worker.clear()
  else:
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg, out_dir, holdout, realtime)) as pool:
      for row in pool.imap_unordered(_run_task, tasks):
        report(row)

  # Only the parent writes the index, so parallel workers cannot clash
  rows.sort(key=lambda r: r["episode"])
  append_index(out_dir, rows)

  elapsed = time.time() - start
  print(f"\ndone: {episodes} episodes in {elapsed:.0f}s "
        f"({elapsed / episodes:.1f}s/episode, {workers} workers) -> {out_dir}")
  stats = {}
  for row in rows:
    s = stats.setdefault(row["policy"], [0, 0])
    s[0] += int(row["success"])
    s[1] += 1
  for kind in sorted(stats):
    print(f"  {kind}: {stats[kind][0]}/{stats[kind][1]} successful")
  return rows

def main():
  p = argparse.ArgumentParser(description="Collect pick-and-place episodes")
  p.add_argument("--config", default=None)
  p.add_argument("--episodes", type=int, default=100)
  p.add_argument("--out", default=None)
  p.add_argument("--seed", type=int, default=None)
  p.add_argument("--workers", type=int, default=None)
  p.add_argument("--holdout", action="store_true")
  p.add_argument("--pointcloud", action="store_true")
  p.add_argument("--realtime", action="store_true", help="pace the simulation to wall clock time instead of running as fast as it can")
  args = p.parse_args()

  cfg = load_config(args.config)
  if args.pointcloud:
    cfg.cameras.pointcloud.enabled = True
  out = args.out or f"{cfg.data.root}/episodes"
  collect(cfg, args.episodes, out, seed=args.seed, workers=args.workers, holdout=args.holdout,
          realtime=args.realtime)

if __name__ == "__main__":
  main()
