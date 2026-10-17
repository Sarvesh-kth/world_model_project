"""Fetch only the recorded latents/model bundle; reuse byte-identical local inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def link_verified(source, destination, oid, size):
    if source.is_file() and source.stat().st_size == size and sha(source) == oid:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        destination.symlink_to(source.resolve())
        return True
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--existing-root", type=Path, required=True, help="existing world_model_project checkout on the server")
    p.add_argument("--source-imagined", type=Path, help="Calle's actual saved imagined-SAC run, if not at existing-root/simulation/data/sac_imagined_v1")
    args = p.parse_args()
    old = args.existing_root.resolve()
    if old == ROOT:
        p.error("use a separate experiment checkout, leaving the existing repository intact")
    paths = git("ls-files", "simulation/models/v3c", "simulation/data/episodes_v3").splitlines()
    required = [name for name in paths if
                name.endswith(("/manifest.json", "/scene_settings.json", "/meta.json", "/info.json", "/pca.npz")) or
                "/latents.npy.part" in name or
                name in [f"simulation/models/v3c/models/{role}.pt" for role in ("dynamics", "readout", "reward", "obstacle")]]
    pending = []
    for name in required:
        pointer = git("show", f"HEAD:{name}")
        if not pointer.startswith("version https://git-lfs.github.com/spec/"):
            continue
        oid = next(line.split(":", 1)[1] for line in pointer.splitlines() if line.startswith("oid sha256:"))
        size = int(next(line.split()[1] for line in pointer.splitlines() if line.startswith("size ")))
        dest = ROOT / name
        if dest.is_file() and dest.stat().st_size == size and sha(dest) == oid:
            continue
        if dest.is_file() and not dest.is_symlink():
            with dest.open("rb") as current:
                if not current.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                    raise FileExistsError(f"{dest} contains different data; move it aside explicitly")
        if not link_verified(old / name, dest, oid, size):
            pending.append(name)
    if pending:
        print(f"Fetching {len(pending)} required LFS files; no raw camera archives or unrelated weights", flush=True)
        subprocess.run(["git", "lfs", "pull", "--include=" + ",".join(pending), "--exclude="], cwd=ROOT, check=True)
    features = ROOT / "simulation/data/episodes_v3/features"
    parts = sorted(features.glob("latents.npy.part*"))
    if not parts:
        raise FileNotFoundError("tracked latent parts are missing")
    combined = hashlib.sha256()
    size = 0
    for part in parts:
        with part.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"LFS download did not materialize {part}")
            stream.seek(0)
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                combined.update(block)
                size += len(block)
    latent = features / "latents.npy"
    valid = latent.is_file() and latent.stat().st_size == size and sha(latent) == combined.hexdigest()
    if not valid:
        valid = link_verified(old / "simulation/data/episodes_v3/features/latents.npy", latent, combined.hexdigest(), size)
    if not valid:
        if latent.exists() or latent.is_symlink():
            raise FileExistsError(f"{latent} is not the tracked dataset; move it aside explicitly")
        if shutil.disk_usage(features).free < size + 5 * 1024 ** 3:
            raise OSError("need assembled latent size plus 5 GiB free, or a matching existing latents.npy to reuse")
        temporary = latent.with_suffix(".npy.tmp")
        with temporary.open("wb") as output:
            for part in parts:
                with part.open("rb") as stream:
                    shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
        temporary.replace(latent)
    baseline = args.source_imagined or old / "simulation/data/sac_imagined_v1"
    baseline = baseline.resolve()
    files = ("models/sac_best.zip", "pipeline.json", "imagined_training.json")
    destination = ROOT / "simulation/data/sac_imagined_v1"
    if all((baseline / name).is_file() for name in files):
        if json.loads((baseline / "pipeline.json").read_text()).get("kind") != "imagined_sac":
            raise ValueError(f"{baseline} is not an imagined_sac run; ordinary SAC is not a substitute")
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != baseline:
                raise FileExistsError(f"refusing to replace {destination}")
        else:
            destination.symlink_to(baseline, target_is_directory=True)
        print(f"Exact source actor linked from {baseline}; SHA256={sha(baseline / files[0])}")
    elif not all((destination / name).is_file() for name in files):
        print("\nINPUT STILL NEEDED: Calle's original imagined-SAC run is not in Git.")
        print(f"Copy these three files into {destination}: " + ", ".join(files))
        print("Or rerun this script with --source-imagined /path/to/Calle/run. No original actor will be recreated silently.")
    print("Data/model setup complete. Training uses recorded latents, so old raw images are not needed.")


if __name__ == "__main__":
    main()
