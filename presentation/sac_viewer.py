"""Read-only Jupyter GIF inspection and presentation plots for saved SAC/MPC runs.

Lives outside simulation/ so pulling this viewer does not change a running experiment's source signature.
"""
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import random
import statistics
from urllib.parse import quote
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

POLICIES = ("train_empty", "train_mixed")
CONDITIONS = ("empty", "obstacles")
LABELS = {"train_empty": "Empty-trained SAC", "train_mixed": "Mixed-trained SAC", "mpc": "Calle MPC", "empty": "Empty table", "obstacles": "Obstacles"}
COLORS = {"train_empty": "#005293", "train_mixed": "#D67A20", "mpc": "#228833"}


class SACViewer:
    def __init__(self, run, random_seed=20261017, mpc_runs=None):
        self.run = Path(run).resolve()
        self.random_seed = random_seed
        self.mpc_runs = {key: Path(value).resolve() for key, value in (mpc_runs or {}).items()}
        if self.mpc_runs and set(self.mpc_runs) != set(CONDITIONS):
            raise ValueError("Provide both empty and obstacles MPC run folders")
        self.policies = POLICIES + (("mpc",) if self.mpc_runs else ())
        self.stem = "sac_mpc" if self.mpc_runs else "sac"
        if not (self.run / "pipeline.json").is_file():
            raise FileNotFoundError(f"No pipeline.json under {self.run}")
        # Write only into a separate presentation subfolder of the small reports directory.
        self.output = self.run.parents[2] / "results" / self.run.name / "presentation"
        if self.mpc_runs:
            self.output = self.output / "with_mpc"
        self.output.mkdir(parents=True, exist_ok=True)

    def snapshot(self):
        state = json.loads((self.run / "pipeline.json").read_text())
        rows = []
        for policy in POLICIES:
            for condition in CONDITIONS:
                root = self.run / "evaluation" / policy / condition / "episodes/sac_imagined"
                for marker in sorted(root.glob("seed_*/complete.json")):
                    completion = json.loads(marker.read_text())
                    if completion["run_signature"] != state["signature"]:
                        raise ValueError(f"Episode belongs to another run: {marker}")
                    result = json.loads(marker.with_name("result.json").read_text())
                    distance = float(result["final_goal_distance_cm"])
                    if not math.isfinite(distance) or distance < 0:
                        raise ValueError(f"Invalid final distance: {marker}")
                    rows.append({**result, "policy": policy, "condition": condition,
                                 "folder": str(marker.parent.relative_to(self.run)),
                                 "actor_sha256": completion["actor_sha256"],
                                 "scene_sha256": completion["scene_sha256"],
                                 "initial_state_sha256": completion["initial_state_sha256"]})
        if self.mpc_runs:
            mpc_rows, recordings = self._mpc_rows(state, rows)
            rows.extend(mpc_rows)
            state = {**state, "mpc_recordings": recordings,
                     "complete": state["complete"] and all(
                         sum(r["condition"] == c for r in mpc_rows) == state["options"]["episodes"] for c in CONDITIONS)}
        return state, rows

    def _mpc_rows(self, state, sac_rows):
        """Read saved MPC outputs, checking each layout and reset against SAC evidence."""
        import numpy as np
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
        scenes = json.loads((self.run / "scenes.json").read_text())
        reference = {(r["condition"], r["seed"]): r for r in sac_rows if r["policy"] == POLICIES[0]}
        recorded_inputs = json.loads((self.run / "inputs.json").read_text())["hashes"]
        rows, recordings = [], {}
        for condition, root in self.mpc_runs.items():
            if not (root / "settings.json").is_file():
                continue
            settings = json.loads((root / "settings.json").read_text())
            if ("mpc" not in settings["methods"] or not settings["random_ends"] or
                    settings["position_jitter"] != state["options"]["position_jitter"] or settings["tag"] or
                    settings["obstacles"] != (None if condition == "empty" else [1, 2])):
                raise ValueError(f"MPC protocol differs from SAC: {root}")
            if (any(settings[key] for key in ("blind", "no_goal", "no_penalties", "no_width_gate", "cube_anchor")) or
                    settings["reward_source"] != "readout" or settings["obstacle_source"] != "model"):
                raise ValueError(f"Unexpected MPC perception/model options: {root}")
            intended = {s["seed"]: s for s in scenes if s["condition"] == condition}
            if {settings["seed"] + 20000 + i for i in range(settings["test_episodes"])} != set(intended):
                raise ValueError(f"MPC planned seeds differ from SAC: {root}")
            bundle = Path(settings["models_run"])
            if not bundle.is_absolute():
                bundle = self.run.parents[1] / bundle
            model_hashes = {}
            for name in ("info.json", "features/meta.json", "features/pca.npz", "models/dynamics.pt",
                         "models/readout.pt", "models/reward.pt", "models/obstacle.pt"):
                path = (bundle / name).resolve()
                value = hashlib.sha256(path.read_bytes()).hexdigest()
                if recorded_inputs.get(str(path)) != value:
                    raise ValueError(f"Current MPC world-model file differs from recorded SAC input: {path}")
                model_hashes[name] = value
            planner_settings = None
            for path in sorted((root / "episodes/mpc").glob("seed_*/result.json")):
                result = json.loads(path.read_text())
                seed = result["seed"]
                if (result["method"] != "mpc" or seed not in intended or path.parent.name != f"seed_{seed}" or
                        result["layout"] != intended[seed]["layout"]):
                    raise ValueError(f"MPC method/seed/layout mismatch: {path}")
                distance = float(result["final_goal_distance_cm"])
                if not math.isfinite(distance) or distance < 0:
                    raise ValueError(f"Invalid MPC final distance: {path}")
                settings_here = json.loads(path.with_name("mpc_settings.json").read_text())
                if planner_settings is not None and settings_here != planner_settings:
                    raise ValueError(f"Mixed MPC configurations within {root}")
                planner_settings = settings_here
                if not path.with_name("actual.gif").is_file():
                    raise FileNotFoundError(path.with_name("actual.gif"))
                with np.load(path.with_name("trajectory.npz"), allow_pickle=False) as trajectory:
                    initial_hash = digest({key: trajectory[key][0].tolist() for key in ("qpos", "p", "object_xyz", "held")})
                original = reference[condition, seed]
                if initial_hash != original["initial_state_sha256"]:
                    raise ValueError(f"MPC initial state differs from SAC: {path}")
                rows.append({**result, "policy": "mpc", "condition": condition, "folder": str(path.parent),
                             "scene_sha256": digest(intended[seed]), "initial_state_sha256": initial_hash})
            recordings[condition] = {"folder": str(root), "settings": settings,
                                     "mpc_settings": planner_settings, "current_model_hashes": model_hashes}
        configurations = [r["mpc_settings"] for r in recordings.values() if r["mpc_settings"] is not None]
        if configurations and any(s != configurations[0] for s in configurations):
            raise ValueError("Empty and obstacle MPC runs used different planner configurations")
        return rows, recordings

    def _selection(self, policy, condition):
        if policy not in self.policies or condition not in CONDITIONS:
            raise ValueError(f"Use {self.policies} and empty/obstacles")
        state, rows = self.snapshot()
        rows = [r for r in rows if r["policy"] == policy and r["condition"] == condition]
        successes = [r for r in rows if r["task_success"]]
        failures = [r for r in rows if not r["task_success"]]
        rng = random.Random(self.random_seed + 10 * self.policies.index(policy) + CONDITIONS.index(condition))
        closest = sorted(failures, key=lambda r: (r["final_goal_distance_cm"], r["seed"]))[:3]
        remaining = [r for r in failures if r["seed"] not in {e["seed"] for e in closest}]
        groups = [("Successes — reproducible random sample", rng.sample(successes, min(4, len(successes))), 4),
                  ("Failures — closest final distances", closest, 3),
                  ("Failures — random, excluding the closest examples", rng.sample(remaining, min(3, len(remaining))), 3)]
        selected = [r for _, group, _ in groups for r in group]
        assert len({r["folder"] for r in selected}) == len(selected), "Duplicate example selection"
        selection = [{"category": title, "seed": row["seed"], "folder": row["folder"],
                      "task_success": row["task_success"], "final_goal_distance_cm": row["final_goal_distance_cm"]}
                     for title, group, _ in groups for row in group]
        manifest = self.output / f"selected_{policy}_{condition}.json"
        manifest.write_text(json.dumps({"random_seed": self.random_seed,
                                       "run_signature": state["signature"], "examples": selection}, indent=2) + "\n")
        return state, rows, groups

    def show(self, policy, condition):
        """Four seeded random successes, three closest failures, three OTHER random failures."""
        from IPython.display import Image, Markdown, display
        state, rows, groups = self._selection(policy, condition)
        successes = [r for r in rows if r["task_success"]]
        failures = [r for r in rows if not r["task_success"]]
        display(Markdown(f"## {LABELS[policy]} — {LABELS[condition]}"))
        expected = state["options"]["episodes"]
        print(f"Completed {len(rows)}/{expected}; successes={len(successes)}; failures={len(failures)}")
        if not state["complete"]:
            print("Run still in progress. Rerun this cell to refresh completed episodes.")
        for title, group, requested in groups:
            display(Markdown(f"### {title}"))
            if len(group) < requested:
                print(f"Only {len(group)} of {requested} requested examples available; no duplicates added.")
            for number, row in enumerate(group, 1):
                print(f"{number}. seed={row['seed']} | placed={row['task_success']} | "
                      f"final B={row['final_goal_distance_cm']:.2f} cm | "
                      f"obstacle contact steps={row['obstacle_contact_steps']}")
                print(row["folder"])
                gif = self.run / row["folder"] / "actual.gif"
                if gif.is_file():
                    display(Image(data=gif.read_bytes(), format="gif", width=450))
                else:
                    print(f"Recorded GIF is missing: {gif}")

    def presentation_figures(self):
        """Use ALL completed results; require the full campaign before saving presentation figures."""
        state, rows = self.snapshot()
        expected = state["options"]["episodes"]
        groups = {(p, c): [r for r in rows if r["policy"] == p and r["condition"] == c]
                  for p in self.policies for c in CONDITIONS}
        if not state["complete"] or any(len(group) != expected for group in groups.values()):
            print("Final figures are waiting for the complete campaign:")
            for (p, c), group in groups.items():
                print(f"  {LABELS[p]} / {LABELS[c]}: {len(group)}/{expected}")
            print("Rerun this final cell after the pipeline prints COMPLETE.")
            return False
        scenes = json.loads((self.run / "scenes.json").read_text())
        for condition in CONDITIONS:
            intended = {s["seed"] for s in scenes if s["condition"] == condition}
            for policy in self.policies:
                assert {r["seed"] for r in groups[policy, condition]} == intended, "Wrong evaluation seeds"
            left = {r["seed"]: r for r in groups[POLICIES[0], condition]}
            for right in [r for p in self.policies[1:] for r in groups[p, condition]]:
                for key in ("scene_sha256", "initial_state_sha256"):
                    if left[right["seed"]][key] != right[key]:
                        raise ValueError("Controllers did not start from identical scenes/states")
        table = []
        for (policy, condition), group in groups.items():
            failed = [r["final_goal_distance_cm"] for r in group if not r["task_success"]]
            selected = state["selected_checkpoints"].get(policy)
            table.append({"policy": policy, "condition": condition, "episodes": len(group),
                          "successes": sum(bool(r["task_success"]) for r in group), "failures": len(failed),
                          "mean_final_distance_cm": statistics.mean(r["final_goal_distance_cm"] for r in group),
                          "failure_only_mean_cm": statistics.mean(failed) if failed else None,
                          "obstacle_contact_steps": sum(r["obstacle_contact_steps"] for r in group),
                          "mean_decision_ms": statistics.mean(r["mean_decision_seconds"] * 1000 for r in group),
                          "selected_checkpoint": selected["file"] if selected else "MPC: no SAC actor",
                          "selected_SAC_transitions": selected["transitions"] if selected else None})
        with (self.output / "presentation_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        selection_note = "Selected checkpoints: " + "; ".join(
            f"{LABELS[p]} — " + ("BC only" if state["selected_checkpoints"][p]["transitions"] == 0
                                 else f"{state['selected_checkpoints'][p]['transitions']:,} SAC transitions")
            for p in POLICIES)
        self._plot_metrics(groups, table, selection_note)
        self._plot_outcomes(groups, selection_note)
        (self.output / "figure_provenance.json").write_text(json.dumps({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "run_signature": state["signature"],
            "all_episodes": rows, "selected_checkpoints": state["selected_checkpoints"],
            "mpc_recordings": state.get("mpc_recordings"),
            "selection_note": "Figures use all evaluation episodes, independent of GIF selections.",
            "limits": "One training seed per SAC variant; contact timesteps are not unique collision events. "
                      "Decision-time CSV reports the mean of episode-level means. " +
                      ("MPC protocol/layout/reset and current model hashes checked. Legacy MPC outputs have no runtime "
                      "code/model hash snapshot: current hashes cannot prove historical file contents. "
                      "MPC has different obstacle inputs and score terms: this compares complete systems." if self.mpc_runs else "")
        }, indent=2) + "\n")
        for row in table:
            n, successes = row["episodes"], row["successes"]
            print(f"{LABELS[row['policy']]} / {LABELS[row['condition']]}: {successes}/{n} "
                  f"({100 * successes / n:.1f}%); mean B={row['mean_final_distance_cm']:.2f} cm; "
                  f"contacts={row['obstacle_contact_steps']}; selected={row['selected_checkpoint']}")
        print(f"\nSaved PNG (300 dpi), SVG, metrics CSV and provenance under:\n{self.output}")
        return True

    def export_zip(self):
        """Build a final presentation download from the same selections as the galleries."""
        from IPython.display import HTML, display
        if not self.presentation_figures():
            return None
        files = []
        counts = []
        categories = ("successes", "failures_closest", "failures_random")
        for policy in self.policies:
            for condition in CONDITIONS:
                _, _, groups = self._selection(policy, condition)
                for category, (_, group, requested) in zip(categories, groups):
                    counts.append(f"{policy}/{condition}/{category}: {len(group)}/{requested} clips")
                    for index, row in enumerate(group, 1):
                        folder = self.run / row["folder"]
                        target = f"videos/{policy}/{condition}/{category}/{index:02d}_seed_{row['seed']}"
                        files.extend([(folder / "actual.gif", f"{target}.gif"),
                                      (folder / "result.json", f"{target}_result.json")])
                manifest = self.output / f"selected_{policy}_{condition}.json"
                files.append((manifest, f"metrics/{manifest.name}"))
        for name in (f"{self.stem}_comparison.png", f"{self.stem}_comparison.svg", f"{self.stem}_outcome_stages.png", f"{self.stem}_outcome_stages.svg"):
            files.append((self.output / name, f"graphs/{name}"))
        for name in ("presentation_metrics.csv", "figure_provenance.json"):
            files.append((self.output / name, f"metrics/{name}"))
        missing = [str(path) for path, _ in files if not path.is_file()]
        if missing:
            raise FileNotFoundError("ZIP not created; missing files:\n" + "\n".join(missing))
        assert len({name for _, name in files}) == len(files), "Duplicate ZIP filenames"
        # A fixed project path also works when VS Code's kernel starts in a different directory.
        archive = self.run.parents[2] / "presentation" / ("sac_mpc_comparison_presentation.zip" if self.mpc_runs
                                                        else f"{self.run.name}_presentation.zip")
        temporary = archive.with_suffix(".zip.tmp")
        readme = ("SAC / MPC presentation package\n\n"
                  "GIFs: up to 4 seeded-random successes, 3 closest final-distance failures and\n"
                  "3 other seeded-random failures per policy/table condition. No duplicates.\n"
                  "Selections match galleries rerun after pipeline completion. Fewer examples\n"
                  "are included when a category has fewer available episodes.\n"
                  "Graphs and CSV use ALL evaluation episodes, independent of selected clips.\n"
                  "Placement uses the recorded task_success flag, not distance alone.\n"
                  "One training seed per SAC policy; contacts count timesteps, not unique collisions.\n"
                  "MPC has different obstacle inputs and score terms; this is a complete-system comparison.\n"
                  "Checkpoint identities and BC-only selections, if any, are recorded in metrics.\n\n"
                  + "\n".join(counts) + "\n")
        try:
            with ZipFile(temporary, "w") as bundle:
                bundle.writestr("README.txt", readme, compress_type=ZIP_DEFLATED)
                for path, name in files:
                    bundle.write(path, name, compress_type=ZIP_STORED if path.suffix == ".gif" else ZIP_DEFLATED)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
        print("\n".join(counts))
        print(f"\nZIP saved: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)")
        relative = archive.relative_to(Path.home()).as_posix()
        prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/user/kuba/").rstrip("/")
        url = f"https://gpu1.eecs.kth.se{prefix}/files/{quote(relative, safe='/')}?download=1"
        display(HTML(f'<a href="{escape(url, quote=True)}" target="_blank" download="{archive.name}">Download ZIP</a>'))
        return archive

    def _plot_metrics(self, groups, table, selection_note):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            width = .8 / len(self.policies)
            for index, policy in enumerate(self.policies):
                selected = [r for r in table if r["policy"] == policy]
                xs = [i + (index - (len(self.policies) - 1) / 2) * width for i in range(2)]
                rates = [100 * r["successes"] / r["episodes"] for r in selected]
                bars = axes[0].bar(xs, rates, width=width * .9, color=COLORS[policy], label=LABELS[policy])
                for bar, row in zip(bars, selected):
                    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                                 f"{row['successes']}/{row['episodes']}", ha="center", fontsize=10)
                means = [r["obstacle_contact_steps"] / r["episodes"] for r in selected]
                bars = axes[2].bar(xs, means, width=width * .9, color=COLORS[policy])
                for bar, row in zip(bars, selected):
                    axes[2].annotate(f"{row['obstacle_contact_steps']} total", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                     xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
            axes[0].set(title="Placement success", ylabel="Successful placements (%)", ylim=(0, 115))
            axes[0].legend(fontsize=9)
            axes[2].set(title="Obstacle contact exposure", ylabel="Contact timesteps per episode")
            axes[2].margins(y=.25)
            axes[2].set_ylim(bottom=0)
            if not any(row["obstacle_contact_steps"] for row in table):
                axes[2].set_ylim(0, 1)
            for ax in (axes[0], axes[2]):
                ax.set_xticks([0, 1], [LABELS[c] for c in CONDITIONS])
            ticks = []
            rng = random.Random(self.random_seed)
            for index, ((policy, condition), group) in enumerate(groups.items()):
                ticks.append(f"{LABELS[policy]}\n{LABELS[condition]}")
                for success in (True, False):
                    subset = [r for r in group if bool(r["task_success"]) == success]
                    axes[1].scatter([index + rng.uniform(-.12, .12) for _ in subset],
                                    [r["final_goal_distance_cm"] for r in subset], s=25,
                                    marker="o" if success else "x", color=COLORS[policy], alpha=.75)
                axes[1].scatter(index, statistics.mean(r["final_goal_distance_cm"] for r in group),
                                marker="D", color="black", s=45, zorder=4)
            axes[1].set_xticks(range(len(groups)), ticks, fontsize=8, rotation=15)
            axes[1].set(title="Final cube-to-goal distance", ylabel="Distance (cm)", ylim=(0, None))
            axes[1].legend(handles=[Line2D([], [], marker=m, color="black", linestyle="", label=label)
                                    for m, label in (("o", "Success"), ("x", "Failure"), ("D", "Mean"))], fontsize=8)
            for ax in axes:
                ax.grid(axis="y", alpha=.18)
                ax.set_axisbelow(True)
            title = "Imagined SAC versus Calle MPC" if self.mpc_runs else "Imagined SAC: empty-only versus mixed-scene training"
            fig.suptitle(title, color="#000061", fontsize=16)
            fig.text(.02, .02, "All matched test episodes; one training seed per SAC policy. Placement requires lift, release and settling.\n"
                     "Dots show individual episodes; contact timesteps are not counts of separate collision events.\n" + selection_note, fontsize=9)
            fig.tight_layout(rect=(0, .14, 1, .93))
            self._save(fig, f"{self.stem}_comparison")

    def _plot_outcomes(self, groups, selection_note):
        import matplotlib.pyplot as plt
        names = ("Placed", "Never held", "Held, never lifted 4 cm", "Lifted; held at end", "Lifted; unheld at end, no placement")
        colors = ("#228833", "#BBBBBB", "#E6AB02", "#4477AA", "#CC6677")
        counts = [[0] * len(groups) for _ in names]
        for j, group in enumerate(groups.values()):
            for row in group:
                if row["task_success"]:
                    category = 0
                elif not row["ever_held"]:
                    category = 1
                elif not row["ever_lifted_4cm"]:
                    category = 2
                else:
                    category = 3 if row["final_held"] else 4
                counts[category][j] += 1
            assert sum(category[j] for category in counts) == len(group)
        fig, ax = plt.subplots(figsize=(14, 6))
        bottom = [0.] * len(groups)
        for name, color, category in zip(names, colors, counts):
            heights = [100 * n / len(group) for n, group in zip(category, groups.values())]
            ax.bar(range(len(groups)), heights, bottom=bottom, color=color, label=name, width=.65)
            for j, (n, height) in enumerate(zip(category, heights)):
                if height >= 8:
                    ax.text(j, bottom[j] + height / 2, str(n), ha="center", va="center")
            bottom = [low + height for low, height in zip(bottom, heights)]
        ax.set_xticks(range(len(groups)), [f"{LABELS[p]}\n{LABELS[c]}" for p, c in groups], fontsize=9)
        ax.set(ylabel="Episodes (%)", ylim=(0, 100), title="Observed outcomes: which stage was reached?")
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(loc="lower center", bbox_to_anchor=(.5, .07), ncol=3, fontsize=9)
        fig.text(.02, .015, "All episodes. Segment labels are episode counts. Categories describe recorded outcomes, not inferred failure causes.\n" + selection_note, fontsize=9)
        fig.tight_layout(rect=(0, .22, 1, .98))
        self._save(fig, f"{self.stem}_outcome_stages")

    def _save(self, figure, name):
        import matplotlib.pyplot as plt
        figure.savefig(self.output / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
        figure.savefig(self.output / f"{name}.svg", bbox_inches="tight", facecolor="white")
        plt.show()
        plt.close(figure)
