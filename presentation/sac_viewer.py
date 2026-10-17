"""Read-only Jupyter GIF inspection and presentation plots for the fresh SAC run.

Lives outside simulation/ so pulling this viewer does not change a running experiment's source signature.
"""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

POLICIES = ("train_empty", "train_mixed")
CONDITIONS = ("empty", "obstacles")
LABELS = {"train_empty": "Empty-trained", "train_mixed": "Mixed-trained", "empty": "Empty table", "obstacles": "Obstacles"}
COLORS = {"train_empty": "#005293", "train_mixed": "#D67A20"}


class SACViewer:
    def __init__(self, run, random_seed=20261017):
        self.run = Path(run).resolve()
        self.random_seed = random_seed
        if not (self.run / "pipeline.json").is_file():
            raise FileNotFoundError(f"No pipeline.json under {self.run}")
        # Write only into a separate presentation subfolder of the small reports directory.
        self.output = self.run.parents[2] / "results" / self.run.name / "presentation"
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
        return state, rows

    def _selection(self, policy, condition):
        if policy not in POLICIES or condition not in CONDITIONS:
            raise ValueError("Use train_empty/train_mixed and empty/obstacles")
        state, rows = self.snapshot()
        rows = [r for r in rows if r["policy"] == policy and r["condition"] == condition]
        successes = [r for r in rows if r["task_success"]]
        failures = [r for r in rows if not r["task_success"]]
        rng = random.Random(self.random_seed + 10 * POLICIES.index(policy) + CONDITIONS.index(condition))
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
                  for p in POLICIES for c in CONDITIONS}
        if not state["complete"] or any(len(group) != expected for group in groups.values()):
            print("Final figures are waiting for the complete campaign:")
            for (p, c), group in groups.items():
                print(f"  {LABELS[p]} / {LABELS[c]}: {len(group)}/{expected}")
            print("Rerun this final cell after the pipeline prints COMPLETE.")
            return False
        scenes = json.loads((self.run / "scenes.json").read_text())
        for condition in CONDITIONS:
            intended = {s["seed"] for s in scenes if s["condition"] == condition}
            for policy in POLICIES:
                assert {r["seed"] for r in groups[policy, condition]} == intended, "Wrong evaluation seeds"
            left = {r["seed"]: r for r in groups[POLICIES[0], condition]}
            for right in groups[POLICIES[1], condition]:
                for key in ("scene_sha256", "initial_state_sha256"):
                    if left[right["seed"]][key] != right[key]:
                        raise ValueError("The two policies did not start from identical scenes/states")
        table = []
        for (policy, condition), group in groups.items():
            failed = [r["final_goal_distance_cm"] for r in group if not r["task_success"]]
            selected = state["selected_checkpoints"][policy]
            table.append({"policy": policy, "condition": condition, "episodes": len(group),
                          "successes": sum(bool(r["task_success"]) for r in group), "failures": len(failed),
                          "mean_final_distance_cm": statistics.mean(r["final_goal_distance_cm"] for r in group),
                          "failure_only_mean_cm": statistics.mean(failed) if failed else None,
                          "obstacle_contact_steps": sum(r["obstacle_contact_steps"] for r in group),
                          "selected_checkpoint": selected["file"], "selected_SAC_transitions": selected["transitions"]})
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
            "selection_note": "Figures use all evaluation episodes, independent of GIF selections.",
            "limits": "One training seed per variant; contact timesteps are not unique collision events."
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
        from IPython.display import FileLink, display
        if not self.presentation_figures():
            return None
        files = []
        counts = []
        categories = ("successes", "failures_closest", "failures_random")
        for policy in POLICIES:
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
        for name in ("sac_comparison.png", "sac_comparison.svg", "sac_outcome_stages.png", "sac_outcome_stages.svg"):
            files.append((self.output / name, f"graphs/{name}"))
        for name in ("presentation_metrics.csv", "figure_provenance.json"):
            files.append((self.output / name, f"metrics/{name}"))
        missing = [str(path) for path, _ in files if not path.is_file()]
        if missing:
            raise FileNotFoundError("ZIP not created; missing files:\n" + "\n".join(missing))
        assert len({name for _, name in files}) == len(files), "Duplicate ZIP filenames"
        # Save beside the notebook so its relative download URL stays within Jupyter's served tree.
        archive = Path.cwd() / f"{self.run.name}_presentation.zip"
        temporary = archive.with_suffix(".zip.tmp")
        readme = ("Imagined SAC presentation package\n\n"
                  "GIFs: up to 4 seeded-random successes, 3 closest final-distance failures and\n"
                  "3 other seeded-random failures per policy/table condition. No duplicates.\n"
                  "Selections match galleries rerun after pipeline completion. Fewer examples\n"
                  "are included when a category has fewer available episodes.\n"
                  "Graphs and CSV use ALL evaluation episodes, independent of selected clips.\n"
                  "Placement uses the recorded task_success flag, not distance alone.\n"
                  "One training seed per policy; contacts count timesteps, not unique collisions.\n"
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
        display(FileLink(archive.name, result_html_prefix="Download ZIP: "))
        return archive

    def _plot_metrics(self, groups, table, selection_note):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
            fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
            positions = [-.18, .18]
            for index, policy in enumerate(POLICIES):
                selected = [r for r in table if r["policy"] == policy]
                xs = [i + positions[index] for i in range(2)]
                rates = [100 * r["successes"] / r["episodes"] for r in selected]
                bars = axes[0].bar(xs, rates, width=.32, color=COLORS[policy], label=LABELS[policy])
                for bar, row in zip(bars, selected):
                    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                                 f"{row['successes']}/{row['episodes']}", ha="center", fontsize=10)
                means = [r["obstacle_contact_steps"] / r["episodes"] for r in selected]
                bars = axes[2].bar(xs, means, width=.32, color=COLORS[policy])
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
            axes[1].set_xticks(range(4), ticks, fontsize=8)
            axes[1].set(title="Final cube-to-goal distance", ylabel="Distance (cm)", ylim=(0, None))
            axes[1].legend(handles=[Line2D([], [], marker=m, color="black", linestyle="", label=label)
                                    for m, label in (("o", "Success"), ("x", "Failure"), ("D", "Mean"))], fontsize=8)
            for ax in axes:
                ax.grid(axis="y", alpha=.18)
                ax.set_axisbelow(True)
            fig.suptitle("Imagined SAC: empty-only versus mixed-scene training", color="#000061", fontsize=16)
            fig.text(.02, .02, "All matched test episodes; one training seed per policy. Placement requires lift, release and settling.\n"
                     "Dots show individual episodes; contact timesteps are not counts of separate collision events.\n" + selection_note, fontsize=9)
            fig.tight_layout(rect=(0, .14, 1, .93))
            self._save(fig, "sac_comparison")

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
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = [0.] * len(groups)
        for name, color, category in zip(names, colors, counts):
            heights = [100 * n / len(group) for n, group in zip(category, groups.values())]
            ax.bar(range(4), heights, bottom=bottom, color=color, label=name, width=.65)
            for j, (n, height) in enumerate(zip(category, heights)):
                if height >= 8:
                    ax.text(j, bottom[j] + height / 2, str(n), ha="center", va="center")
            bottom = [low + height for low, height in zip(bottom, heights)]
        ax.set_xticks(range(4), [f"{LABELS[p]}\n{LABELS[c]}" for p, c in groups])
        ax.set(ylabel="Episodes (%)", ylim=(0, 100), title="Observed outcomes: which stage was reached?")
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(loc="lower center", bbox_to_anchor=(.5, .07), ncol=3, fontsize=9)
        fig.text(.02, .015, "All episodes. Segment labels are episode counts. Categories describe recorded outcomes, not inferred failure causes.\n" + selection_note, fontsize=9)
        fig.tight_layout(rect=(0, .22, 1, .98))
        self._save(fig, "sac_outcome_stages")

    def _save(self, figure, name):
        import matplotlib.pyplot as plt
        figure.savefig(self.output / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
        figure.savefig(self.output / f"{name}.svg", bbox_inches="tight", facecolor="white")
        plt.show()
        plt.close(figure)
