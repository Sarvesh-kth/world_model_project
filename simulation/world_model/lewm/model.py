"""Official LeWM core with this project's causal frame/action and native-cost adapters."""
import pathlib

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import ViTConfig, ViTModel

from .upstream.jepa import JEPA
from .upstream.module import ARPredictor, Embedder, MLP
from world_model.vision.data import digest

UPSTREAM = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
DIM, IMAGE_SIZE = 192, 224


def build(history=3, frameskip=5):
    encoder = ViTModel(ViTConfig(image_size=IMAGE_SIZE, patch_size=14,
        hidden_size=DIM, num_hidden_layers=12, num_attention_heads=3,
        intermediate_size=4*DIM, hidden_dropout_prob=0, attention_probs_dropout_prob=0),
        add_pooling_layer=False, use_mask_token=False)
    projector = lambda: MLP(DIM, 2048, DIM, norm_fn=nn.BatchNorm1d)
    return JEPA(encoder=encoder,
        predictor=ARPredictor(num_frames=history, input_dim=DIM, hidden_dim=DIM,
            output_dim=DIM, depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=.1),
        action_encoder=Embedder(input_dim=5*frameskip, emb_dim=DIM),
        projector=projector(), pred_proj=projector())


def pixels(image):
    """Identical offline/live preprocessing: RGB, bilinear224, ImageNet normalization."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    array = np.array(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR))
    value = torch.from_numpy(array).permute(2, 0, 1).float()/255
    return (value-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]


def indices_and_actions(rollout, start, history, frameskip, target=False):
    """Frame at t -> every actually executed action in [t,t+frameskip) -> next frame."""
    stops = list(range(start-(history-1)*frameskip, start+1, frameskip))
    if target:
        stops.append(start+frameskip)
    indices = [rollout["states"][max(0, i)] for i in stops]
    blocks = [[rollout["actions"][i] if i >= 0 else [0.]*5
               for i in range(t, t+frameskip)] for t in stops[:-1]]
    return indices, np.asarray(blocks, np.float32).reshape(len(stops)-1, 5*frameskip)


class Windows(Dataset):
    """Full scene groups remain in one split; padded history never crosses episodes."""
    def __init__(self, root, manifest, split, history, frameskip):
        self.root = pathlib.Path(root)
        self.frames = [s["frames"][-1] for s in manifest["states"]]
        self.samples = []
        for r in manifest["rollouts"]:
            if r["split"] == split:
                for start in range(0, len(r["actions"])-frameskip+1, frameskip):
                    self.samples.append(indices_and_actions(r, start, history, frameskip, target=True))
        if len(self.samples) < 2:
            raise ValueError(f"insufficient {split} training windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ids, actions = self.samples[index]
        frames = []
        for i in ids:
            with Image.open(self.root / self.frames[i]) as image:
                frames.append(pixels(image))
        return torch.stack(frames), torch.from_numpy(actions)


def load_core(root, device="cuda"):
    path = pathlib.Path(root) / "models/lewm.pt"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["upstream_commit"] != UPSTREAM or ck["manifest_sha256"] != digest(pathlib.Path(root)/"manifest.json"):
        raise ValueError("LeWM checkpoint belongs to different code/data")
    model = build(ck["history_frames"], ck["frameskip"])
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval(), ck


class NativeModel:
    """Pixels/actions -> imagined latents -> official goal-image cost. No Q or D_p."""
    def __init__(self, root):
        self.core, self.ck = load_core(root)
        self.history, self.frameskip = self.ck["history_frames"], self.ck["frameskip"]
        self.mean = self.ck["action_mean"].to("cuda")
        self.std = self.ck["action_std"].to("cuda")

    @torch.inference_mode()
    def encode(self, frames):
        value = torch.stack([pixels(f) for f in frames])[None].to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.core.encode({"pixels": value})["emb"]
        z = z.float()[0]
        if not torch.isfinite(z).all():
            raise ValueError("nonfinite LeWM embedding")
        return z

    @torch.inference_mode()
    def rollout(self, visual, past, actions):
        """Each candidate owns its complete history; only its initial context is real."""
        actions = torch.as_tensor(actions, dtype=torch.float32, device="cuda")
        n, horizon, _ = actions.shape
        z = torch.as_tensor(visual, dtype=torch.float32, device="cuda").unsqueeze(0).expand(n, -1, -1)
        a = torch.as_tensor(past, dtype=torch.float32, device="cuda").unsqueeze(0).expand(n, -1, -1)
        predicted = []
        for t in range(horizon):
            a = torch.cat((a, actions[:, t:t+1]), 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                future = self.core.predict(z, self.core.action_encoder((a-self.mean)/self.std))[:, -1:]
            future = future.float()
            predicted.append(future[:, 0])
            z = torch.cat((z[:, 1:], future), 1)
            a = a[:, 1:]
        return torch.stack(predicted, 1)

    def cost(self, predicted, goal):
        # Use the upstream function itself: SUM squared terminal-coordinate differences.
        goal = torch.as_tensor(goal, dtype=torch.float32, device="cuda")
        return self.core.criterion({"predicted_emb": predicted[:, None],
            "goal_emb": goal[None, None, None].expand(len(predicted), 1, 1, -1)})[:, 0]


def plan(model, visual, past, goal, args, rng, previous=None):
    """Native CEM, terminal latent goal distance; first physical command is executed."""
    blocks = args.horizon//model.frameskip
    shape = (blocks, 5*model.frameskip)
    mean, std = np.zeros(shape, np.float32), np.ones(shape, np.float32)
    if previous is not None:
        # Replan after one control action, so shift by ONE command, not one 5-action block.
        flat = previous.reshape(-1, 5)
        mean = np.concatenate((flat[1:], flat[-1:])).reshape(shape)
    best = None
    for iteration in range(args.iterations):
        actions = np.clip(rng.normal(mean, std, (args.population, *shape)), -1, 1).astype(np.float32)
        commands = actions.reshape(args.population, args.horizon, 5)
        commands[..., -1] = np.where(commands[..., -1] > 0, 1, -1)
        actions[0] = mean
        # Mean gripper values are converted too, keeping all candidates in the trained action domain.
        commands[..., -1] = np.where(commands[..., -1] > 0, 1, -1)
        with torch.inference_mode():
            predicted = model.rollout(visual, past, actions)
            costs = model.cost(predicted, goal).cpu().numpy()
        finite = np.isfinite(costs) & torch.isfinite(predicted).all(dim=(1, 2)).cpu().numpy()
        costs[~finite] = np.inf
        if not finite.any():
            raise RuntimeError("all LeWM candidates are nonfinite; inspect training/collapse diagnostics")
        selected = int(np.argmin(costs))
        if best is None or costs[selected] < best["costs"][best["selected"]]:
            best = {"actions": actions.copy(), "costs": costs.copy(), "selected": selected,
                    "selected_z": predicted[selected].cpu().numpy(), "iteration": iteration,
                    "valid": finite.copy()}
        elites = actions[np.argsort(costs)[:min(args.elites, int(finite.sum()))]]
        mean, std = elites.mean(0), np.maximum(elites.std(0), .15)
    best["sequence"] = best["actions"][best["selected"]].reshape(-1, 5)
    best["action"] = best["sequence"][0].copy()
    return best
