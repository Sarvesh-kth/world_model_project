"""M2's world model (simulation/world_model/, from the M2_Kuba branch) behind M3's interfaces.

The only file in controller/ that imports M2's code.

    JEPAEncoder    frozen V-JEPA 2: a 64-frame clip -> mean-pooled z [1024]. Same computation as
                   M2's world_model/encode.py, but on cuda, mps or cpu (M2's script needs CUDA).
    ClipBuffer     the last 64 camera frames; at episode start the first frame is repeated (M2's
                   convention). Frames are JPEG round-tripped like the recorded training data.
    JEPADynamics   M2's trained D checkpoint as a dynamics_fn over the planner state
                   s = [z normalized (1024) | p normalized (20)], p = M1's proprio vector.
"""

from collections import deque

import cv2
import numpy as np
import torch

from controller.adapters.m1_adapter import SIM_DIR  # noqa: F401  (puts simulation/ on the import path)
from controller.config import get_device
from world_model.train_dynamics import model_from_checkpoint  # noqa: E402

VJEPA_MODEL = "facebook/vjepa2-vitl-fpc64-256"  # M2's choice (world_model/encode.py)
JPEG_QUALITY = 90  # M1's collector default (data.jpeg_quality)


class JEPAEncoder:
    def __init__(self, device=None, dtype=None, model=VJEPA_MODEL):
        from transformers import AutoModel, AutoVideoProcessor

        self.device = get_device(device)
        # Half precision on GPUs (M2 uses bfloat16 on CUDA; float16 is the fast one on Apple MPS)
        default = {"cuda": torch.bfloat16, "mps": torch.float16}.get(self.device.type, torch.float32)
        self.dtype = dtype or default
        self.model_name = model
        self.processor = AutoVideoProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, attn_implementation="sdpa").to(self.device, self.dtype).eval()

    @torch.inference_mode()
    def encode_clip(self, frames):
        """frames [T, H, W, 3] uint8 RGB (T = 64) -> z [1024] float32, mean over all tokens."""
        video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        inputs = self.processor(video, return_tensors="pt")
        inputs = {k: v.to(self.device, self.dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
        tokens = self.model(**inputs, skip_predictor=True).last_hidden_state
        return tokens.float().mean(dim=1).squeeze(0).cpu().numpy()


def jpeg_roundtrip(frame, quality=JPEG_QUALITY):
    """Compress and decompress an RGB frame the way M1's collector stores training images."""
    ok, buf = cv2.imencode(".jpg", frame[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]


class ClipBuffer:
    def __init__(self, n_frames=64, jpeg=True):
        self.n_frames, self.jpeg = n_frames, jpeg
        self.frames = deque(maxlen=n_frames)

    def reset(self, frame):
        """Start of an episode: fill the clip with the first frame."""
        frame = jpeg_roundtrip(frame) if self.jpeg else frame
        self.frames.clear()
        self.frames.extend([frame] * self.n_frames)

    def push(self, frame):
        self.frames.append(jpeg_roundtrip(frame) if self.jpeg else frame)

    def clip(self):
        return np.stack(self.frames)


class JEPADynamics:
    def __init__(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.device = torch.device(device)
        self.model = model_from_checkpoint(ckpt).to(self.device).eval()
        self.z_dim, self.p_dim, self.a_dim = ckpt["z_dim"], ckpt["p_dim"], ckpt["a_dim"]
        self.clip_frames, self.camera = ckpt["clip_frames"], ckpt["camera"]
        self.encoder_model, self.architecture = ckpt["encoder_model"], ckpt.get("architecture", "Dynamics")
        stats = {k: ckpt[k].float().to(self.device) for k in ("z_mean", "z_std", "p_mean", "p_std")}
        self.z_mean, self.z_std, self.p_mean, self.p_std = (stats[k] for k in ("z_mean", "z_std", "p_mean", "p_std"))

    def state(self, z, p):
        """Planner state from a raw latent z [1024] and proprio p [20] (numpy or torch)."""
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        p = torch.as_tensor(p, dtype=torch.float32, device=self.device)
        return torch.cat([(z - self.z_mean) / self.z_std, (p - self.p_mean) / self.p_std], dim=-1)

    def proprio(self, s):
        """Raw proprio [..., 20] back out of planner states [..., 1044]."""
        return s[..., self.z_dim:] * self.p_std + self.p_mean

    @torch.inference_mode()
    def __call__(self, s, a):
        next_z, next_p = self.model(s[..., :self.z_dim], s[..., self.z_dim:], a.to(s.dtype))
        return torch.cat([next_z, next_p], dim=-1)
