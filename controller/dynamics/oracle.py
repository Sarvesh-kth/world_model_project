"""The simulator itself as the world model: the upper bound for any learned dynamics.

State z = [12 task features | full M1 sim state from M1Adapter.get_state()], float64 on the CPU
(sim states need full precision to restore exactly). To imagine one step for a sample, restore
its sim state in a private copy of the env, apply the action, read the result back. Not batched
on a GPU like a learned model, so N has to stay small.
"""

import numpy as np
import torch

from controller.interfaces import N_FEATURES


class OracleDynamics:
    def __init__(self, adapter):
        """adapter: the real env (an M1Adapter). Imagination runs in a clone, never in it."""
        self.sim = adapter.clone()

    @staticmethod
    def observe(adapter):
        """The oracle's z for the real env right now."""
        return torch.from_numpy(np.concatenate([adapter.task_features(), adapter.get_state()]))

    def __call__(self, z, a):
        z_next = np.empty(tuple(z.shape))
        states, actions = z[:, N_FEATURES:].numpy(), a.double().numpy()
        for i in range(len(states)):
            self.sim.set_state(states[i])
            self.sim.step(actions[i])
            z_next[i, :N_FEATURES] = self.sim.task_features()
            z_next[i, N_FEATURES:] = self.sim.get_state()
        return torch.from_numpy(z_next)
