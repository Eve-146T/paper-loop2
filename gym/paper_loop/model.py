"""Actor-critic network. CNN over the egocentric crop + MLP over scalars, shared
trunk, separate policy (N_ACTIONS logits) and value heads.

The stem downsamples immediately (stride-2): at 21x21 the convs were launch-overhead
bound, so we drop to ~11x11 then ~6x6 with wider channels — faster on the 3090 AND
higher capacity. We flatten the final 6x6 map (not global-pool) to keep coarse
"where is the threat" position info, which matters for an egocentric view.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rules import OBS_C, OBS_S, N_ACTIONS, OBS_R_DEFAULT


def _ortho(m, gain=1.0):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    return m


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, ch), ch)
        self.c1 = _ortho(nn.Conv2d(ch, ch, 3, 1, 1), 2 ** 0.5)
        self.n2 = nn.GroupNorm(min(8, ch), ch)
        self.c2 = _ortho(nn.Conv2d(ch, ch, 3, 1, 1), 2 ** 0.5)

    def forward(self, x):
        y = self.c1(F.relu(self.n1(x)))
        y = self.c2(F.relu(self.n2(y)))
        return x + y


class ActorCritic(nn.Module):
    """grid: [N, OBS_C, H, W] (binary), scal: [N, OBS_S]."""

    def __init__(self, width=96, blocks=4, scalar_dim=128, trunk_dim=512,
                 c=OBS_C, s=OBS_S, r=OBS_R_DEFAULT):
        super().__init__()
        W = width
        self.stem = _ortho(nn.Conv2d(c, W, 3, 2, 1), 2 ** 0.5)        # 21 -> 11
        self.b1 = nn.Sequential(*[ResBlock(W) for _ in range(2)])
        self.d1 = _ortho(nn.Conv2d(W, 2 * W, 3, 2, 1), 2 ** 0.5)      # 11 -> 6
        self.b2 = nn.Sequential(*[ResBlock(2 * W) for _ in range(blocks)])
        H = 2 * r + 1
        with torch.no_grad():
            d = torch.zeros(1, c, H, H)
            d = self.b2(F.relu(self.d1(self.b1(F.relu(self.stem(d))))))
            flat = d.numel()
        self.scalar = nn.Sequential(
            _ortho(nn.Linear(s, scalar_dim), 2 ** 0.5), nn.ReLU(),
            _ortho(nn.Linear(scalar_dim, scalar_dim), 2 ** 0.5), nn.ReLU())
        self.trunk = nn.Sequential(
            _ortho(nn.Linear(flat + scalar_dim, trunk_dim), 2 ** 0.5), nn.ReLU(),
            _ortho(nn.Linear(trunk_dim, trunk_dim), 2 ** 0.5), nn.ReLU())
        self.pi = _ortho(nn.Linear(trunk_dim, N_ACTIONS), 0.01)
        self.v = _ortho(nn.Linear(trunk_dim, 1), 1.0)

    def features(self, grid, scal):
        x = F.relu(self.stem(grid))
        x = self.b1(x)
        x = F.relu(self.d1(x))
        x = self.b2(x)
        x = x.flatten(1)
        return self.trunk(torch.cat([x, self.scalar(scal)], dim=1))

    def forward(self, grid, scal):
        h = self.features(grid, scal)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, grid, scal, greedy=False):
        logits, value = self.forward(grid, scal)
        if greedy:
            action = logits.argmax(-1)
            logp = torch.zeros_like(action, dtype=torch.float32)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action, logp, value

    def evaluate(self, grid, scal, action):
        logits, value = self.forward(grid, scal)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value


def count_params(model):
    return sum(p.numel() for p in model.parameters())
