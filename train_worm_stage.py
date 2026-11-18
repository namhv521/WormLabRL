"""Train and test a separate self-play PPO checkpoint for one WormLabRL stage."""

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from build_worm_training_replay import build
from train_worm import FEATURES, ROOT, dataset, make_policy
from train_worm_mimic import H, W, Observations, write_json
from train_worm_ppo import Rollouts, allowed, collect, masked_logits, recorded_replay, update

HTML = ROOT / "hungry_worm_map_viewer.html"
PPO = ROOT / "worm_ppo_run" / "round_4.pt"
GRID = 11 * H * W


def legal_mask(features):
    legal = ~features[..., 10].astype(bool)
    return np.where(legal.any(axis=-1, keepdims=True), legal, True)


def no_mask(features):
    return np.ones(features.shape[:-1], dtype=bool)


MASKS = {"safe": allowed, "legal": legal_mask, "none": no_mask}


class StageRollouts(Rollouts):
    def __init__(self, row, rng, output, max_steps, architecture):
        super().__init__([row], rng, output, max_steps=max_steps)
        self.arena = row[1]
        self.grid = None
        if architecture == "cnn":
            observation = Observations(self.arena)
            self.grid = lru_cache(maxsize=4096)(
                lambda state: observation.encode(state, self.max_steps)[0].ravel())

    def observation(self, episode):
        features, transitions = super().observation(episode)
        if self.grid is not None:
            grid = np.broadcast_to(self.grid(episode["state"]), (4, GRID))
            features = np.concatenate((features, grid), axis=-1)
        return features, transitions


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(11, 16, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(24 * 3 * 6, 48), nn.ReLU())

    def forward(self, features):
        grid = features[:, 0, FEATURES:].reshape(-1, 11, H, W)
        return self.net(grid)


class StageActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialEncoder()
        self.score = nn.Sequential(nn.Linear(FEATURES + 48, 48), nn.Tanh(), nn.Linear(48, 1))

    def forward(self, features):
        global_state = self.spatial(features).unsqueeze(1).expand(-1, 4, -1)
        return self.score(torch.cat((features[:, :, :FEATURES], global_state), dim=-1))


class StageCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialEncoder()
        self.value = nn.Sequential(nn.Linear(4 * FEATURES + 48, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, flat):
        features = flat.reshape(-1, 4, FEATURES + GRID)
        return self.value(torch.cat((features[:, :, :FEATURES].flatten(1),
                                     self.spatial(features)), dim=1))


def networks(architecture):
    if architecture == "cnn":
        return StageActor(), StageCritic()
    return make_policy(), nn.Sequential(nn.Linear(4 * FEATURES, 64), nn.Tanh(), nn.Linear(64, 1))


def configured_map(args):
    name = f"{args.world}{args.stage}"
    rows = dataset(HTML)
    for entry, arena, trace in rows:
        if entry["name"] == name:
            if entry["verification"] != "win":
                raise ValueError(f"{name} does not have a verified simulator start and solution")
            # The reference trace validates the simulator. Rollouts never receive its actions.
            return entry, arena, ()
    raise ValueError(f"Map {name} is absent")


@torch.inference_mode()
def greedy(row, policy, config, device):
    policy.eval()
    env = StageRollouts(row, np.random.default_rng(0), StringIO(),
                        config["max_episode_steps"], config["model"])
    episode = env.episode(0)
    for _ in range(config["max_episode_steps"]):
        observed, options = env.observation(episode)
        inputs = observed[None, ...]
        features = torch.from_numpy(inputs).to(device)
        mask = torch.from_numpy(MASKS[config["action_mask"]](inputs)).to(device)
        choice = int(masked_logits(policy, features, mask).argmax(1).item())
        episode, _, record = env.step(episode, choice, options, 0)
        if record is not None:
            return record
    raise RuntimeError("Episode failed to terminate")


def rank(record):
    if record["result"] == "win":
        return 1, -len(record["actions"])
    return 0, record["reward"]


def save(path, actor, critic, optimizer, config, progress):
    temporary = path.with_suffix(".tmp")
    torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(),
                    optimizer=optimizer.state_dict(), config=config, progress=progress), temporary)
    temporary.replace(path)


def replay(folder, row, record, *, captures=(), start=None):
    entry, arena, _ = row
    attempts = []
    if start:
        attempts.append(dict(recorded_replay(start, arena), mode="stage_start"))
    attempts.extend(recorded_replay(value, arena) for value in captures)
    attempts.append(dict(recorded_replay(record, arena), mode="stage_greedy"))
    data = dict(info=dict(stage_ppo=True, splits=dict(train=1, test=0, duplicate=0, unverified=0)),
                maps=[dict(entry, split="train", attempts=attempts)])
    path = folder / "replay.json"
    write_json(path, data)
    build(HTML, path, folder / "replay.html")


def train(args, row, device):
    quantum = args.envs * args.horizon
    if min(args.steps, args.envs, args.horizon, args.epochs, args.minibatch,
           args.eval_every, args.max_episode_steps) <= 0 or args.steps % args.envs:
        raise ValueError("Steps must be a positive multiple of the number of environments")
    if args.lr <= 0 or args.entropy < 0 or not math.isfinite(args.lr + args.entropy):
        raise ValueError("Use a finite positive learning rate and nonnegative entropy")
    if args.init == "ppo" and args.model != "mlp":
        raise ValueError("PPO round 4 weights require --model mlp")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Use an empty output folder")
    config = {key: value for key, value in vars(args).items() if key != "out"}
    config.update(map=row[0]["name"], data_sha256=hashlib.sha256(HTML.read_bytes()).hexdigest())
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    actor, critic = (model.to(device) for model in networks(args.model))
    if args.init == "ppo":
        loaded = torch.load(PPO, map_location=device, weights_only=True)
        actor.load_state_dict(loaded["model"])
        critic.load_state_dict(loaded["critic"])
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=args.lr)
    first = best = greedy(row, actor, config, device)
    best_steps = 0
    history = []
    save(args.out / "best.pt", actor, critic, optimizer, config, dict(steps=0, greedy=best))
    counts, captures = Counter(), {}
    started = time.monotonic()
    print(f"Map {config['map']} {args.model}/{args.init} · {args.envs} slots · "
          f"{args.steps} steps · greedy before: {first['result']}", flush=True)
    with (args.out / "episodes.jsonl").open("w", encoding="utf-8") as output:
        rollouts = StageRollouts(row, rng, output, args.max_episode_steps, args.model)
        slots = [rollouts.episode(0) for _ in range(args.envs)]
        for index, previous in enumerate(range(0, args.steps, quantum), 1):
            steps = min(args.steps, previous + quantum)
            horizon = (steps - previous) // args.envs
            rollouts.round = index
            actor.train()
            data = collect(rollouts, slots, actor, critic, horizon, 0, counts, captures,
                           action_mask=MASKS[args.action_mask], device=device)
            update(actor, critic, optimizer, data, None, args.epochs, 0,
                   minibatch=args.minibatch, entropy=args.entropy)
            if index % args.eval_every and steps != args.steps:
                continue
            result = greedy(row, actor, config, device)
            point = dict(steps=steps, train_episodes=counts["root_episodes"],
                         train_wins=counts["root_wins"], greedy=result)
            history.append(point)
            if rank(result) > rank(best):
                best, best_steps = result, steps
                save(args.out / "best.pt", actor, critic, optimizer, config, point)
            save(args.out / "latest.pt", actor, critic, optimizer, config, point)
            output.flush()
            print(f"{config['map']} {steps}/{args.steps} · wins {point['train_wins']}/"
                  f"{point['train_episodes']} · greedy {result['result']} "
                  f"in {len(result['actions'])} steps", flush=True)
    metrics = dict(config=config, device=str(device), real_steps=args.steps,
                   elapsed_seconds=round(time.monotonic() - started, 2),
                   initialization=first, best=best, best_step=best_steps,
                   final=history[-1]["greedy"], history=history,
                   train_episodes=counts["root_episodes"], train_wins=counts["root_wins"],
                   unfinished_steps=sum(len(episode["moves"]) for episode in slots))
    write_json(args.out / "metrics.json", metrics)
    replay(args.out, row, best, start=first, captures=captures.get(config["map"], []))
    print(f"Best greedy: {best['result']} at training step {best_steps}", flush=True)


def test(args, row, device):
    loaded = torch.load(args.checkpoint, weights_only=True, map_location=device)
    config = loaded["config"]
    expected_hash = hashlib.sha256(HTML.read_bytes()).hexdigest()
    if config["map"] != row[0]["name"] or config["data_sha256"] != expected_hash:
        raise ValueError("Checkpoint belongs to a different stage or map data version")
    policy, _ = networks(config["model"])
    policy.to(device)
    policy.load_state_dict(loaded["actor"])
    result = greedy(row, policy, config, device)
    print(f"{config['map']}: {result['result']} in {len(result['actions'])} steps")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        replay(args.out, row, result)
        write_json(args.out / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--world", choices=("A", "B"), required=True, type=str.upper)
    common.add_argument("--stage", type=int, required=True)
    common.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", parents=[common])
    training.add_argument("--out", type=Path)
    training.add_argument("--steps", type=int, default=131072)
    training.add_argument("--envs", type=int, default=8)
    training.add_argument("--horizon", type=int, default=128)
    training.add_argument("--max-episode-steps", type=int, default=250)
    training.add_argument("--epochs", type=int, default=4)
    training.add_argument("--minibatch", type=int, default=256)
    training.add_argument("--eval-every", type=int, default=8)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--entropy", type=float, default=0.015)
    training.add_argument("--seed", type=int, default=73)
    training.add_argument("--model", choices=("mlp", "cnn"), default="mlp")
    training.add_argument("--init", choices=("scratch", "ppo"), default="scratch")
    training.add_argument("--action-mask", choices=MASKS, default="safe")
    testing = sub.add_parser("test", parents=[common])
    testing.add_argument("--checkpoint", type=Path)
    testing.add_argument("--out", type=Path)
    args = parser.parse_args()
    row = configured_map(args)
    folder = ROOT / "worm_stage_runs" / row[0]["name"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(1)
    if args.command == "train":
        args.out = args.out or folder
        train(args, row, device)
    else:
        args.checkpoint = args.checkpoint or folder / "best.pt"
        test(args, row, device)


if __name__ == "__main__":
    main()
