"""Train a spatial policy on verified demonstrations and simulator-verified repairs."""

import argparse
from collections import Counter, deque
from copy import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_worm import ACTIONS, BLOCKED, ROOT, Arena, actor, dataset, frame, make_policy, search
from worm_physics import State

H, W, LIMIT = 12, 22, 250
GAMMA, STEP_COST, WIN = 0.99, 0.01, 10.0


class SpatialPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(11, 64, 3, padding=1)
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1))
            for _ in range(3))
        self.attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            64, 4, dim_feedforward=128, dropout=0, batch_first=True, norm_first=True),
            2, enable_nested_tensor=False)
        self.overflow = nn.Sequential(nn.Linear(5, 24), nn.SiLU(), nn.Linear(24, 16))
        self.actor = nn.Sequential(nn.Linear(209, 128), nn.SiLU(), nn.Linear(128, 4))
        self.critic = nn.Sequential(nn.Linear(209, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, grid, outside, outside_mask, time_left):
        x = self.stem(grid)
        for block in self.blocks:
            x = x + block(x)
        tokens = self.attention(x.flatten(2).transpose(1, 2))
        head = (tokens * grid[:, 5].flatten(1).unsqueeze(-1)).sum(1)
        encoded = self.overflow(outside).masked_fill(~outside_mask.unsqueeze(-1), -torch.inf)
        extras = encoded.amax(1)
        extras = torch.where(outside_mask.any(1, keepdim=True), extras, 0)
        embedding = torch.cat((head, tokens.mean(1), tokens.amax(1), extras, time_left), dim=1)
        return self.actor(embedding), self.critic(embedding).squeeze(-1)


class Observations:
    def __init__(self, arena):
        board = arena.board
        if (board.height, board.width) != (H, W):
            raise ValueError("Expected a 22 by 12 map")
        self.base = np.zeros((11, H, W), np.float32)
        self.base[9, :, :] = np.linspace(-1, 1, W)
        self.base[10, :, :] = np.linspace(-1, 1, H)[:, None]
        for index, points in ((0, board.walls), (3, board.hazards), (4, board.portals)):
            for x, y in points:
                self.base[index, y, x] = 1
        self.support = board.support
        self.apple_support = board.rules.apple_support

    def encode(self, state, remaining):
        grid = self.base.copy()
        overflow = []
        for channel, points in ((1, state.stones), (2, state.apples), (7, state.body)):
            length = len(points)
            for index, (x, y) in enumerate(points):
                order = (length - index) / length if channel == 7 else 0
                if 0 <= x < W and 0 <= y < H:
                    grid[channel, y, x] = order if channel == 7 else 1
                else:
                    overflow.append((x / W, y / H, order, channel / 7, 1))
        for channel, (x, y) in ((5, state.body[0]), (6, state.body[-1])):
            if 0 <= x < W and 0 <= y < H:
                grid[channel, y, x] = 1
        support = self.support | state.stones
        if self.apple_support:
            support |= state.apples
        for x, y in state.body:
            if 0 <= x < W and 0 <= y < H and (x, y - 1) in support:
                grid[8, y, x] = 1
        return grid, overflow, remaining / LIMIT


def batch_observations(observations, device):
    width = max(1, max(len(item[1]) for item in observations))
    outside = np.zeros((len(observations), width, 5), np.float32)
    mask = np.zeros((len(observations), width), bool)
    for index, (_, extras, _) in enumerate(observations):
        if extras:
            outside[index, :len(extras)] = extras
            mask[index, :len(extras)] = True
    times = np.asarray([[item[2]] for item in observations], dtype=np.float32)
    return tuple(torch.from_numpy(array).to(device) for array in (
        np.stack([item[0] for item in observations]), outside, mask, times))


def reflect(arena, trajectory):
    def point(pair):
        return W - 1 - pair[0], pair[1]

    def mirror_set(points):
        return frozenset(map(point, points))

    def mirror_state(state):
        return State(tuple(map(point, state.body)), mirror_set(state.apples), mirror_set(state.stones))

    board = copy(arena.board)
    for field in ("walls", "hazards", "portals", "support", "fixed"):
        setattr(board, field, mirror_set(getattr(board, field)))
    board.initial = mirror_state(board.initial)
    result = [(mirror_state(state), {1: 2, 2: 1}.get(action, action),
               mirror_state(after), event) for state, action, after, event in trajectory]
    return Arena(board), result


def checked(board, initial, trace, remaining):
    if not trace or len(trace) > remaining:
        return False
    state = initial
    for index, (before, action, after, event) in enumerate(trace):
        if (before != state or board.step(state, action) != (after, event)
                or event != ("win" if index == len(trace) - 1 else "ok")):
            return False
        state = after
    return True


def event_reward(event):
    terminal = {"win": WIN, "wall": -2, "body": -2, "stone": -2,
                "trap": -5, "fall": -5, "crushed": -5, "stuck": -5, "limit": -1}
    return -STEP_COST + terminal.get(event, 0)


def path_return(length):
    return (GAMMA ** (length - 1) * WIN
            - STEP_COST * (1 - GAMMA ** length) / (1 - GAMMA))


def action_options(board, state):
    return [board.step(state, action) for action in ACTIONS]


def move_mask(options):
    return np.asarray([event not in BLOCKED for _, event in options])


def sample(obs, state, remaining, target, mask, length):
    return obs.encode(state, remaining), target, mask, path_return(length)


def demonstration(arena, trace):
    obs = Observations(arena)
    for index, (state, action, _, _) in enumerate(trace):
        target = np.eye(4, dtype=np.float32)[action - 1]
        yield sample(obs, state, LIMIT - index, target,
                     move_mask(action_options(arena.board, state)), len(trace) - index)


def fit(model, optimizer, examples, repairs, count, batch_size, rng, device):
    model.train()
    loss_sum = 0.0
    for _ in range(count):
        repair_count = batch_size // 2 if repairs else 0
        rows = rng.choices(examples, k=batch_size - repair_count)
        if repair_count:
            rows += rng.choices(repairs, k=repair_count)
        observations, targets, legal, returns = zip(*rows)
        logits, values = model(*batch_observations(observations, device))
        target = torch.as_tensor(np.stack(targets), device=device)
        mask = torch.as_tensor(np.stack(legal), device=device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
        logp = logits.masked_fill(~mask, -1e6).log_softmax(dim=1)
        actor_loss = -(target * logp).sum(1).mean()
        critic_loss = F.smooth_l1_loss(values, returns)
        loss = actor_loss + 0.25 * critic_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        loss_sum += float(loss.detach())
    return loss_sum / count


@torch.inference_mode()
def play_maps(rows, model, device, *, masked=True, late_probability=0, rng=None):
    model.eval()
    active = []
    for entry, arena, trace in rows:
        start = (rng.randrange(len(trace) // 2, len(trace))
                 if rng and rng.random() < late_probability else 0)
        state = trace[start][0] if start else arena.board.initial
        active.append(dict(entry=entry, arena=arena, obs=Observations(arena), state=state,
                           seen={state}, trace=[], start=start))
    finished = []
    while active:
        remaining = [LIMIT - len(item["trace"]) - item["start"] for item in active]
        states = [item["obs"].encode(item["state"], left) for item, left in zip(active, remaining)]
        logits = model(*batch_observations(states, device))[0].cpu().numpy()
        pending = []
        for item, scores, left in zip(active, logits, remaining):
            state = item["state"]
            options = action_options(item["arena"].board, state)
            legal = move_mask(options)
            if masked and not legal.any():
                item["result"] = "stuck"
                finished.append(item)
                continue
            if masked:
                scores = np.where(legal, scores, -np.inf)
            if rng:
                scores = scores + np.asarray([-math.log(-math.log(max(1e-12, rng.random())))
                                              for _ in ACTIONS]) * 0.35
            choice = int(scores.argmax())
            after, event = options[choice]
            item["trace"].append((state, ACTIONS[choice], after, event))
            if event == "ok" and after in item["seen"]:
                event = "loop"
            if event == "ok" and left == 1:
                event = "limit"
            if event != "ok":
                item["result"] = event
                finished.append(item)
            else:
                item["state"] = after
                item["seen"].add(after)
                pending.append(item)
        active = pending
    return finished


def repair(item, index, scorer, budget):
    state, played, _, _ = item["trace"][index]
    arena = item["arena"]
    remaining = LIMIT - item["start"] - index
    options = action_options(arena.board, state)
    legal = move_mask(options)
    paths, statuses = {}, {}
    scanned = 0
    for action, (after, event) in zip(ACTIONS, options):
        prefix = [(state, action, after, event)]
        if event == "win":
            candidate = prefix
        elif event == "ok" and remaining > 1:
            rest, expansions = search(arena, scorer, budget, root=after, max_depth=remaining - 1)
            scanned += expansions
            candidate = prefix + rest if rest else []
        else:
            candidate = []
        if candidate and checked(arena.board, state, candidate, remaining):
            paths[action] = candidate
            statuses[action] = "win_verified"
        else:
            statuses[action] = ("blocked" if event in BLOCKED else
                                "dead_immediate" if event != "ok" else "unknown_budget")
    if not paths:
        return None, None, scanned
    best = min(paths, key=lambda key: len(paths[key]))
    length = len(paths[best])
    target = np.asarray([math.exp((length - len(paths[a])) / 4) if a in paths else 0
                         for a in ACTIONS], dtype=np.float32)
    target /= target.sum()
    meta = dict(map=item["entry"]["name"], start_hint=item["start"], step=index,
                played=played, taught=best, outcomes=statuses, expanded=scanned,
                state=frame(state), branches={action: [row[1] for row in path]
                                               for action, path in paths.items()})
    return sample(item["obs"], state, remaining, target, legal, length), meta, scanned


def report(rollouts):
    wins = {item["entry"]["name"]: len(item["trace"]) for item in rollouts if item["result"] == "win"}
    return dict(maps=len(rollouts), wins=len(wins), solved_steps=wins,
                outcomes=dict(Counter(item["result"] for item in rollouts)))


def baseline(rows, model, masked):
    score = actor(model)
    episodes = []
    for entry, arena, _ in rows:
        state = arena.board.initial
        seen = {state}
        trajectory = []
        for _ in range(LIMIT):
            features, options = arena.choices(state, seen)
            legal = move_mask(options)
            if masked and not legal.any():
                result = "stuck"
                break
            scores = score(features) + 8 * features[:, 12]
            if masked:
                scores = np.where(legal, scores, -np.inf)
            index = int(scores.argmax())
            after, result = options[index]
            trajectory.append((state, ACTIONS[index], after, result))
            if result != "ok":
                break
            if after in seen:
                result = "loop"
                break
            state = after
            seen.add(state)
        else:
            result = "limit"
        episodes.append(dict(entry=entry, trace=trajectory, result=result, start=0, arena=arena))
    return episodes


def attempt(item, mode):
    trace = item["trace"]
    arena = item["arena"]
    initial = trace[0][0] if trace else arena.board.initial
    frames = [frame(initial)]
    for index, (_, action, after, event) in enumerate(trace):
        if index == len(trace) - 1 and item["result"] in {"loop", "limit"}:
            event = item["result"]
        frames.append(frame(after, action=action, event=event, reward=event_reward(event)))
    return dict(mode=mode, outcome=item["result"], steps=len(trace), frames=frames)


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def checkpoint(path, model, optimizer, metrics):
    temporary = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metrics=metrics),
               temporary)
    temporary.replace(path)


def better(current, previous):
    if current["wins"] != previous["wins"]:
        return current["wins"] > previous["wins"]
    a, b = current["solved_steps"], previous["solved_steps"]
    common = a.keys() & b.keys()
    return sum(a[key] for key in common) < sum(b[key] for key in common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "worm_mimic_run")
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--rounds", type=int, default=4, help="Minimum number of repair rounds")
    parser.add_argument("--warm-epochs", type=int, default=20)
    parser.add_argument("--warm-updates", type=int, help="Updates per warm epoch")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.minutes < 0 or min(args.rounds, args.warm_epochs, args.samples, args.updates,
                              args.batch, args.threads) < 1:
        parser.error("Minutes must be nonnegative and counts must be positive")
    if args.warm_updates is not None and args.warm_updates < 1:
        parser.error("Warm updates must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("Use an empty output directory")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.zeros(1, device=device).add_(1)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    source = ROOT / "hungry_worm_map_viewer.html"
    rows = dataset(source)
    pool = [row for row in rows if row[0]["split"] == "train"]
    validation = [row for index, row in enumerate(pool, 1) if index % 7 == 0]
    train = [row for index, row in enumerate(pool, 1) if index % 7 != 0]
    test = [row for row in rows if row[0]["split"] == "test"]
    for entry, _, _ in validation:
        entry["split"] = "validation"
    demonstrations = []
    mirrored = 0
    for _, arena, trajectory in train:
        if not checked(arena.board, arena.board.initial, trajectory, LIMIT):
            raise RuntimeError("Training demonstration failed verification")
        demonstrations.extend(demonstration(arena, trajectory))
        mirrored_arena, mirrored_trace = reflect(arena, trajectory)
        if checked(mirrored_arena.board, mirrored_arena.board.initial, mirrored_trace, LIMIT):
            demonstrations.extend(demonstration(mirrored_arena, mirrored_trace))
            mirrored += 1
    old = make_policy()
    old.load_state_dict(torch.load(ROOT / "worm_ppo_run" / "round_4.pt", weights_only=True,
                                   map_location="cpu")["model"])
    teacher = actor(old)
    model = SpatialPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    config = dict(vars(args))
    config["out"] = str(args.out.resolve())
    metrics = dict(config=config, device=str(device), data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   splits=dict(Counter(entry["split"] for entry, _, _ in rows)),
                   imitation_samples=len(demonstrations), verified_mirrors=mirrored,
                   baseline_validation=report(baseline(validation, old, True)),
                   warmup=[], rounds=[], status="running")
    write_json(args.out / "metrics.json", metrics)
    best_report = None
    start = time.monotonic()
    print(f"Training on {device}, {len(demonstrations)} demonstrations, minimum {args.minutes:.1f} min",
          flush=True)
    updates = 0
    warm_path = args.out / "warm.pt"
    best_path = args.out / "best.pt"
    for epoch in range(1, args.warm_epochs + 1):
        count = args.warm_updates or math.ceil(len(demonstrations) / args.batch)
        loss = fit(model, optimizer, demonstrations, [], count, args.batch, rng, device)
        updates += count
        val = report(play_maps(validation, model, device))
        metrics["warmup"].append(dict(epoch=epoch, loss=loss, validation=val, updates=updates))
        if best_report is None or better(val, best_report):
            best_report = val
            metrics["selected"] = f"warm {epoch}"
            checkpoint(warm_path, model, optimizer, metrics)
            checkpoint(best_path, model, optimizer, metrics)
        metrics["elapsed_seconds"] = round(time.monotonic() - start, 2)
        write_json(args.out / "metrics.json", metrics)
        print(f"Warm {epoch}/{args.warm_epochs} loss={loss:.4f} validation={val['wins']}/{val['maps']}",
              flush=True)

    saved = torch.load(warm_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    control = SpatialPolicy().to(device)
    control.load_state_dict(saved["model"])
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=3e-4)
    control_optimizer.load_state_dict(saved["optimizer"])
    control_rng = random.Random(args.seed)
    control_best = validation_start = report(play_maps(validation, model, device))
    control_path = args.out / "control.pt"
    checkpoint(control_path, control, control_optimizer, metrics)
    repairs = deque(maxlen=12000)
    seen_repairs = set()
    round_no = 0
    metrics["warm_validation"] = validation_start
    replay_samples = {}
    with (args.out / "episodes.jsonl").open("w", encoding="utf-8", buffering=1) as episodes, (
        args.out / "repairs.jsonl").open("w", encoding="utf-8", buffering=1) as repair_log:
        while round_no < args.rounds or time.monotonic() - start < args.minutes * 60:
            round_no += 1
            subset = rng.sample(train, min(40, len(train)))
            plays = []
            for offset in range(0, len(subset), 16):
                plays.extend(play_maps(subset[offset:offset + 16], model, device,
                                       late_probability=(0.35, 0.20, 0.10, 0)[min(round_no - 1, 3)],
                                       rng=rng))
            candidates = []
            for item in plays:
                entry, trace = item["entry"], item["trace"]
                episodes.write(json.dumps(dict(round=round_no, map=entry["name"],
                                               start_hint=item["start"], outcome=item["result"],
                                               actions=[step[1] for step in trace])) + "\n")
                if not trace:
                    continue
                priority = list(dict.fromkeys([len(trace) - 1, len(trace) // 2, 0]))
                priority.extend(rng.sample(range(len(trace)), min(4, len(trace))))
                for index in dict.fromkeys(priority):
                    candidates.append((item["result"] == "win", item, index))
            # Failed rollouts supply the first states queried in each round.
            rng.shuffle(candidates)
            candidates.sort(key=lambda candidate: candidate[0])
            added = scanned = queried = 0
            for _, item, index in candidates[:args.samples]:
                state = item["trace"][index][0]
                key = (item["entry"]["name"], state, item["start"] + index)
                if key in seen_repairs:
                    continue
                queried += 1
                budget = 1024 if round_no > 1 and queried % 5 == 0 else 256
                example, metadata, nodes = repair(item, index, teacher, budget)
                scanned += nodes
                if example is not None:
                    seen_repairs.add(key)
                    repairs.append(example)
                    added += 1
                    repair_log.write(json.dumps(dict(round=round_no, **metadata)) + "\n")
            loss = fit(model, optimizer, demonstrations, list(repairs), args.updates,
                       args.batch, rng, device)
            control_loss = fit(control, control_optimizer, demonstrations, [], args.updates,
                               args.batch, control_rng, device)
            updates += args.updates
            val = report(play_maps(validation, model, device))
            control_val = report(play_maps(validation, control, device))
            elapsed = round(time.monotonic() - start, 2)
            metrics["elapsed_seconds"] = elapsed
            metrics["rounds"].append(dict(round=round_no, elapsed_seconds=elapsed, loss=loss,
                                           states_queried=queried, repairs=added, nodes=scanned,
                                           episodes=len(plays), environment_steps=sum(
                                               len(item["trace"]) for item in plays),
                                           updates=updates, validation=val,
                                           control_loss=control_loss, control_validation=control_val))
            if better(control_val, control_best):
                control_best = control_val
                metrics["selected_control"] = round_no
                checkpoint(control_path, control, control_optimizer, metrics)
            if better(val, best_report):
                best_report = val
                metrics["selected"] = f"repair {round_no}"
                checkpoint(best_path, model, optimizer, metrics)
            checkpoint(args.out / "latest.pt", model, optimizer, metrics)
            checkpoint(args.out / f"round_{round_no:03}.pt", model, optimizer, metrics)
            write_json(args.out / "metrics.json", metrics)
            for item in plays:
                key = item["entry"]["name"]
                if key not in replay_samples or item["result"] == "win":
                    replay_samples[key] = attempt(item, f"mimic_round_{round_no}")
            print(f"Round {round_no} elapsed={elapsed:.0f}s repairs={added} "
                  f"updates={updates} validation={val['wins']}/{val['maps']}", flush=True)

    before = baseline(test, old, True)
    old_unmasked = baseline(test, old, False)
    model.load_state_dict(torch.load(warm_path, map_location=device, weights_only=True)["model"])
    warm_test = play_maps(test, model, device)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model"])
    after = play_maps(test, model, device)
    unmasked = play_maps(test, model, device, masked=False)
    control.load_state_dict(torch.load(control_path, map_location=device, weights_only=True)["model"])
    control_test = play_maps(test, control, device)
    metrics["historical_test"] = dict(baseline=report(before), baseline_unmasked=report(old_unmasked),
                                      warm=report(warm_test), selected=report(after),
                                      selected_unmasked=report(unmasked), control=report(control_test))
    for mode, evaluated in (("mimic_baseline", before), ("mimic_warm", warm_test),
                            ("mimic_selected", after), ("mimic_unmasked", unmasked),
                            ("mimic_control", control_test)):
        for item in evaluated:
            item["entry"].setdefault("attempts", []).append(attempt(item, mode))
    for item in play_maps(validation, model, device):
        item["entry"]["attempts"] = [attempt(item, "mimic_selected")]
    for entry, _, _ in train:
        if entry["name"] in replay_samples:
            entry["attempts"] = [replay_samples[entry["name"]]]
    metrics["status"] = "completed"
    write_json(args.out / "replay.json", dict(info=dict(mimic=metrics, splits=metrics["splits"]),
                                             maps=[entry for entry, _, _ in rows]))
    from build_worm_training_replay import build
    build(source, args.out / "replay.json", args.out / "replay.html")
    write_json(args.out / "metrics.json", metrics)
    print(f"Completed after {metrics['elapsed_seconds']:.1f}s of training. "
          f"Selected {metrics['selected']}, test {metrics['historical_test']['selected']['wins']}/"
          f"{len(test)}", flush=True)


if __name__ == "__main__":
    main()
