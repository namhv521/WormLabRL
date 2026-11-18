"""Continue the WormLabRL baseline with several measured rounds of masked PPO."""

import argparse
import json
import time
from collections import Counter
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_worm import (
    ACTIONS, FEATURES, ROOT, actor, dataset, evaluate, examples, frame,
    make_policy, packed, play, reward,
)


GAMMA = 0.99
GAE_LAMBDA = 0.95
LIMIT = 250


def allowed(features):
    safe = ~(features[..., 10].astype(bool) | features[..., 11].astype(bool))
    legal = ~features[..., 10].astype(bool)
    return np.where(safe.any(axis=-1, keepdims=True), safe,
                    np.where(legal.any(axis=-1, keepdims=True), legal, True))


def masked_score(score):
    def score_safe(features):
        return np.where(allowed(features), score(features), -1e4)
    return score_safe


def masked_logits(policy, features, mask):
    raw = policy(features).squeeze(-1) + 8 * features[..., 12]
    return raw.masked_fill(~mask, -1e4)


class Rollouts:
    def __init__(self, rows, rng, output, *, max_steps=LIMIT):
        self.rows = rows
        self.rng = rng
        self.output = output
        self.max_steps = max_steps
        self.cache = lru_cache(maxsize=85000)(lambda arena, state: arena.choices(state))
        self.order = []
        self.round = 0

    def episode(self, start_late):
        if not self.order:
            self.order = self.rng.permutation(len(self.rows)).tolist()
        index = self.order.pop()
        row, arena, reference = self.rows[index]
        hint_index = (int(self.rng.integers(max(1, len(reference) // 3), len(reference)))
                      if len(reference) > 1 and self.rng.random() < start_late else 0)
        state = reference[hint_index][0] if hint_index else arena.board.initial
        return dict(index=index, name=row["name"], arena=arena, hint=hint_index,
                    state=state, seen={state}, moves=[], total=0.0, round=self.round)

    def observation(self, episode):
        features, transitions = self.cache(episode["arena"], episode["state"])
        features = features.copy()
        features[:, 22] = [after in episode["seen"] for after, _ in transitions]
        return features, transitions

    def step(self, episode, choice, transitions, start_late):
        state = episode["state"]
        after, event = transitions[choice]
        repeated = after in episode["seen"]
        points = reward(event, state, after, episode["seen"], episode["arena"])
        episode["moves"].append(str(ACTIONS[choice]))
        episode["total"] += points
        if event == "ok" and repeated:
            event = "loop"
        if event == "ok" and len(episode["moves"]) >= self.max_steps:
            event = "limit"
        if event == "ok":
            episode["seen"].add(after)
            episode["state"] = after
            return episode, points, None

        record = dict(map=episode["name"], round=episode["round"], start_hint=episode["hint"],
                      actions="".join(episode["moves"]), result=event,
                      reward=round(episode["total"], 3))
        self.output.write(json.dumps(record, separators=(",", ":")) + "\n")
        if event == "loop":
            points -= 0.65
        elif event == "limit":
            points -= 0.5
        return self.episode(start_late), points, record


def collect(rollouts, slots, policy, critic, horizon, start_late, counts, captures,
            *, action_mask=allowed, device="cpu"):
    rows = []
    for _ in range(horizon):
        observed = [rollouts.observation(ep) for ep in slots]
        inputs = np.stack([item[0] for item in observed])
        features = torch.from_numpy(inputs).to(device)
        mask = torch.from_numpy(action_mask(inputs)).to(device)
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=masked_logits(policy, features, mask))
            action = dist.sample()
            old_log = dist.log_prob(action)
            value = critic(features.flatten(1)).squeeze(-1)
        points, done = [], []
        for index, choice in enumerate(action.tolist()):
            slots[index], received, record = rollouts.step(
                slots[index], choice, observed[index][1], start_late)
            points.append(received)
            done.append(record is not None)
            if record is not None:
                key = "hint_start" if record["start_hint"] else "root"
                counts[key + "_episodes"] += 1
                counts[key + "_wins"] += record["result"] == "win"
                if not record["start_hint"]:
                    saved = captures.setdefault(record["map"], [])
                    if len(saved) < 2 or (record["result"] == "win"
                                          and not any(r["result"] == "win" for r in saved)):
                        saved.append(record)
        rows.append((features, mask, action, old_log, value,
                     torch.tensor(points, device=device), torch.tensor(done, device=device)))
    arrays = tuple(torch.stack(items) for items in zip(*rows))
    obs, mask, actions, old_logs, values, rewards, dones = arrays
    with torch.no_grad():
        final_obs = torch.from_numpy(np.stack([rollouts.observation(ep)[0] for ep in slots])).to(device)
        next_value = critic(final_obs.flatten(1)).squeeze(-1)
    advantages = torch.empty_like(rewards)
    running = torch.zeros_like(next_value)
    for t in reversed(range(horizon)):
        continuation = (~dones[t]).float()
        delta = rewards[t] + GAMMA * next_value * continuation - values[t]
        running = delta + GAMMA * GAE_LAMBDA * continuation * running
        advantages[t] = running
        next_value = values[t]
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    return tuple(t.flatten(0, 1) for t in (obs, mask, actions, old_logs, advantages, returns))


def update(policy, critic, optimizer, data, demo, epochs, anchor, *,
           minibatch=256, entropy=0.015):
    obs, masks, actions, old_logs, advantages, returns = data
    for _ in range(epochs):
        for ids in torch.randperm(len(obs), device=obs.device).split(minibatch):
            logits = masked_logits(policy, obs[ids], masks[ids])
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(actions[ids]) - old_logs[ids]).exp()
            advantage = advantages[ids]
            ppo_loss = -torch.minimum(ratio * advantage,
                                      ratio.clamp(0.8, 1.2) * advantage).mean()
            value_loss = F.mse_loss(critic(obs[ids].flatten(1)).squeeze(-1), returns[ids])
            loss = ppo_loss + 0.5 * value_loss - entropy * dist.entropy().mean()
            if anchor:
                demo_obs, demo_actions = demo
                sample = torch.randint(len(demo_obs), (96,), device=demo_obs.device)
                imitation = F.cross_entropy(policy(demo_obs[sample]).squeeze(-1), demo_actions[sample])
                loss = loss + anchor * imitation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 0.5)
            optimizer.step()


def compare(rows, policy, label):
    raw = evaluate(rows, policy, label)
    score = masked_score(actor(policy))
    masked = Counter(play(arena, score, LIMIT)[0] for _, arena, _ in rows)
    raw["masked_wins"] = masked["win"]
    raw["masked_failures"] = {kind: n for kind, n in masked.items() if kind != "win"}
    return raw


def recorded_replay(record, arena):
    state = arena.board.initial
    seen = {state}
    frames = [frame(state)]
    for value in record["actions"]:
        action = int(value)
        after, event = arena.board.step(state, action)
        points = reward(event, state, after, seen, arena)
        frames.append(frame(after, action=action, event=event, reward=points))
        state = after
        seen.add(state)
    frames[-1]["event"] = record["result"]
    if record["result"] == "loop":
        frames[-1]["reward"] -= 0.65
    elif record["result"] == "limit":
        frames[-1]["reward"] -= 0.5
    return dict(mode=f"ppo_{record['round']}", outcome=record["result"],
                steps=len(record["actions"]), frames=frames)


def save_checkpoint(path, policy, critic, optimizer, metrics):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": policy.state_dict(), "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(), "metrics": metrics}, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps-per-round", type=int, default=512000)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--out", default=ROOT / "worm_ppo_run", type=type(ROOT))
    parser.add_argument("--resume", type=type(ROOT), help="PPO round checkpoint")
    args = parser.parse_args()
    if args.resume and (args.out.resolve() == ROOT / "worm_ppo_run"
                        or (args.out.exists() and any(args.out.iterdir()))):
        parser.error("--resume requires a separate empty --out directory")
    if min(args.rounds, args.steps_per_round, args.envs, args.horizon, args.ppo_epochs) < 1:
        parser.error("rounds, steps, environments, horizon and epochs must be positive")
    batch_size = args.envs * args.horizon
    if args.steps_per_round % batch_size:
        parser.error("steps per round must be divisible by envs times horizon")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    entries = dataset(ROOT / "hungry_worm_map_viewer.html")
    train_pool = [row for row in entries if row[0]["split"] == "train"]
    train = [row for i, row in enumerate(train_pool, 1) if i % 7]
    validation = [row for i, row in enumerate(train_pool, 1) if not i % 7]
    test = [row for row in entries if row[0]["split"] == "test"]
    for entry, _, _ in validation:
        entry["split"] = "validation"
    demo = packed([example for _, arena, trace in train for example in examples(arena, trace)])
    policy = make_policy()
    warm_path = args.resume or ROOT / "worm_run" / "policy.pt"
    warm = torch.load(warm_path, weights_only=True, map_location="cpu")
    policy.load_state_dict(warm["model"])
    critic = nn.Sequential(nn.Linear(4 * FEATURES, 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.AdamW([*policy.parameters(), *critic.parameters()], lr=2.5e-4)
    start_round, prior_steps = 0, 0
    if args.resume:
        previous = warm["metrics"]["rounds"][-1]
        start_round, prior_steps = previous["round"], previous["total_steps"]
        critic.load_state_dict(warm["critic"])
        optimizer.load_state_dict(warm["optimizer"])
    metrics = dict(start_round=start_round, prior_steps=prior_steps,
                   config=dict(seed=args.seed, rounds=args.rounds, steps_per_round=args.steps_per_round,
                               envs=args.envs, horizon=args.horizon, ppo_epochs=args.ppo_epochs,
                               resume=str(args.resume.resolve()) if args.resume else None),
                   splits=dict(Counter(row[0]["split"] for row in entries)),
                   baseline_validation=compare(validation, policy, "baseline"),
                   timeline=[compare(test, policy, "before")], rounds=[])
    captures = {}
    started = time.monotonic()
    print(f"Train={len(train)} validation={len(validation)} test={len(test)} "
          f"target_steps={args.rounds * args.steps_per_round}", flush=True)
    with (args.out / "episodes.jsonl").open("w", encoding="utf-8", buffering=262144) as output:
        rollouts = Rollouts(train, rng, output)
        for round_no in range(start_round + 1, start_round + args.rounds + 1):
            rollouts.round = round_no
            start_late = (0.65, 0.35, 0.15, 0.0)[min(round_no - 1, 3)]
            anchor = (0.32, 0.24, 0.18, 0.14)[min(round_no - 1, 3)]
            slots = [rollouts.episode(start_late) for _ in range(args.envs)]
            counts = Counter()
            round_captures = {}
            for step in range(batch_size, args.steps_per_round + 1, batch_size):
                data = collect(rollouts, slots, policy, critic, args.horizon,
                               start_late, counts, round_captures)
                update(policy, critic, optimizer, data, demo, args.ppo_epochs, anchor)
                if step % (batch_size * 25) == 0:
                    output.flush()
                    print(f"Round {round_no} {step}/{args.steps_per_round}: "
                          f"{counts['root_wins']}/{counts['root_episodes']} root wins; "
                          f"{time.monotonic() - started:.0f}s total", flush=True)
            captures[round_no] = round_captures
            val = compare(validation, policy, f"round_{round_no}")
            heldout = compare(test, policy, f"round_{round_no}")
            total = prior_steps + (round_no - start_round) * args.steps_per_round
            metrics["rounds"].append(dict(
                round=round_no, total_steps=total, hint_start_probability=start_late,
                completed=dict(counts), unfinished_episodes=len(slots),
                steps_in_unfinished_episodes=sum(len(slot["moves"]) for slot in slots),
                validation=val, test=heldout))
            metrics["timeline"].append(heldout)
            save_checkpoint(args.out / f"round_{round_no}.pt", policy, critic, optimizer, metrics)
            (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"Completed round {round_no}: validation raw/masked {val['wins']}/{val['masked_wins']}, "
                  f"test raw/masked {heldout['wins']}/{heldout['masked_wins']} of {len(test)}", flush=True)

    baseline = metrics["baseline_validation"]
    best = max([(start_round, baseline)] + [(row["round"], row["validation"]) for row in metrics["rounds"]],
               key=lambda item: (item[1]["wins"], item[1]["masked_wins"], item[1]["hint_accuracy"]))
    metrics["selected_round"] = best[0]
    metrics["selected_test"] = metrics["timeline"][best[0] - start_round]
    metrics["elapsed_seconds"] = round(time.monotonic() - started)
    if best[0] != start_round:
        selected = torch.load(args.out / f"round_{best[0]}.pt", weights_only=True, map_location="cpu")
    else:
        selected = warm
    policy.load_state_dict(selected["model"])
    torch.save({"model": selected["model"], "metrics": metrics}, args.out / "best.pt")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for entry, arena, _ in train:
        attempts = []
        for round_no in captures:
            for record in captures[round_no].get(entry["name"], []):
                attempts.append(recorded_replay(record, arena))
        if attempts:
            entry["attempts"] = attempts
    add_test_replays(args.out, [row for row, _, _ in test], metrics, test)
    summary = dict(metrics, resolved_by={})
    replay = dict(info=summary, maps=[entry for entry, _, _ in entries])
    (args.out / "replay.json").write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
    print(f"Selected round {best[0]} on validation; finished {args.rounds * args.steps_per_round} steps",
          flush=True)


def add_test_replays(output, rows, metrics, test):
    resume = metrics["config"].get("resume")
    baseline = torch.load(resume or ROOT / "worm_run" / "policy.pt", weights_only=True,
                          map_location="cpu")
    start_round = metrics.get("start_round", 0)
    baseline_mode = f"ppo_eval_{start_round}" if start_round else "ppo_baseline"
    scores = [(baseline_mode, actor(make_loaded(baseline["model"])))]
    for checkpoint in metrics["rounds"]:
        index = checkpoint["round"]
        saved = torch.load(output / f"round_{index}.pt", weights_only=True, map_location="cpu")
        scores.append((f"ppo_eval_{index}", actor(make_loaded(saved["model"]))))
    scores.append(("ppo_masked_last", masked_score(scores[-1][1])))
    arenas = {row["name"]: arena for row, arena, _ in test}
    for row in rows:
        arena = arenas[row["name"]]
        attempts = []
        for mode, score in scores:
            outcome, steps, frames, _ = play(arena, score, LIMIT, log=True)
            attempts.append(dict(mode=mode, outcome=outcome, steps=steps, frames=frames))
        row["attempts"] = attempts


def make_loaded(weights):
    model = make_policy()
    model.load_state_dict(weights)
    return model


if __name__ == "__main__":
    main()
