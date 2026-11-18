"""Pilot: fine-tune PPO round 4 on complete, verified recovery paths."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from build_worm_training_replay import build
from train_worm import (
    ACTIONS, ROOT, actor, dataset, evaluate, examples, make_policy, play, search, trace_frames,
)
from train_worm_mimic import checked, write_json

LIMIT = 250
HTML = ROOT / "hungry_worm_map_viewer.html"
BASELINE = ROOT / "worm_ppo_run" / "round_4.pt"


def transcript(experience):
    return [(state, ACTIONS[choice], after, event)
            for state, _, choice, _, after, event in experience]


def recover(arena, original, teacher, budget):
    nodes = 0
    # Repair near the failure first, then backtrack if that branch cannot be recovered.
    for position in dict.fromkeys(max(0, len(original) - depth) for depth in (1, 3, 6, 12, 24, LIMIT)):
        prefix = original[:position]
        state = original[position][0]
        previous = {step[0] for step in prefix}
        path, expanded = search(arena, teacher, budget, root=state,
                                max_depth=LIMIT - position, avoid=previous)
        nodes += expanded
        if not path or path[0][1] == original[position][1]:
            continue
        result = prefix + path
        if checked(arena.board, arena.board.initial, result, LIMIT):
            return position, result, nodes
    return None, [], nodes


def evaluation(rows, model):
    score = actor(model)
    outcomes = Counter()
    solved = {}
    for entry, arena, _ in rows:
        outcome, steps, _, _ = play(arena, score, LIMIT)
        outcomes[outcome] += 1
        if outcome == "win":
            solved[entry["name"]] = steps
    return dict(maps=len(rows), wins=len(solved), outcomes=dict(outcomes), solved_steps=solved)


def rank(result, baseline):
    solved = result["solved_steps"]
    original = baseline["solved_steps"]
    retained = solved.keys() & original.keys()
    return result["wins"], len(retained), sum(original[k] - solved[k] for k in retained)


def draw(rows, count, rng):
    sample = [rows[i] for i in rng.integers(len(rows), size=count)]
    ob, action = zip(*sample)
    return torch.from_numpy(np.stack(ob)), torch.tensor(action)


def fine_tune(model, original, optimizer, repairs, anchor, constraint, updates, rng):
    for _ in range(updates):
        x, y = draw(repairs, 96, rng)
        retained, actions = draw(anchor, 96, rng)
        logits = model(x).squeeze(-1) + 8 * x[:, :, 12]
        preserved = model(retained).squeeze(-1) + 8 * retained[:, :, 12]
        with torch.no_grad():
            before = original(retained).squeeze(-1) + 8 * retained[:, :, 12]
        loss = (F.cross_entropy(logits, y)
                + constraint * F.cross_entropy(preserved, actions)
                + 2 * F.kl_div(preserved.log_softmax(1), before.softmax(1), reduction="batchmean"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


def save_weights(out, name, model, source, validation):
    torch.save(dict(model=model.state_dict(), source=source, validation=validation), out / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "worm_recovery_pilot")
    parser.add_argument("--seeds", type=int, nargs="+", default=[73, 89, 101])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=1024)
    parser.add_argument("--maps-per-round", type=int, default=96)
    args = parser.parse_args()
    if min(args.rounds, args.nodes, args.maps_per_round) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive counts and distinct seeds")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("Use an empty output directory")
    torch.set_num_threads(1)
    rows = dataset(HTML)
    pool = [row for row in rows if row[0]["split"] == "train"]
    train = [row for index, row in enumerate(pool, 1) if index % 7]
    validation = [row for index, row in enumerate(pool, 1) if not index % 7]
    test = [row for row in rows if row[0]["split"] == "test"]
    assert len(train) == 141 and len(validation) == 23 and len(test) == 40
    baseline = make_policy()
    baseline.load_state_dict(torch.load(BASELINE, map_location="cpu", weights_only=True)["model"])
    teacher = actor(baseline)
    baseline_validation = evaluation(validation, baseline)
    anchor = []
    for _, arena, _ in train:
        outcome, _, _, experience = play(arena, teacher, LIMIT)
        if outcome == "win":
            trace = transcript(experience)
            if not checked(arena.board, arena.board.initial, trace, LIMIT):
                raise RuntimeError("Baseline win failed replay")
            anchor.extend(examples(arena, trace))

    expected_base = rank(baseline_validation, baseline_validation)
    records = []
    selected, selected_val, selected_source = deepcopy(baseline), baseline_validation, "ppo_round_4"
    best_candidate, best_candidate_val, best_source = None, None, None
    replay_pairs = {}
    cache = {}
    metrics = dict(baseline_validation=baseline_validation, rounds=records,
                   seeds=args.seeds, nodes_per_search=args.nodes,
                   split_maps=dict(train=[row[0]["name"] for row in train],
                                   validation=[row[0]["name"] for row in validation],
                                   historical_test=[row[0]["name"] for row in test]),
                   anchor_steps=len(anchor),
                   checkpoint_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
                   data_sha256=hashlib.sha256(HTML.read_bytes()).hexdigest())
    write_json(args.out / "metrics.json", metrics)
    print(f"Baseline {baseline_validation['wins']}/23, preserved {len(anchor)} winning steps", flush=True)
    with (args.out / "repairs.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            model = deepcopy(baseline)
            # A fixed anchor and small updates preserve the behavior of PPO's successful routes.
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            repairs = []
            used = set()
            seed_best = None
            for number in range(1, args.rounds + 1):
                score = actor(model)
                new, nodes, ignored = 0, 0, 0
                failures = Counter()
                for index in rng.permutation(len(train))[:args.maps_per_round]:
                    entry, arena, _ = train[index]
                    outcome, _, frames, experience = play(arena, score, LIMIT, log=True)
                    if outcome == "win":
                        continue
                    failures[outcome] += 1
                    original = transcript(experience)
                    actions = tuple(step[1] for step in original)
                    key = entry["name"], actions
                    if key in used:
                        ignored += 1
                        continue
                    used.add(key)
                    if key in cache:
                        position, path = cache[key]
                    else:
                        position, path, expanded = recover(arena, original, teacher, args.nodes)
                        nodes += expanded
                        cache[key] = position, path
                    if not path:
                        continue
                    # Carry the visited states from the whole rollout into each branch feature.
                    repairs.extend(list(examples(arena, path))[position:])
                    new += 1
                    log.write(json.dumps(dict(seed=seed, round=number, map=entry["name"],
                                              failure=outcome, pivot=position,
                                              played=list(actions),
                                              recovered=[step[1] for step in path])) + "\n")
                    if entry["name"] not in replay_pairs and len(replay_pairs) < 40:
                        replay_pairs[entry["name"]] = [
                            dict(mode="policy", outcome=outcome, steps=len(original), frames=frames),
                            dict(mode="search", outcome="win", steps=len(path),
                                 frames=trace_frames(arena, path))]
                if repairs:
                    fine_tune(model, baseline, optimizer, repairs, anchor, 1.0, 100, rng)
                validation_score = evaluation(validation, model)
                candidate = dict(seed=seed, round=number, repairs=new, repair_steps=len(repairs),
                                 new_nodes=nodes, duplicates=ignored, failures=dict(failures),
                                 training_updates=100 if repairs else 0, validation=validation_score)
                records.append(candidate)
                source = f"seed_{seed}_round_{number}"
                if seed_best is None or rank(validation_score, baseline_validation) > rank(seed_best, baseline_validation):
                    seed_best = validation_score
                    save_weights(args.out, f"seed_{seed}.pt", model, source, validation_score)
                if (best_candidate_val is None or rank(validation_score, baseline_validation)
                        > rank(best_candidate_val, baseline_validation)):
                    best_candidate, best_candidate_val, best_source = (
                        deepcopy(model), validation_score, source)
                if rank(validation_score, baseline_validation) > rank(selected_val, baseline_validation):
                    selected, selected_val, selected_source = deepcopy(model), validation_score, source
                write_json(args.out / "metrics.json", metrics)
                print(f"Seed {seed} round {number}: +{new} recovery paths, "
                      f"validation {validation_score['wins']}/23", flush=True)

    save_weights(args.out, "selected.pt", selected, selected_source, selected_val)
    save_weights(args.out, "best_candidate.pt", best_candidate, best_source, best_candidate_val)
    metrics.update(selected=selected_source, selected_validation=selected_val,
                   best_candidate=best_source, best_candidate_validation=best_candidate_val,
                   improving_seeds=sum(any(r["seed"] == seed and r["validation"]["wins"] > expected_base[0]
                                           for r in records) for seed in args.seeds))
    # Test runs are delayed until both models have been selected on validation.
    timeline = [evaluate(test, model, label, include_replay=True)
                for model, label in ((baseline, "before"), (best_candidate, "after"))]
    metrics["historical_test"] = dict(baseline=evaluation(test, baseline),
                                      best_candidate=evaluation(test, best_candidate),
                                      selected=evaluation(test, selected))
    metrics["status"] = "completed"
    for entry, _, _ in train:
        if entry["name"] in replay_pairs:
            entry["attempts"] = replay_pairs[entry["name"]]
    info = dict(splits=dict(Counter(entry["split"] for entry, _, _ in rows)),
                timeline=timeline, resolved_by={}, recovery_pilot=True,
                recovery_paths=sum(round_data["repairs"] for round_data in records))
    replay_path = args.out / "replay.json"
    write_json(replay_path, dict(info=info, maps=[entry for entry, _, _ in rows]))
    build(HTML, replay_path, args.out / "replay.html")
    write_json(args.out / "metrics.json", metrics)
    print(f"Selected {selected_source}; candidate test {timeline[1]['wins']}/40", flush=True)


if __name__ == "__main__":
    main()
