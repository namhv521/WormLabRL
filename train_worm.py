"""Train a CPU policy from play and verified HTML demonstrations, with full replay."""

import argparse
import hashlib
import heapq
import itertools
import json
import random
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from worm_physics import DIRECTIONS, WormMap, load_maps, trace_hints


ROOT = Path(__file__).resolve().parent
ACTIONS = tuple(DIRECTIONS)
BLOCKED = {"wall", "body", "stone"}
FATAL = {"trap", "fall", "crushed"}
TERMINAL = BLOCKED | FATAL | {"win"}
FEATURES = 23


class Arena:
    def __init__(self, board):
        self.board = board
        self.goal = min(board.portals)
        self.distance = {p: 0 for p in board.portals}
        queue = deque(sorted(board.portals))
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS.values():
                p = x + dx, y + dy
                if (0 <= p[0] < board.width and 0 <= p[1] < board.height
                        and p not in self.distance and p not in board.walls | board.hazards):
                    self.distance[p] = self.distance[(x, y)] + 1
                    queue.append(p)

    def dist(self, point):
        return self.distance.get(point, 50 + abs(point[0] - self.goal[0])
                                 + abs(point[1] - self.goal[1]))

    def choices(self, state, seen=None):
        board = self.board
        x, y = state.body[0]
        gx, gy = self.goal
        distance = self.dist((x, y))
        cells = []
        transitions = []
        occupied = set(state.body)
        supports = board.support | state.apples | state.stones
        for action, (dx, dy) in DIRECTIONS.items():
            target = x + dx, y + dy
            after, event = board.step(state, action)
            nx, ny = after.body[0]
            transitions.append((after, event))
            feature = [
                dx, dy, (gx - x) / 22, (gy - y) / 12,
                (gx - x) * dx / 22, (gy - y) * dy / 12,
                max(-1, min(1, (distance - self.dist((nx, ny))) / 4)),
                min(distance / 50, 2), (ny - y) / 6, len(state.body) / 20,
                event in BLOCKED, event in FATAL, event == "win",
                len(after.apples) < len(state.apples), after.stones != state.stones,
                target in board.walls, target in board.hazards, target in occupied,
                target in state.stones, target in state.apples,
                sum((bx, by - 1) in supports for bx, by in after.body) / len(after.body),
                min((abs(nx - ax) + abs(ny - ay) for ax, ay in after.apples), default=0) / 22,
                seen is not None and after in seen,
            ]
            cells.append(feature)
        return np.asarray(cells, dtype=np.float32), transitions


def make_policy():
    return nn.Sequential(nn.Linear(FEATURES, 32), nn.Tanh(), nn.Linear(32, 1))


def actor(model):
    first, _, last = model
    weights = first.weight.detach().numpy().T.copy()
    bias = first.bias.detach().numpy().copy()
    output = last.weight.detach().numpy().T.copy()
    return lambda features: (np.tanh(features @ weights + bias) @ output).ravel()


def frame(state, **extra):
    return dict(body=state.body, apples=sorted(state.apples), stones=sorted(state.stones), **extra)


def reward(event, previous, state, visited, arena):
    if event == "win":
        return 4.0
    if event in TERMINAL:
        return -2.0
    delta = (arena.dist(previous.body[0]) - arena.dist(state.body[0])) * 0.07
    return round(-0.025 + max(-0.25, min(0.25, delta))
                 + 0.3 * (len(previous.apples) > len(state.apples))
                 - 0.35 * (state in visited), 3)


def play(arena, score, limit, rng=None, penalties=None, log=False):
    state = arena.board.initial
    visited = {state}
    frames = [frame(state)] if log else []
    experience = []
    event = "limit"
    for step in range(1, limit + 1):
        features, options = arena.choices(state, visited)
        logits = score(features)
        if penalties:
            logits -= penalties.get(state, np.zeros(4, dtype=np.float32))
        # A fixed safe-exit prior makes a visible portal an absorbing goal.
        logits += 8 * features[:, 12]
        if rng is None:
            choice = int(logits.argmax())
        else:
            probs = np.exp((logits - logits.max()) / 0.85)
            choice = int(rng.choice(4, p=probs / probs.sum()))
        after, event = options[choice]
        points = reward(event, state, after, visited, arena)
        experience.append((state, features, choice, points, after, event))
        repeated = after in visited and event == "ok"
        if repeated:
            event = "loop"
        if log:
            frames.append(frame(after, action=ACTIONS[choice], event=event, reward=points))
        if event != "ok":
            break
        visited.add(after)
        state = after
    else:
        event = "limit"
        if log:
            frames[-1]["event"] = event
    return event, step, frames, experience


def learn_play(model, optimizer, experience, penalties, won):
    _, features, choices, rewards, _, _ = zip(*experience)
    tail = 0.0
    returns = []
    for points in reversed(rewards):
        tail = points + 0.88 * tail
        returns.append(tail)
    logits = model(torch.from_numpy(np.asarray(features))).squeeze(-1)
    logits = logits + 8 * torch.from_numpy(np.asarray(features)[:, :, 12])
    losses = F.cross_entropy(logits, torch.tensor(choices), reduction="none")
    weights = torch.tensor(returns[::-1], dtype=torch.float32).clamp(-2, 2)
    optimizer.zero_grad()
    (weights * losses).mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    if not won:
        for offset, (state, _, choice, _, _, _) in enumerate(reversed(experience[-5:])):
            vector = penalties.setdefault(state, np.zeros(4, dtype=np.float32))
            vector[choice] += (1.4 if offset == 0 else 0.35 / offset)


def search(arena, score, budget, *, root=None, max_depth=250, avoid=()):
    """Bounded best-first search. Policy ranks candidates and the simulator checks them."""
    if root is None:
        root = arena.board.initial
    serial = itertools.count()
    queue = [(0.0, next(serial), root, 0)]
    parent = {root: None}
    forbidden = frozenset(avoid)
    for visited in range(1, budget + 1):
        if not queue:
            return [], visited - 1
        _, _, state, depth = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        features, transitions = arena.choices(state)
        scores = score(features)
        for idx, (after, event) in enumerate(transitions):
            if event == "win":
                trace = [(state, ACTIONS[idx], after, event)]
                node = state
                while parent[node] is not None:
                    previous, action = parent[node]
                    trace.append((previous, action, node, "ok"))
                    node = previous
                return trace[::-1], visited
            if event != "ok" or after in parent or after in forbidden:
                continue
            parent[after] = state, ACTIONS[idx]
            priority = 0.65 * (depth + 1) + 1.5 * arena.dist(after.body[0]) - 2 * scores[idx]
            heapq.heappush(queue, (priority, next(serial), after, depth + 1))
    return [], budget


def trace_frames(arena, trace):
    frames = [frame(arena.board.initial)]
    seen = {arena.board.initial}
    for state, action, after, event in trace:
        frames.append(frame(after, action=action, event=event,
                            reward=reward(event, state, after, seen, arena)))
        seen.add(after)
    return frames


def examples(arena, trace):
    visited = set()
    for state, action, _, _ in trace:
        visited.add(state)
        yield arena.choices(state, visited)[0], action - 1


def packed(items):
    features, actions = zip(*items)
    return torch.from_numpy(np.asarray(features)), torch.tensor(actions)


def learn_batch(model, optimizer, items):
    features, labels = packed(items)
    logits = model(features).squeeze(-1)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def geometry_key(level):
    # Canonicalize mirror-equivalent maps as well as exact duplicates.
    cells = sorted((b["x"], b["y"], b["id"]) for b in level["bricks"])
    mirror_id = {9: 11, 11: 9}
    mirror = sorted((level["width"] - 1 - x, y, mirror_id.get(kind, kind))
                    for x, y, kind in cells)
    return (level["width"], level["height"], tuple(min(cells, mirror)))


def dataset(html):
    entries = []
    groups = {}
    unique = 0
    for pack, level in load_maps(html):
        name = f"{pack}{level['id']}"
        board = WormMap(level)
        trace, reason = trace_hints(board, level["hints"])
        entry = {"name": name, "width": board.width, "height": board.height,
                 "bricks": [b for b in level["bricks"] if b["id"] not in {2, 4, 5, 6}],
                 "initial": frame(board.initial), "hint_steps": len(level["hints"]),
                 "verified_steps": len(trace), "verification": reason}
        if reason != "win":
            entry["split"] = "unverified"
        else:
            key = geometry_key(level)
            if key in groups:
                entry.update(split="duplicate", alias=groups[key])
            else:
                groups[key] = name
                unique += 1
                entry["split"] = "test" if unique % 5 == 0 else "train"
        entries.append((entry, Arena(board), trace))
    return entries


def evaluate(entries, model, label, include_replay=False, search_nodes=0):
    score = actor(model)
    wins, matching, total, steps, losses, search_wins = 0, 0, 0, 0, 0, 0
    ratios = []
    failures = Counter()
    search_work = {}
    for entry, arena, trace in entries:
        outcome, moves, frames, _ = play(arena, score, 250, log=include_replay)
        wins += outcome == "win"
        losses += outcome != "win"
        if outcome != "win":
            failures[outcome] += 1
        steps += moves
        if outcome == "win":
            ratios.append(moves / len(trace))
        for features, action in examples(arena, trace):
            matching += int(score(features).argmax()) == action
            total += 1
        if include_replay:
            entry.setdefault("attempts", []).append(dict(
                mode=label, outcome=outcome, steps=moves, frames=frames))
        if search_nodes and outcome == "win":
            search_work[entry["name"]] = 0
        elif search_nodes:
            found, expanded = search(arena, score, search_nodes)
            search_wins += bool(found)
            if found:
                search_work[entry["name"]] = expanded
            if found and include_replay and label == "after":
                entry["attempts"].append(dict(mode="search", outcome="win", steps=len(found),
                                              frames=trace_frames(arena, found), expanded=expanded))
    return dict(label=label, maps=len(entries), wins=wins, failed=losses, steps=steps,
                hint_accuracy=round(matching / total, 4),
                avg_win_ratio=round(sum(ratios) / len(ratios), 3) if ratios else None,
                failures=dict(failures),
                wins_with_search=wins + search_wins if search_nodes else None,
                search_work=search_work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=ROOT / "hungry_worm_map_viewer.html")
    parser.add_argument("--out", type=Path, default=ROOT / "worm_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--search-nodes", type=int, default=320)
    parser.add_argument("--eval-search-nodes", type=int, default=4000)
    parser.add_argument("--consolidation", type=int, default=12)
    args = parser.parse_args()
    if args.attempts < 1 or min(args.search_nodes, args.eval_search_nodes, args.consolidation) < 0:
        parser.error("expected at least one attempt and nonnegative search/epoch counts")
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    entries = dataset(args.html)
    train = [row for row in entries if row[0]["split"] == "train"]
    test = [row for row in entries if row[0]["split"] == "test"]
    model = make_policy()
    online = torch.optim.AdamW(model.parameters(), lr=0.001)
    imitation = torch.optim.AdamW(model.parameters(), lr=0.004)
    memory = []
    timeline = [evaluate(test, model, "before", include_replay=True,
                         search_nodes=args.eval_search_nodes)]
    print(f"Validated {sum(e['verification'] == 'win' for e, _, _ in entries)}/{len(entries)}; "
          f"train={len(train)}, test={len(test)}", flush=True)
    counts = Counter()
    for index, (entry, arena, reference) in enumerate(train, 1):
        entry["attempts"] = attempts = []
        penalties = {}
        solution = []
        for attempt in range(args.attempts):
            outcome, steps, frames, experience = play(
                arena, actor(model), 250, None if attempt == 0 else rng, penalties, log=True)
            attempts.append(dict(mode="policy", outcome=outcome, steps=steps, frames=frames))
            learn_play(model, online, experience, penalties, outcome == "win")
            if outcome == "win":
                solution = [(state, ACTIONS[choice], after, event)
                            for state, _, choice, _, after, event in experience]
                break
        method = "policy"
        if not solution:
            found, expanded = search(arena, actor(model), args.search_nodes) if args.search_nodes else ([], 0)
            method = "search" if found else "expert"
            solution = found or reference
            attempts.append(dict(mode=method, outcome="win", steps=len(solution),
                                 frames=trace_frames(arena, solution), expanded=expanded))
        counts[method] += 1
        entry["resolved_by"] = method
        # Keep validated demonstrations in replay memory even when the agent finds its own route.
        new_items = list(examples(arena, reference))
        if method != "expert" and len(solution) < len(reference):
            new_items += list(examples(arena, solution))
        memory.extend(new_items)
        for _ in range(5):
            batch = random.sample(memory, min(len(memory), 160))
            batch.extend(random.sample(new_items, min(len(new_items), 40)))
            learn_batch(model, imitation, batch)
        if index % 35 == 0 or index == len(train):
            baseline = evaluate(test, model, str(index))
            timeline.append(baseline)
            print(f"Map {index}/{len(train)} {entry['name']}, {dict(counts)}. "
                  f"Test: {baseline['wins']}/{len(test)}, hint acc={baseline['hint_accuracy']:.1%}",
                  flush=True)

    for _ in range(args.consolidation):
        random.shuffle(memory)
        for start in range(0, len(memory), 192):
            learn_batch(model, imitation, memory[start:start + 192])
    timeline.append(evaluate(test, model, "after", include_replay=True, search_nodes=args.eval_search_nodes))
    train_metrics = evaluate(train, model, "train_after")
    work_before, work_after = (timeline[i]["search_work"] for i in (0, -1))
    paired = sorted(work_before.keys() & work_after.keys())
    prior_failures = sum(len(e["attempts"]) - 1 for e, _, _ in train)
    first = train[0][0]["attempts"][0]["frames"][1]
    info = dict(seed=args.seed, source_sha256=hashlib.sha256(args.html.read_bytes()).hexdigest(),
                config=dict(attempts=args.attempts, search_nodes=args.search_nodes,
                            eval_search_nodes=args.eval_search_nodes, consolidation=args.consolidation),
                splits=dict(Counter(e["split"] for e, _, _ in entries)),
                verified=sum(e["verification"] == "win" for e, _, _ in entries),
                resolved_by=dict(counts), timeline=timeline, train_final=train_metrics,
                failed_training_attempts=prior_failures, first_action=first["action"],
                first_event=first["event"], paired_search=dict(
                    maps=len(paired), before=sum(work_before[k] for k in paired),
                    after=sum(work_after[k] for k in paired)))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metrics": info}, args.out / "policy.pt")
    (args.out / "metrics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    replay = dict(info=info, maps=[entry for entry, _, _ in entries])
    (args.out / "replay.json").write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "timeline"}, indent=2), flush=True)
    for row in timeline:
        print({k: v for k, v in row.items() if k != "search_work"}, flush=True)


if __name__ == "__main__":
    main()
