"""Deterministic Hungry Worm simulator and validation against the HTML hints."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple


DIRECTIONS = {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}
HAZARDS = {3, 7, 9, 10, 11}


def load_maps(path):
    html = Path(path).read_text(encoding="utf-8-sig")
    marker = re.search(r"\bconst\s+DATA\s*=\s*", html)
    if marker is None:
        raise ValueError("HTML is missing the embedded DATA object")
    data, _ = json.JSONDecoder().raw_decode(html, marker.end())
    return [(pack, level) for pack in ("A", "B") for level in data["levels" + pack]]


class State(NamedTuple):
    body: tuple
    apples: frozenset
    stones: frozenset


class Rules(NamedTuple):
    apple_support: bool = True
    portal_support: bool = True
    hazard_support: bool = False
    worm_supports_stone: bool = True
    push_vertical: bool = True
    portal_needs_apples: bool = False
    stone_on_hazard: bool = True
    chain_push: bool = True
    stone_stops_on_hazard: bool = True
    destroy_stone_on_hazard: bool = False


class WormMap:
    def __init__(self, level, rules=Rules()):
        self.width, self.height = level["width"], level["height"]
        bricks = {(b["x"], b["y"]): b["id"] for b in level["bricks"]}
        self.walls = frozenset(p for p, kind in bricks.items() if kind == 1)
        self.hazards = frozenset(p for p, kind in bricks.items() if kind in HAZARDS)
        self.portals = frozenset(p for p, kind in bricks.items() if kind == 8)
        head, = (p for p, kind in bricks.items() if kind == 5)
        tail = {p for p, kind in bricks.items() if kind == 6}

        def paths(body, remaining):
            if not remaining:
                yield body
                return
            x, y = body[-1]
            for dx, dy in DIRECTIONS.values():
                p = (x + dx, y + dy)
                if p in remaining:
                    yield from paths(body + (p,), remaining - {p})

        self.bodies = tuple(paths((head,), tail))
        if not self.bodies:
            raise ValueError("Disconnected starting worm")
        self.initial = State(
            self.bodies[0],
            frozenset(p for p, kind in bricks.items() if kind == 4),
            frozenset(p for p, kind in bricks.items() if kind == 2),
        )
        self.rules = rules
        self.fixed = self.walls | self.portals
        if not rules.stone_on_hazard:
            self.fixed |= self.hazards
        self.support = self.walls
        if rules.portal_support:
            self.support |= self.portals
        if rules.hazard_support:
            self.support |= self.hazards

    def step(self, state, action):
        body, apples, stones = state
        dx, dy = DIRECTIONS[action]
        x, y = body[0]
        target = (x + dx, y + dy)
        if target in self.walls:
            return state, "wall"
        if target in body[1:-1] or target in body[1:2]:
            return state, "body"
        if target in stones:
            push = (target[0] + dx, target[1] + dy)
            while self.rules.chain_push and push in stones:
                push = (push[0] + dx, push[1] + dy)
            if (dy and not self.rules.push_vertical) or push in (
                self.fixed | stones | apples | set(body)
            ):
                return state, "stone"
            stones = stones - {target}
            if not (self.rules.destroy_stone_on_hazard and push in self.hazards):
                stones |= {push}
        if target in apples:
            apples = apples - {target}
            body = (target,) + body
        else:
            body = (target,) + body[:-1]
        return self.settle(State(body, apples, stones))

    def settle(self, state):
        frames, event = self.settle_trace(state)
        return frames[-1], event

    def settle_trace(self, state):
        body, apples, stones = state
        support = self.support | apples if self.rules.apple_support else self.support
        frames = []
        while True:
            state = State(body, apples, stones)
            frames.append(state)
            if any(p in self.hazards for p in body):
                return frames, "trap"
            if body[0] in self.portals and (not self.rules.portal_needs_apples or not apples):
                return frames, "win"
            if max(y for _, y in body) < 0:
                return frames, "fall"
            below = tuple((x, y - 1) for x, y in body)
            worm_moves = all(p not in support and p not in stones for p in below)
            next_stones = set(stones)
            for x, y in sorted(stones, key=lambda p: p[1]):
                dest = (x, y - 1)
                if dest in self.fixed or dest in apples or dest in next_stones:
                    continue
                if self.rules.stone_stops_on_hazard and dest in self.hazards:
                    continue
                if dest in body:
                    if self.rules.worm_supports_stone:
                        continue
                    frames.append(State(body, apples, frozenset(next_stones)))
                    return frames, "crushed"
                next_stones.remove((x, y))
                if y > 0:
                    next_stones.add(dest)
            if not worm_moves and next_stones == stones:
                return frames, "ok"
            stones = frozenset(next_stones)
            if worm_moves:
                body = below

    def step_trace(self, state, action):
        body, apples, stones = state
        dx, dy = DIRECTIONS[action]
        x, y = body[0]
        target = (x + dx, y + dy)
        if target in self.walls:
            return [state], "wall"
        if target in body[1:-1] or target in body[1:2]:
            return [state], "body"
        if target in stones:
            push = (target[0] + dx, target[1] + dy)
            while self.rules.chain_push and push in stones:
                push = (push[0] + dx, push[1] + dy)
            if (dy and not self.rules.push_vertical) or push in (
                self.fixed | stones | apples | set(body)
            ):
                return [state], "stone"
            stones = stones - {target}
            if not (self.rules.destroy_stone_on_hazard and push in self.hazards):
                stones |= {push}
        if target in apples:
            apples = apples - {target}
            body = (target,) + body
        else:
            body = (target,) + body[:-1]
        return self.settle_trace(State(body, apples, stones))


def trace_hints(board, hints):
    best = ([], "empty")
    for body in board.bodies:
        state = board.initial._replace(body=body)
        trace = []
        reason = "empty"
        for index, hint in enumerate(hints):
            action = hint["d"]
            state_after, reason = board.step(state, action)
            expected = (hint["x"], hint["y"])
            dx, dy = DIRECTIONS[action]
            x, y = state.body[0]
            if expected not in ((x + dx, y + dy), state_after.body[0]):
                reason = f"head:{reason}:{state_after.body[0]}!={expected}"
                break
            if reason not in {"win", "ok"}:
                break
            trace.append((state, action, state_after, reason))
            state = state_after
            if reason == "win":
                if index == len(hints) - 1:
                    board.initial = board.initial._replace(body=body)
                    return trace, "win"
                reason = "early_win"
                break
        if len(trace) >= len(best[0]):
            best = (trace, reason if reason != "ok" else "no_win")
    return best


if __name__ == "__main__":
    levels = load_maps(Path(__file__).with_name("hungry_worm_map_viewer.html"))
    results = Counter()
    for pack, level in levels:
        trace, reason = trace_hints(WormMap(level), level["hints"])
        results["win" if reason == "win" else reason.split(":")[0]] += 1
        if reason != "win" and level["id"] <= 16:
            print(pack, level["id"], len(trace), reason)
    print(results)
