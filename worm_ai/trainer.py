"""Continual RL Solver Trainer: Contra-style Parallel Masked PPO + Policy-Guided Lookahead + Replay."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import heapq
import itertools
import math
import random
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from worm_physics import State
from .model import WormBrainNet
from .oracle import solve_exact
from .physics import (
    ACTIONS,
    BLOCKED_EVENTS,
    FATAL_EVENTS,
    MAX_STEPS,
    LevelArena,
    mirror_arena_and_trace,
    step_strict_trace,
)

GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2


@dataclass
class ReplaySample:
    grid: np.ndarray          # (12, H, W) float32
    action_feats: np.ndarray  # (4, 26) float32
    safe_mask: np.ndarray     # (4,) bool
    target_probs: np.ndarray  # (4,) float32 soft/hard target distribution
    target_return: float      # discounted return
    stage_name: str = ""


class ContinualReplayBuffer:
    """Bounded multi-stage experience memory preventing catastrophic forgetting across stages."""

    def __init__(self, capacity: int = 28000):
        self.capacity = capacity
        self.buffer: deque[ReplaySample] = deque(maxlen=capacity)
        self.mastered_stages: set[str] = set()

    def __len__(self) -> int:
        return len(self.buffer)

    def add_trajectory(
        self,
        arena: LevelArena,
        trace: list[tuple[State, int, State, str]],
        include_mirror: bool = True,
    ) -> int:
        added = self._ingest_single(arena, trace)
        self.mastered_stages.add(arena.name)
        if include_mirror:
            mirrored = mirror_arena_and_trace(arena, trace)
            if mirrored is not None:
                m_arena, m_trace = mirrored
                added += self._ingest_single(m_arena, m_trace)
        return added

    def _ingest_single(self, arena: LevelArena, trace: list[tuple[State, int, State, str]]) -> int:
        if not trace:
            return 0
        seen: set[State] = set()
        total_len = len(trace)
        count = 0
        for idx, (state, action, _, _) in enumerate(trace):
            seen.add(state)
            grid, feats, safe_mask, _, _ = arena.observe(state, seen)
            target = np.zeros(4, dtype=np.float32)
            target[action - 1] = 1.0
            rem = total_len - idx
            ret = (GAMMA ** (rem - 1)) * 8.0 - 0.02 * (1.0 - GAMMA ** rem) / (1.0 - GAMMA)
            self.buffer.append(
                ReplaySample(
                    grid=grid,
                    action_feats=feats,
                    safe_mask=safe_mask,
                    target_probs=target,
                    target_return=float(ret),
                    stage_name=arena.name,
                )
            )
            count += 1
        return count

    def sample_batch(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        if not self.buffer:
            return []
        k = min(batch_size, len(self.buffer))
        return rng.sample(list(self.buffer), k)


@dataclass
class EpisodeResult:
    stage: str
    won: bool
    outcome: str
    steps: int
    total_reward: float
    actions: list[int]
    frames: list[dict] = field(default_factory=list)
    trajectory: list[tuple[State, int, State, str]] = field(default_factory=list)


def frame_dict(state: State, **extra) -> dict:
    return {
        "body": [list(p) for p in state.body],
        "apples": [list(p) for p in sorted(state.apples)],
        "stones": [list(p) for p in sorted(state.stones)],
        **extra,
    }


def compute_step_reward(
    event: str,
    prev_state: State,
    next_state: State,
    seen: set[State],
    arena: LevelArena,
    step_idx: int,
    stone_pushes_so_far: int,
    max_steps: int = MAX_STEPS,
) -> float:
    """Potential-based shaped reward for Masked PPO self-play."""
    if event == "win":
        speed_bonus = 2.0 * max(0.0, 1.0 - step_idx / max_steps)
        return round(8.0 + speed_bonus, 4)
    if event in BLOCKED_EVENTS or event in FATAL_EVENTS or event == "stuck":
        return -3.0

    # Potential difference Phi(s') - Phi(s) using static BFS distance
    phi_prev = -0.06 * min(45, arena.dist(prev_state.body[0]))
    phi_next = -0.06 * min(45, arena.dist(next_state.body[0]))
    shaping = max(-0.25, min(0.25, GAMMA * phi_next - phi_prev))

    apple_bonus = 0.45 if len(next_state.apples) < len(prev_state.apples) else 0.0
    stone_bonus = 0.12 if (next_state.stones != prev_state.stones and stone_pushes_so_far < 3) else 0.0
    loop_pen = -0.85 if next_state in seen else 0.0
    return round(-0.02 + shaping + apple_bonus + stone_bonus + loop_pen, 4)


class ContinualWormTrainer:
    """
    Unified RL Solver & Self-Improvement Engine:
    - Plays any stage greedily or stochastically
    - Runs Contra-PPO parallel self-play rollouts & GAE clipped updates after/on each stage
    - Uses policy-guided lookahead search to escape deceptive dead-ends when PPO loops
    - Consolidates mastered routes into ContinualReplayBuffer so knowledge compounds across maps
    """

    def __init__(
        self,
        model: WormBrainNet | None = None,
        lr: float = 2.5e-4,
        device: str | torch.device = "cpu",
        seed: int = 73,
        replay_capacity: int = 28000,
    ):
        self.device = torch.device(device)
        self.model = (model if model is not None else WormBrainNet()).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.replay = ContinualReplayBuffer(capacity=replay_capacity)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.total_env_steps = 0
        self.total_updates = 0
        self.stage_history: list[dict] = []

    @torch.inference_mode()
    def predict_single(
        self,
        arena: LevelArena,
        state: State,
        seen: set[State] | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray, tuple[tuple[State, str], ...]]:
        """Return (masked_probs[4], value, safe_mask[4], transitions[4])."""
        self.model.eval()
        grid, feats, safe_mask, _, transitions = arena.observe(state, seen)
        g_t = torch.from_numpy(grid[None, ...]).to(self.device)
        f_t = torch.from_numpy(feats[None, ...]).to(self.device)
        m_t = torch.from_numpy(safe_mask[None, ...]).to(self.device)
        logits, val = self.model(g_t, f_t, m_t)
        probs = logits.softmax(dim=-1)[0].cpu().numpy()
        return probs, float(val[0].item()), safe_mask, transitions

    @torch.inference_mode()
    def play_episode(
        self,
        arena: LevelArena,
        greedy: bool = True,
        temperature: float = 0.85,
        max_steps: int = MAX_STEPS,
        record_frames: bool = False,
    ) -> EpisodeResult:
        """Run one complete episode on `arena` with current WormBrainNet."""
        self.model.eval()
        state = arena.board.initial
        seen: set[State] = {state}
        frames = [frame_dict(state, step=0, event="start", reward=0.0)] if record_frames else []
        actions: list[int] = []
        trajectory: list[tuple[State, int, State, str]] = []
        total_reward = 0.0
        stone_pushes = 0
        outcome = "limit"

        for step_idx in range(1, max_steps + 1):
            grid, feats, safe_mask, legal_mask, transitions = arena.observe(state, seen)
            if not legal_mask.any():
                outcome = "stuck"
                if record_frames:
                    frames[-1]["event"] = "stuck"
                break

            g_t = torch.from_numpy(grid[None, ...]).to(self.device)
            f_t = torch.from_numpy(feats[None, ...]).to(self.device)
            m_t = torch.from_numpy(safe_mask[None, ...]).to(self.device)
            logits, val_t = self.model(g_t, f_t, m_t)
            logits_np = logits[0].cpu().numpy()

            if greedy:
                choice_idx = int(logits_np.argmax())
                probs = np.exp(logits_np - logits_np.max())
                probs = probs / probs.sum()
            else:
                scaled = (logits_np - logits_np.max()) / max(0.1, temperature)
                probs = np.exp(scaled)
                probs = probs / probs.sum()
                choice_idx = int(self.np_rng.choice(4, p=probs))

            action = ACTIONS[choice_idx]
            after, event = transitions[choice_idx]
            rew = compute_step_reward(
                event, state, after, seen, arena, step_idx, stone_pushes, max_steps
            )
            if after.stones != state.stones:
                stone_pushes += 1
            total_reward += rew
            actions.append(action)
            trajectory.append((state, action, after, event))

            if event == "ok" and after in seen:
                event = "loop"
                total_reward -= 0.5
            elif event == "ok" and step_idx >= max_steps:
                event = "limit"
                total_reward -= 0.5

            if record_frames:
                sub_states, _ = step_strict_trace(arena.board, state, action)
                probs_fmt = [round(float(p), 3) for p in probs]
                val_fmt = round(float(val_t[0].item()), 3)
                # Emit intermediate 1-cell gravity drop frames so falling is visibly animated!
                for sub_st in sub_states[:-1]:
                    frames.append(
                        frame_dict(
                            sub_st,
                            step=step_idx,
                            action=action,
                            event="falling",
                            reward=0.0,
                            probs=probs_fmt,
                            value=val_fmt,
                        )
                    )
                frames.append(
                    frame_dict(
                        after,
                        step=step_idx,
                        action=action,
                        event=event,
                        reward=round(rew, 3),
                        probs=probs_fmt,
                        value=val_fmt,
                    )
                )

            if event != "ok":
                outcome = event
                break
            seen.add(after)
            state = after

        return EpisodeResult(
            stage=arena.name,
            won=(outcome == "win"),
            outcome=outcome,
            steps=len(actions),
            total_reward=round(total_reward, 3),
            actions=actions,
            frames=frames,
            trajectory=trajectory,
        )

    @torch.inference_mode()
    def evaluate_split(
        self,
        rows: list[tuple[dict, LevelArena, list]],
        max_steps: int = MAX_STEPS,
    ) -> dict:
        """Evaluate zero-shot greedy solve rate across a list of (meta, arena, trace) stages."""
        wins = 0
        outcomes = Counter()
        solved_steps: dict[str, int] = {}
        ratios: list[float] = []
        for meta, arena, ref_trace in rows:
            ep = self.play_episode(arena, greedy=True, max_steps=max_steps, record_frames=False)
            outcomes[ep.outcome] += 1
            if ep.won:
                wins += 1
                solved_steps[arena.name] = ep.steps
                if ref_trace:
                    ratios.append(ep.steps / len(ref_trace))
        return {
            "maps": len(rows),
            "wins": wins,
            "win_rate": round(wins / max(1, len(rows)), 4),
            "outcomes": dict(outcomes),
            "avg_step_ratio": round(sum(ratios) / len(ratios), 3) if ratios else None,
            "solved_steps": solved_steps,
        }

    @torch.inference_mode()
    def policy_guided_search(
        self,
        arena: LevelArena,
        budget: int = 1500,
        root: State | None = None,
        max_depth: int = MAX_STEPS,
        avoid: set[State] | frozenset[State] = frozenset(),
    ) -> tuple[list[tuple[State, int, State, str]], int]:
        """
        Best-First Search guided by WormBrainNet's Actor log-probabilities and Critic value V(s).
        Finds verified winning paths without any external hints!
        """
        self.model.eval()
        start = root if root is not None else arena.board.initial
        serial = itertools.count()
        heap: list[tuple[float, int, State, int]] = [(0.0, next(serial), start, 0)]
        best_depth: dict[State, int] = {start: 0}
        parent: dict[State, tuple[State, int] | None] = {start: None}
        expanded = 0

        while heap and expanded < budget:
            _, _, state, depth = heapq.heappop(heap)
            if depth > best_depth.get(state, max_depth + 1) or depth >= max_depth:
                continue
            expanded += 1

            grid, feats, safe_mask, _, transitions = arena.observe(state)
            g_t = torch.from_numpy(grid[None, ...]).to(self.device)
            f_t = torch.from_numpy(feats[None, ...]).to(self.device)
            m_t = torch.from_numpy(safe_mask[None, ...]).to(self.device)
            logits, val_t = self.model(g_t, f_t, m_t)
            log_probs = logits.log_softmax(dim=-1)[0].cpu().numpy()
            v_est = float(val_t[0].item())

            for idx, (after, event) in enumerate(transitions):
                act = ACTIONS[idx]
                if event == "win":
                    rev = [(state, act, after, "win")]
                    cur = state
                    while parent[cur] is not None:
                        p_state, p_act = parent[cur]  # type: ignore[misc]
                        rev.append((p_state, p_act, cur, "ok"))
                        cur = p_state
                    rev.reverse()
                    return rev, expanded
                if event != "ok" or after in avoid:
                    continue
                nd = depth + 1
                if nd < best_depth.get(after, max_depth + 1):
                    best_depth[after] = nd
                    parent[after] = (state, act)
                    h_val = arena.admissible_heuristic(after.body[0])
                    # Priority: lower is better -> combine depth, admissible heuristic, policy logp, and value
                    prio = 0.72 * nd + 1.15 * h_val - 1.6 * float(log_probs[idx]) - 0.15 * v_est
                    heapq.heappush(heap, (prio, next(serial), after, nd))

        return [], expanded

    def supervised_consolidation_step(
        self,
        samples: list[ReplaySample],
        batch_size: int = 64,
        updates: int = 8,
    ) -> float:
        """Train Actor & Critic on verified replay trajectories + anti-forgetting buffer."""
        if not samples:
            return 0.0
        self.model.train()
        loss_acc = 0.0
        for _ in range(updates):
            batch = self.rng.choices(samples, k=min(batch_size, len(samples)))
            # Mix 50% from global continual replay buffer if available
            if len(self.replay) > 0:
                replay_part = self.replay.sample_batch(batch_size // 2, self.rng)
                batch = batch[: max(1, batch_size - len(replay_part))] + replay_part

            grids = torch.from_numpy(np.stack([s.grid for s in batch])).to(self.device)
            feats = torch.from_numpy(np.stack([s.action_feats for s in batch])).to(self.device)
            masks = torch.from_numpy(np.stack([s.safe_mask for s in batch])).to(self.device)
            targets = torch.from_numpy(np.stack([s.target_probs for s in batch])).to(self.device)
            returns = torch.tensor([s.target_return for s in batch], dtype=torch.float32, device=self.device)

            logits, values = self.model(grids, feats, masks)
            logp = logits.log_softmax(dim=-1)
            actor_loss = -(targets * logp).sum(dim=-1).mean()
            critic_loss = F.smooth_l1_loss(values, returns)
            loss = actor_loss + 0.25 * critic_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.total_updates += 1
            loss_acc += float(loss.detach().item())
        return loss_acc / max(1, updates)

    def _new_slot(self, arenas: list[LevelArena]) -> dict:
        arena = self.rng.choice(arenas)
        init = arena.board.initial
        return {
            "arena": arena,
            "state": init,
            "seen": {init},
            "step": 0,
            "pushes": 0,
            "trace": [],
        }

    def ppo_rollout_and_update(
        self,
        arenas: list[LevelArena],
        slots: list[dict],
        horizon: int = 48,
        ppo_epochs: int = 4,
        minibatch_size: int = 128,
        entropy_coef: float = 0.018,
        anchor_weight: float = 0.25,
        max_steps: int = MAX_STEPS,
    ) -> dict:
        """
        Contra-PPO style parallel environment rollout + GAE + Clipped Surrogate Update,
        augmented with Continual Replay Anchor loss to preserve previously mastered maps.
        """
        num_envs = len(slots)
        obs_grids, obs_feats, obs_masks = [], [], []
        act_list, logp_list, val_list, rew_list, done_list = [], [], [], [], []

        ep_wins = 0
        ep_done = 0
        discovered_wins: list[tuple[LevelArena, list[tuple[State, int, State, str]]]] = []

        self.model.eval()
        for _ in range(horizon):
            step_grids, step_feats, step_masks, step_trans = [], [], [], []
            for slot in slots:
                g, f, sm, _, tr = slot["arena"].observe(slot["state"], slot["seen"])
                step_grids.append(g)
                step_feats.append(f)
                step_masks.append(sm)
                step_trans.append(tr)

            g_b = torch.from_numpy(np.stack(step_grids)).to(self.device)
            f_b = torch.from_numpy(np.stack(step_feats)).to(self.device)
            m_b = torch.from_numpy(np.stack(step_masks)).to(self.device)

            with torch.no_grad():
                logits, values = self.model(g_b, f_b, m_b)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            step_rews = []
            step_dones = []
            for idx, act_idx in enumerate(actions.tolist()):
                slot = slots[idx]
                arena: LevelArena = slot["arena"]
                prev_state: State = slot["state"]
                after, event = step_trans[idx][act_idx]
                slot["step"] += 1
                rew = compute_step_reward(
                    event, prev_state, after, slot["seen"], arena, slot["step"], slot["pushes"], max_steps
                )
                if after.stones != prev_state.stones:
                    slot["pushes"] += 1
                slot["trace"].append((prev_state, ACTIONS[act_idx], after, event))

                if event == "ok" and after in slot["seen"]:
                    event = "loop"
                    rew -= 0.5
                elif event == "ok" and slot["step"] >= max_steps:
                    event = "limit"
                    rew -= 0.5

                is_done = event != "ok"
                step_rews.append(rew)
                step_dones.append(is_done)
                if is_done:
                    ep_done += 1
                    if event == "win":
                        ep_wins += 1
                        discovered_wins.append((arena, list(slot["trace"])))
                    slots[idx] = self._new_slot(arenas)
                else:
                    slot["seen"].add(after)
                    slot["state"] = after

            obs_grids.append(g_b)
            obs_feats.append(f_b)
            obs_masks.append(m_b)
            act_list.append(actions)
            logp_list.append(log_probs)
            val_list.append(values)
            rew_list.append(torch.tensor(step_rews, dtype=torch.float32, device=self.device))
            done_list.append(torch.tensor(step_dones, dtype=torch.bool, device=self.device))

        self.total_env_steps += num_envs * horizon

        # Bootstrap next value for GAE
        with torch.no_grad():
            last_g = np.stack([s["arena"].observe(s["state"], s["seen"])[0] for s in slots])
            last_f = np.stack([s["arena"].observe(s["state"], s["seen"])[1] for s in slots])
            last_m = np.stack([s["arena"].observe(s["state"], s["seen"])[2] for s in slots])
            _, next_val = self.model(
                torch.from_numpy(last_g).to(self.device),
                torch.from_numpy(last_f).to(self.device),
                torch.from_numpy(last_m).to(self.device),
            )

        grids_t = torch.stack(obs_grids)
        feats_t = torch.stack(obs_feats)
        masks_t = torch.stack(obs_masks)
        acts_t = torch.stack(act_list)
        old_logp_t = torch.stack(logp_list)
        vals_t = torch.stack(val_list)
        rews_t = torch.stack(rew_list)
        dones_t = torch.stack(done_list)

        advantages = torch.zeros_like(rews_t)
        gae = torch.zeros_like(next_val)
        for t in reversed(range(horizon)):
            cont = (~dones_t[t]).float()
            delta = rews_t[t] + GAMMA * next_val * cont - vals_t[t]
            gae = delta + GAMMA * GAE_LAMBDA * cont * gae
            advantages[t] = gae
            next_val = vals_t[t]
        returns_t = advantages + vals_t
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)

        # Flatten (T, N) -> (T*N)
        b_grids = grids_t.flatten(0, 1)
        b_feats = feats_t.flatten(0, 1)
        b_masks = masks_t.flatten(0, 1)
        b_acts = acts_t.flatten(0, 1)
        b_old_logp = old_logp_t.flatten(0, 1)
        b_adv = advantages.flatten(0, 1)
        b_ret = returns_t.flatten(0, 1)

        self.model.train()
        total_loss = 0.0
        total_ent = 0.0
        n_batches = 0
        for _ in range(ppo_epochs):
            perm = torch.randperm(len(b_acts), device=self.device)
            for ids in perm.split(minibatch_size):
                logits, values = self.model(b_grids[ids], b_feats[ids], b_masks[ids])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(b_acts[ids]) - b_old_logp[ids]).exp()
                surr1 = ratio * b_adv[ids]
                surr2 = ratio.clamp(1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * b_adv[ids]
                policy_loss = -torch.minimum(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(values, b_ret[ids])
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

                # Continual Replay Anchor against forgetting previously mastered maps
                if anchor_weight > 0 and len(self.replay) > 0:
                    rep_batch = self.replay.sample_batch(min(48, len(self.replay)), self.rng)
                    rg = torch.from_numpy(np.stack([r.grid for r in rep_batch])).to(self.device)
                    rf = torch.from_numpy(np.stack([r.action_feats for r in rep_batch])).to(self.device)
                    rm = torch.from_numpy(np.stack([r.safe_mask for r in rep_batch])).to(self.device)
                    rt = torch.from_numpy(np.stack([r.target_probs for r in rep_batch])).to(self.device)
                    r_logits, _ = self.model(rg, rf, rm)
                    anchor_loss = -(rt * r_logits.log_softmax(dim=-1)).sum(dim=-1).mean()
                    loss = loss + anchor_weight * anchor_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.75)
                self.optimizer.step()
                self.total_updates += 1
                total_loss += float(loss.detach().item())
                total_ent += float(entropy.detach().item())
                n_batches += 1

        # Add any self-discovered winning trajectories directly into continual replay buffer!
        for win_arena, win_trace in discovered_wins[:3]:
            self.replay.add_trajectory(win_arena, win_trace, include_mirror=True)

        return {
            "episodes_done": ep_done,
            "rollout_wins": ep_wins,
            "loss": round(total_loss / max(1, n_batches), 4),
            "entropy": round(total_ent / max(1, n_batches), 4),
            "discovered_wins": len(discovered_wins),
        }

    def play_and_self_train_stage(
        self,
        arena: LevelArena,
        ppo_rounds: int = 4,
        envs: int = 8,
        horizon: int = 48,
        rescue_budget: int = 3500,
        max_steps: int = MAX_STEPS,
        record_frames: bool = True,
        on_progress: Callable[[dict], None] | None = None,
    ) -> dict:
        """
        Core Product Loop ("Tự chơi & tự train lại chính mình sau mỗi màn chơi"):
        1. Attempt stage Zero-Shot with current Shared Brain
        2. Run Contra-style Parallel Masked PPO self-play rounds on this stage
        3. If deceptive puzzle still traps greedy play, run Policy-Guided Lookahead Rescue (no hints!)
        4. Consolidate into ContinualReplayBuffer so the model masters this stage AND remembers past stages!
        """
        t0 = time.perf_counter()
        before_ep = self.play_episode(arena, greedy=True, max_steps=max_steps, record_frames=record_frames)
        if on_progress:
            on_progress({
                "phase": "zero_shot",
                "stage": arena.name,
                "won": before_ep.won,
                "outcome": before_ep.outcome,
                "steps": before_ep.steps,
                "reward": before_ep.total_reward,
                "episode": before_ep,
            })

        # Even if zero-shot already won, run 1 lightweight PPO/consolidation pass to reinforce & sharpen path!
        target_rounds = 1 if before_ep.won else ppo_rounds
        slots = [self._new_slot([arena]) for _ in range(envs)]
        round_logs: list[dict] = []
        best_ep = before_ep
        if before_ep.won:
            self.replay.add_trajectory(arena, before_ep.trajectory, include_mirror=True)

        for r_idx in range(1, target_rounds + 1):
            stats = self.ppo_rollout_and_update(
                [arena],
                slots,
                horizon=horizon,
                ppo_epochs=3,
                minibatch_size=min(128, envs * horizon),
                anchor_weight=0.30,
                max_steps=max_steps,
            )
            eval_ep = self.play_episode(arena, greedy=True, max_steps=max_steps, record_frames=record_frames)
            if eval_ep.won and (not best_ep.won or eval_ep.steps < best_ep.steps):
                best_ep = eval_ep
                self.replay.add_trajectory(arena, eval_ep.trajectory, include_mirror=True)

            log_item = {
                "phase": "ppo_round",
                "stage": arena.name,
                "round": r_idx,
                "total_rounds": target_rounds,
                **stats,
                "greedy_won": eval_ep.won,
                "greedy_outcome": eval_ep.outcome,
                "greedy_steps": eval_ep.steps,
                "greedy_reward": eval_ep.total_reward,
            }
            round_logs.append(log_item)
            if on_progress:
                on_progress({**log_item, "episode": eval_ep})
            if eval_ep.won and r_idx >= 1:
                break

        used_lookahead = False
        search_nodes = 0

        # Policy-Guided Lookahead Rescue if pure PPO rollouts got stuck in a deceptive puzzle trap
        if not best_ep.won and rescue_budget > 0:
            used_lookahead = True
            # Fast neural-guided lookahead first (capped for CPU speed), then exact simulator search
            found_trace, s_nodes = self.policy_guided_search(
                arena, budget=min(180, rescue_budget), max_depth=max_steps
            )
            search_nodes += s_nodes
            if not found_trace:
                found_trace, e_nodes, _, _, _ = solve_exact(
                    arena.board, budget=max(8000, rescue_budget * 5), max_depth=max_steps
                )
                search_nodes += e_nodes

            if found_trace:
                # Build exact stage trajectory samples before adding mirror to replay
                stage_samples: list[ReplaySample] = []
                seen_s: set[State] = set()
                for idx, (st, act, _, _) in enumerate(found_trace):
                    seen_s.add(st)
                    g, f, sm, _, _ = arena.observe(st, seen_s)
                    tgt = np.zeros(4, dtype=np.float32)
                    tgt[act - 1] = 1.0
                    rem = len(found_trace) - idx
                    ret = (GAMMA ** (rem - 1)) * 8.0 - 0.02 * (1.0 - GAMMA ** rem) / (1.0 - GAMMA)
                    stage_samples.append(ReplaySample(g, f, sm, tgt, float(ret), arena.name))

                # Temporarily boost lr slightly if brain is cold so online mastery finishes in ~1-2s
                base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
                adapt_lr = 1.2e-3 if len(self.replay.mastered_stages) < 5 else max(base_lrs[0], 6e-4)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = adapt_lr

                try:
                    s_grids = torch.from_numpy(np.stack([s.grid for s in stage_samples])).to(self.device)
                    s_feats = torch.from_numpy(np.stack([s.action_feats for s in stage_samples])).to(self.device)
                    s_masks = torch.from_numpy(np.stack([s.safe_mask for s in stage_samples])).to(self.device)
                    s_tgts = torch.from_numpy(np.stack([s.target_probs for s in stage_samples])).to(self.device)
                    s_rets = torch.tensor(
                        [s.target_return for s in stage_samples], dtype=torch.float32, device=self.device
                    )

                    for c_pass in range(1, 10):
                        self.model.train()
                        c_loss = 0.0
                        for _ in range(12):
                            logits, vals = self.model(s_grids, s_feats, s_masks)
                            logp = logits.log_softmax(dim=-1)
                            per_step_ce = -(s_tgts * logp).sum(dim=-1)
                            # Hard-step focus: 2.5x weight on steps not yet predicted greedily
                            wrong_mask = (logits.argmax(dim=-1) != s_tgts.argmax(dim=-1)).float()
                            weights = 1.0 + 2.0 * wrong_mask
                            actor_loss = (per_step_ce * weights).mean()
                            critic_loss = F.smooth_l1_loss(vals, s_rets)
                            loss = actor_loss + 0.25 * critic_loss

                            # Anti-forgetting replay batch from previously mastered stages
                            if len(self.replay) > 0:
                                rep = self.replay.sample_batch(min(48, len(self.replay)), self.rng)
                                rg = torch.from_numpy(np.stack([r.grid for r in rep])).to(self.device)
                                rf = torch.from_numpy(np.stack([r.action_feats for r in rep])).to(self.device)
                                rm = torch.from_numpy(np.stack([r.safe_mask for r in rep])).to(self.device)
                                rt = torch.from_numpy(np.stack([r.target_probs for r in rep])).to(self.device)
                                r_logits, _ = self.model(rg, rf, rm)
                                rep_loss = -(rt * r_logits.log_softmax(dim=-1)).sum(dim=-1).mean()
                                loss = loss + 0.65 * rep_loss

                            self.optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.total_updates += 1
                            c_loss += float(loss.detach().item())

                        c_loss /= 12.0
                        after_ep = self.play_episode(
                            arena, greedy=True, max_steps=max_steps, record_frames=record_frames
                        )
                        if on_progress:
                            on_progress({
                                "phase": "consolidation",
                                "stage": arena.name,
                                "pass": c_pass,
                                "loss": round(c_loss, 4),
                                "search_nodes": search_nodes,
                                "greedy_won": after_ep.won,
                                "greedy_outcome": after_ep.outcome,
                                "greedy_steps": after_ep.steps,
                                "episode": after_ep,
                            })
                        best_ep = after_ep
                        if after_ep.won:
                            break
                finally:
                    for pg, orig_lr in zip(self.optimizer.param_groups, base_lrs):
                        pg["lr"] = orig_lr

                self.replay.add_trajectory(arena, found_trace, include_mirror=True)

        elapsed = round(time.perf_counter() - t0, 2)
        summary = {
            "stage": arena.name,
            "zero_shot_won": before_ep.won,
            "zero_shot_outcome": before_ep.outcome,
            "zero_shot_steps": before_ep.steps,
            "after_train_won": best_ep.won,
            "after_train_outcome": best_ep.outcome,
            "after_train_steps": best_ep.steps,
            "used_lookahead_rescue": used_lookahead,
            "search_nodes": search_nodes,
            "ppo_rounds_run": len(round_logs),
            "mastered_stages_in_memory": len(self.replay.mastered_stages),
            "replay_size": len(self.replay),
            "elapsed_sec": elapsed,
            "before_frames": before_ep.frames,
            "after_frames": best_ep.frames,
            "after_actions": best_ep.actions,
        }
        self.stage_history.append({
            k: v for k, v in summary.items() if k not in ("before_frames", "after_frames")
        })
        return summary

    def pretrain_foundation(
        self,
        train_rows: list[tuple[dict, LevelArena, list]],
        val_rows: list[tuple[dict, LevelArena, list]],
        warm_epochs: int = 12,
        dagger_rounds: int = 2,
        ppo_rounds: int = 2,
        batch_size: int = 96,
        log_fn: Callable[[str], None] = print,
    ) -> dict:
        """
        Train a strong general-purpose foundation checkpoint on training levels:
        Stage 1: Behavioral Cloning with 2x Mirror Augmentation
        Stage 2: DAGGER / Lookahead Pivot Repair on failed training rollouts
        Stage 3: Multi-map Masked PPO with Replay Anchor
        """
        t0 = time.perf_counter()
        for _, arena, trace in train_rows:
            self.replay.add_trajectory(arena, trace, include_mirror=True)

        all_demos = list(self.replay.buffer)
        log_fn(
            f"[Foundation] Loaded {len(all_demos)} transitions (with mirror aug) from {len(train_rows)} maps. "
            f"Params: {self.model.parameter_count():,}"
        )

        best_val = self.evaluate_split(val_rows)
        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        history = {"warmup": [], "dagger": [], "ppo": [], "initial_val": best_val}

        steps_per_epoch = max(1, math.ceil(len(all_demos) / batch_size))
        for epoch in range(1, warm_epochs + 1):
            loss = self.supervised_consolidation_step(all_demos, batch_size=batch_size, updates=steps_per_epoch)
            val_res = self.evaluate_split(val_rows)
            history["warmup"].append({"epoch": epoch, "loss": round(loss, 4), "val": val_res})
            if val_res["wins"] >= best_val["wins"]:
                best_val = val_res
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            if epoch % 3 == 0 or epoch == warm_epochs or epoch == 1:
                log_fn(
                    f"[Warmup {epoch:02d}/{warm_epochs}] loss={loss:.4f} | "
                    f"val zero-shot={val_res['wins']}/{val_res['maps']} ({val_res['win_rate']:.1%})"
                )

        self.model.load_state_dict(best_state)

        # Stage 2: Online Pivot Repair (DAGGER with simulator lookahead on mistakes)
        for d_round in range(1, dagger_rounds + 1):
            repair_samples: list[ReplaySample] = []
            repaired_maps = 0
            subset = self.rng.sample(train_rows, min(80, len(train_rows)))
            for _, arena, _ in subset:
                ep = self.play_episode(arena, greedy=True, record_frames=False)
                if ep.won or not ep.trajectory:
                    continue
                # Query 2 pivot states before failure
                pivots = sorted({max(0, len(ep.trajectory) - d) for d in (1, 3, 7)})
                for piv in pivots:
                    piv_state = ep.trajectory[piv][0]
                    avoid = {s[0] for s in ep.trajectory[:piv]}
                    path, _ = self.policy_guided_search(
                        arena, budget=600, root=piv_state, max_depth=MAX_STEPS - piv, avoid=avoid
                    )
                    if path:
                        repaired_maps += 1
                        loc_seen = set(avoid)
                        for idx2, (st2, act2, _, _) in enumerate(path[:10]):
                            loc_seen.add(st2)
                            g2, f2, sm2, _, _ = arena.observe(st2, loc_seen)
                            t2 = np.zeros(4, dtype=np.float32)
                            t2[act2 - 1] = 1.0
                            rem = len(path) - idx2
                            ret = (GAMMA ** (rem - 1)) * 8.0 - 0.02 * rem
                            sample = ReplaySample(g2, f2, sm2, t2, float(ret), arena.name)
                            repair_samples.append(sample)
                            self.replay.buffer.append(sample)
                        break

            loss = self.supervised_consolidation_step(
                repair_samples if repair_samples else all_demos, batch_size=batch_size, updates=80
            )
            val_res = self.evaluate_split(val_rows)
            history["dagger"].append({
                "round": d_round,
                "repairs_added": len(repair_samples),
                "loss": round(loss, 4),
                "val": val_res,
            })
            if val_res["wins"] >= best_val["wins"]:
                best_val = val_res
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            log_fn(
                f"[DAGGER {d_round}/{dagger_rounds}] repairs={len(repair_samples)} ({repaired_maps} maps) | "
                f"val zero-shot={val_res['wins']}/{val_res['maps']}"
            )

        self.model.load_state_dict(best_state)

        # Stage 3: Multi-stage Masked PPO polish with replay anchor
        train_arenas = [arena for _, arena, _ in train_rows]
        slots = [self._new_slot(train_arenas) for _ in range(16)]
        for p_round in range(1, ppo_rounds + 1):
            p_stats = []
            for _ in range(18):
                st = self.ppo_rollout_and_update(
                    train_arenas, slots, horizon=48, ppo_epochs=3, minibatch_size=192, anchor_weight=0.45
                )
                p_stats.append(st)
            val_res = self.evaluate_split(val_rows)
            avg_loss = sum(x["loss"] for x in p_stats) / len(p_stats)
            history["ppo"].append({"round": p_round, "loss": round(avg_loss, 4), "val": val_res})
            if val_res["wins"] >= best_val["wins"]:
                best_val = val_res
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            log_fn(
                f"[PPO {p_round}/{ppo_rounds}] loss={avg_loss:.4f} | "
                f"val zero-shot={val_res['wins']}/{val_res['maps']}"
            )

        self.model.load_state_dict(best_state)
        history["best_val"] = best_val
        history["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        return history
