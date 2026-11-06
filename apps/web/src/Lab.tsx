import { useEffect, useRef, useState } from "react";
import { api, ApiError, type Agent, type GameState, type Level, type SolveResponse, type Tier } from "./api";
import { Board } from "./Board";
import { ErrorBox, TierPicker } from "./ui";

type Mode = "rl" | "bfs";
interface Replay { res: SolveResponse; frames: GameState[] }

const SPEEDS = [0.5, 1, 2];
const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

// Frames come from replaying the returned actions through a server session,
// so the animation shows engine-authoritative states, never client-side rules.
async function buildFrames(levelId: string, res: SolveResponse): Promise<GameState[]> {
  const s = await api.createSession(levelId);
  const frames = [s.state];
  for (const a of res.actions) {
    const r = await api.act(s.session_id, a);
    frames.push(r.state);
    if (r.state.status !== "PLAYING") break;
  }
  return frames;
}

export function Lab({ initialLevelId }: { initialLevelId?: string }) {
  const [level, setLevel] = useState<Level | null>(null);
  const [tier, setTier] = useState<Tier>("medium");
  const [agents, setAgents] = useState<Agent[] | null>(null);
  const [modelId, setModelId] = useState("");
  const [replays, setReplays] = useState<Partial<Record<Mode, Replay>>>({});
  const [active, setActive] = useState<Mode>("rl");
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [busy, setBusy] = useState<Mode | "level" | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const timer = useRef<number>();

  useEffect(() => {
    api.agents().then((a) => { setAgents(a); if (a[0]) setModelId(a[0].model_id); }).catch((e) => setError(e));
  }, []);

  const load = async (p: Promise<Level>) => {
    setBusy("level");
    setError(null);
    try {
      setLevel(await p);
      setReplays({});
      setFrame(0);
      setPlaying(false);
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setBusy(null);
    }
  };

  useEffect(() => {
    if (initialLevelId) load(api.level(initialLevelId));
  }, [initialLevelId]);

  const run = async (mode: Mode) => {
    if (!level) return;
    setBusy(mode);
    setError(null);
    try {
      const res = await api.solve(level.level_id, mode, mode === "rl" ? modelId : undefined);
      const frames = await buildFrames(level.level_id, res);
      setReplays((r) => ({ ...r, [mode]: { res, frames } }));
      setActive(mode);
      setFrame(0);
      setPlaying(!reducedMotion());
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setBusy(null);
    }
  };

  const cur = replays[active];
  const last = cur ? cur.frames.length - 1 : 0;

  useEffect(() => {
    window.clearTimeout(timer.current);
    if (playing && cur && frame < last) {
      timer.current = window.setTimeout(() => setFrame((f) => f + 1), 320 / speed);
    } else if (frame >= last) {
      setPlaying(false);
    }
    return () => window.clearTimeout(timer.current);
  }, [playing, frame, cur, last, speed]);

  const rl = replays.rl?.res, bfs = replays.bfs?.res;
  const gap = rl?.outcome === "SOLVED" && bfs ? rl.steps - bfs.steps : null;
  const shown: GameState | null = cur ? cur.frames[frame] : level ? {
    worm: level.worm, remaining_apples: level.apples, step_count: 0, max_steps: 4 * level.grid.width ** 2 + 20,
    status: "PLAYING", action_mask: [true, true, true, true],
  } : null;

  return (
    <section className="lab">
      <div className="toolbar">
        <TierPicker value={tier} onChange={setTier} disabled={busy !== null} />
        <button onClick={() => load(api.generate(tier).then((g) => g.level))} disabled={busy !== null}>
          {busy === "level" ? "Loading…" : "Load a level"}
        </button>
      </div>

      {error && <ErrorBox error={error}>
        {error.code === "INFERENCE_TIMEOUT" && <>Model attempt timed out. Retry, or view the BFS solution.</>}
      </ErrorBox>}

      {!level && !busy && !error && <div className="empty"><p>Load a level, then compare the RL policy's own attempt with the BFS verified solution.</p></div>}

      {level && shown && (
        <div className="game">
          <div className="board-wrap">
            <Board level={level} state={shown} />
            {cur && (
              <p className={`replay-label ${active}`}>
                {active === "rl" ? `RL attempt · ${cur.res.model_id}` : "BFS verified shortest path (oracle)"}
              </p>
            )}
          </div>

          <aside className="side">
            <p className="meta"><code>{level.level_id}</code> · {level.difficulty_label} · score {level.difficulty_score.toFixed(2)}</p>

            <div className="agent-card rl">
              <h3>RL policy</h3>
              {agents === null ? <p>Loading models…</p> : agents.length === 0 ? (
                <p className="muted">No trained model registered. Train one with <code>python -m ml.train --seed 11</code>.</p>
              ) : (
                <>
                  <label>Model
                    <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                      {agents.map((a) => <option key={a.model_id} value={a.model_id}>{a.model_id} (seed {a.seed})</option>)}
                    </select>
                  </label>
                  <button className="primary" onClick={() => run("rl")} disabled={busy !== null || !modelId}>
                    {busy === "rl" ? "Running policy…" : "Watch RL"}
                  </button>
                </>
              )}
              {rl && <Outcome r={rl} />}
            </div>

            <div className="agent-card bfs">
              <h3>BFS oracle</h3>
              <button onClick={() => run("bfs")} disabled={busy !== null}>{busy === "bfs" ? "Loading…" : "Show BFS solution"}</button>
              {bfs && <Outcome r={bfs} />}
            </div>

            {gap !== null && <p className="gap">Optimality gap: RL used {gap} more move{gap === 1 ? "" : "s"} than the shortest path.</p>}

            {cur && (
              <div className="transport" role="group" aria-label="Replay controls">
                <div className="row">
                  <button onClick={() => { setFrame(0); setPlaying(false); }} aria-label="Restart replay">⏮</button>
                  <button onClick={() => { if (frame >= last) setFrame(0); setPlaying((p) => !p); }} aria-label={playing ? "Pause" : "Play"}>{playing ? "⏸" : "▶"}</button>
                  <button onClick={() => { setPlaying(false); setFrame((f) => Math.min(f + 1, last)); }} aria-label="Step forward">⏭</button>
                  <span className="seg small" role="radiogroup" aria-label="Speed">
                    {SPEEDS.map((s) => <button key={s} role="radio" aria-checked={speed === s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)}>{s}×</button>)}
                  </span>
                </div>
                <input type="range" min={0} max={last} value={frame} aria-label="Replay position"
                  onChange={(e) => { setPlaying(false); setFrame(+e.target.value); }} />
                <p className="muted">Move {frame} / {last}{frame > 0 && cur.res.actions[frame - 1] ? ` · ${cur.res.actions[frame - 1]}` : ""}</p>
                {replays.rl && replays.bfs && (
                  <div className="seg small" role="radiogroup" aria-label="Replay source">
                    {(["rl", "bfs"] as Mode[]).map((m) => <button key={m} role="radio" aria-checked={active === m} className={active === m ? "on" : ""}
                      onClick={() => { setActive(m); setFrame(0); setPlaying(false); }}>{m === "rl" ? "RL attempt" : "BFS solution"}</button>)}
                  </div>
                )}
              </div>
            )}
          </aside>
        </div>
      )}
    </section>
  );
}

function Outcome({ r }: { r: SolveResponse }) {
  if (r.agent_type === "bfs") return <p className="outcome ok">✔ Verified shortest path · {r.steps} moves</p>;
  if (r.outcome === "SOLVED") return <p className="outcome ok">✔ Solved by this model · {r.steps} moves · {r.elapsed_ms} ms</p>;
  return (
    <p className="outcome fail">
      ✖ RL could not solve this level ({r.outcome === "TIMEOUT" ? "ran out of moves" : r.failure_reason === "DEADLOCK" ? "boxed itself in" : "failed"}) after {r.steps} moves.
      <br /><span className="muted">The level is still solvable — see the BFS solution.</span>
    </p>
  );
}
