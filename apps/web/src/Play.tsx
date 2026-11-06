import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type Action, type GameState, type Level, type Tier } from "./api";
import { Board } from "./Board";
import { DPad, ErrorBox, TierPicker } from "./ui";

const KEYS: Record<string, Action> = {
  ArrowUp: "UP", ArrowRight: "RIGHT", ArrowDown: "DOWN", ArrowLeft: "LEFT",
  w: "UP", d: "RIGHT", s: "DOWN", a: "LEFT", W: "UP", D: "RIGHT", S: "DOWN", A: "LEFT",
};

export function Play({ onWatch }: { onWatch: (levelId: string) => void }) {
  const [tier, setTier] = useState<Tier>("easy");
  const [level, setLevel] = useState<Level | null>(null);
  const [optimal, setOptimal] = useState<number | null>(null);
  const [sid, setSid] = useState<string | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [blocked, setBlocked] = useState(0);
  const [note, setNote] = useState("");
  const seen = useRef<string[]>([]);
  const queue = useRef<Promise<void>>(Promise.resolve());
  const statusRef = useRef<GameState["status"] | null>(null);
  useEffect(() => { statusRef.current = state?.status ?? null; }, [state]);

  const newLevel = useCallback(async (t: Tier) => {
    setLoading(true);
    setError(null);
    try {
      const g = await api.generate(t, seen.current);
      seen.current.push(g.level.level_id);
      const s = await api.createSession(g.level.level_id);
      setLevel(g.level);
      setOptimal(g.certification.optimal_steps);
      setSid(s.session_id);
      setState(s.state);
      setNote("");
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setLoading(false);
    }
  }, []);

  // Moves are chained so rapid key presses are sent in order, each after the
  // previous authoritative response, instead of being dropped.
  const move = useCallback((a: Action) => {
    if (!sid) return;
    queue.current = queue.current.then(() => send(sid, a));
  }, [sid]);

  const send = async (sessionId: string, a: Action) => {
    if (statusRef.current !== "PLAYING") return;
    try {
      const r = await api.act(sessionId, a);
      statusRef.current = r.state.status;
      setState(r.state);
      if (!r.event.valid_action) {
        setBlocked((b) => b + 1);
        setNote(`Move blocked (${a.toLowerCase()})`);
      } else {
        setNote(r.event.ate_apple ? "Apple eaten — the worm grew by one" : "");
      }
    } catch (e) {
      const err = e as ApiError;
      setError(err);
      statusRef.current = null;
      if (err.code === "SESSION_EXPIRED") { setSid(null); setState(null); }
    }
  };

  const restart = async () => {
    if (!sid) return;
    try {
      const r = await api.restart(sid);
      setState(r.state);
      setNote("");
      setError(null);
    } catch (e) {
      const err = e as ApiError;
      setError(err);
      if (err.code === "SESSION_EXPIRED") { setSid(null); setState(null); }
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName)) return;
      const a = KEYS[e.key];
      if (a && statusRef.current === "PLAYING") {
        e.preventDefault();
        move(a);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [move]);

  const ended = state && state.status !== "PLAYING";

  return (
    <section className="play">
      <div className="toolbar">
        <TierPicker value={tier} onChange={setTier} disabled={loading} />
        <button className="primary" onClick={() => newLevel(tier)} disabled={loading} aria-busy={loading}>
          {loading ? "Finding a certified level…" : "Generate verified level"}
        </button>
      </div>

      {error && (
        <ErrorBox error={error}>
          {error.code === "LEVEL_POOL_EMPTY" && <>No verified level available at this tier yet. Pick another tier or try again.</>}
          {error.code === "SESSION_EXPIRED" && <>Session expired. <button className="link" onClick={() => newLevel(tier)}>Start again</button></>}
        </ErrorBox>
      )}

      {!level && !loading && !error && (
        <div className="empty">
          <p>Pick a difficulty and generate a level. Every level served has a solution that the BFS solver found and the engine replayed to a win.</p>
        </div>
      )}

      {level && state && (
        <div className="game">
          <div className="board-wrap">
            <Board level={level} state={state} shake={note.startsWith("Move blocked") ? !!(blocked % 2) : undefined} key={blocked % 2} />
            {ended && (
              <div className="overlay" role="alertdialog" aria-live="assertive">
                {state.status === "WON" ? (
                  <>
                    <h2>Solved in {state.step_count} moves</h2>
                    {optimal !== null && <p>Shortest known solution: {optimal} moves{state.step_count === optimal ? " — you matched it." : "."}</p>}
                    <div className="row">
                      <button className="primary" onClick={() => newLevel(tier)} autoFocus>Try next level</button>
                      <button onClick={restart}>Restart</button>
                    </div>
                  </>
                ) : (
                  <>
                    <h2>Attempt ended</h2>
                    <p>{state.status === "LOST" ? "The worm boxed itself in — no legal moves left." : "Out of moves for this level."}</p>
                    <div className="row">
                      <button className="primary" onClick={restart} autoFocus>Restart</button>
                      <button onClick={() => onWatch(level.level_id)}>Watch solution</button>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          <aside className="side">
            <dl className="stats">
              <div><dt>Level</dt><dd><code>{level.level_id}</code></dd></div>
              <div><dt>Tier</dt><dd className={`tier-tag ${level.difficulty_label}`}>{level.difficulty_label} · {level.difficulty_score.toFixed(2)}</dd></div>
              <div><dt>Apples left</dt><dd>{state.remaining_apples.length}</dd></div>
              <div><dt>Moves</dt><dd>{state.step_count} / {state.max_steps}</dd></div>
            </dl>
            <p className="badge verified" title="A solver-found path was replayed through the engine to a win.">✔ Verified solvable</p>
            <p className="note" aria-live="polite">{note || " "}</p>
            <DPad onMove={move} mask={state.action_mask} disabled={!!ended} />
            <p className="hint">Arrow keys or WASD. Eat every apple, then reach the exit.</p>
            <div className="row">
              <button onClick={restart} disabled={!sid}>Restart</button>
              <button onClick={() => onWatch(level.level_id)}>Watch AI on this level</button>
            </div>
          </aside>
        </div>
      )}
    </section>
  );
}
