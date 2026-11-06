import { useEffect, useState } from "react";
import { api, ApiError, type Count, type TrainingRuns } from "./api";
import { ErrorBox, TIERS } from "./ui";

// Series order is fixed by agent identity, never by rank (categorical slots 1-4).
const SERIES = [
  { key: "rl", label: "RL (mean of seeds)" },
  { key: "random", label: "Random valid" },
  { key: "greedy", label: "Greedy Manhattan" },
  { key: "bfs", label: "BFS oracle" },
] as const;

const pct = (c: Count | null | undefined) => (c && c.total ? c.successes / c.total : null);
const fmt = (v: number | null) => (v === null ? "N/A" : `${(v * 100).toFixed(1)}%`);

function std(xs: number[]) {
  if (xs.length < 2) return null;
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
}

export function Experiments() {
  const [data, setData] = useState<TrainingRuns | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [dataset, setDataset] = useState<string>("");
  const [tip, setTip] = useState<string | null>(null);

  useEffect(() => {
    api.trainingRuns().then((d) => {
      setData(d);
      const hashes = [...new Set([...d.runs, ...d.baselines].map((r) => r.dataset_manifest_hash))];
      if (hashes[0]) setDataset(hashes[0]);
    }).catch((e) => setError(e));
  }, []);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <p className="muted">Loading evaluations…</p>;

  const runs = data.runs.filter((r) => r.dataset_manifest_hash === dataset);
  const baselines = data.baselines.filter((b) => b.dataset_manifest_hash === dataset);
  const hashes = [...new Set([...data.runs, ...data.baselines].map((r) => r.dataset_manifest_hash))];

  if (!runs.length && !baselines.length) {
    return (
      <section className="experiments">
        <div className="empty">
          <p><strong>No completed evaluation yet.</strong></p>
          <p>Run <code>python -m ml.baselines</code> and <code>python -m ml.evaluate --model-id &lt;id&gt;</code> on the locked test manifest. Numbers appear here only from real evaluation artifacts.</p>
        </div>
      </section>
    );
  }

  const value = (series: string, tier: string): { v: number | null; label: string } => {
    if (series === "rl") {
      const xs = runs.map((r) => pct(r.metrics[tier as "easy"])).filter((x): x is number => x !== null);
      if (!xs.length) return { v: null, label: "N/A" };
      const m = xs.reduce((a, b) => a + b, 0) / xs.length;
      const s = std(xs);
      return { v: m, label: `${fmt(m)}${s !== null ? ` ± ${(s * 100).toFixed(1)}` : ""} · ${xs.length} seed${xs.length > 1 ? "s" : ""}` };
    }
    const b = baselines.find((x) => x.agent === series);
    const c = b?.metrics[tier as "easy"];
    return { v: pct(c), label: c ? `${c.successes}/${c.total}` : "N/A" };
  };

  const ds = runs[0]?.dataset ?? baselines[0]?.dataset;

  return (
    <section className="experiments">
      <div className="toolbar">
        <label>Dataset
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {hashes.map((h) => <option key={h} value={h}>{h.slice(0, 12)}</option>)}
          </select>
        </label>
        <span className="muted">{ds} · held-out test split · {runs.length} RL run{runs.length === 1 ? "" : "s"}</span>
      </div>

      <figure className="chart" aria-label="Success rate by tier and agent">
        <figcaption>
          <h3>Success rate on held-out test levels</h3>
          <ul className="chart-legend">
            {SERIES.map((s, i) => <li key={s.key}><i className={`sw s${i + 1}`} />{s.label}</li>)}
          </ul>
        </figcaption>
        <div className="bars">
          {TIERS.map((t) => (
            <div className="bar-group" key={t}>
              <div className="group-label">{t[0].toUpperCase() + t.slice(1)}</div>
              {SERIES.map((s, i) => {
                const { v, label } = value(s.key, t);
                return (
                  <div className="bar-row" key={s.key}
                    onMouseEnter={() => setTip(`${t} · ${s.label}: ${label}`)} onMouseLeave={() => setTip(null)}
                    onFocus={() => setTip(`${t} · ${s.label}: ${label}`)} onBlur={() => setTip(null)} tabIndex={0}
                    aria-label={`${t}, ${s.label}: ${label}`}>
                    <div className="track">
                      {v !== null && <div className={`fill s${i + 1}`} style={{ width: `${Math.max(v * 100, 0.6)}%` }} />}
                    </div>
                    <span className="bar-value">{v === null ? "N/A" : fmt(v)}</span>
                  </div>
                );
              })}
            </div>
          ))}
          <div className="axis"><span>0%</span><span>50%</span><span>100%</span></div>
        </div>
        <p className="tip" aria-live="polite">{tip ?? " "}</p>
      </figure>

      <h3>RL runs</h3>
      {runs.length === 0 ? <p className="muted">No RL model has been evaluated on this dataset yet.</p> : (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Model ID</th><th>Seed</th><th>Train steps</th>{TIERS.map((t) => <th key={t}>{t}</th>)}<th>Timeouts</th><th>Deadlocks</th></tr></thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.model_id}>
                  <td><code>{r.model_id}</code></td><td>{r.seed}</td><td>{r.train_steps?.toLocaleString()}</td>
                  {TIERS.map((t) => { const c = r.metrics[t]; return <td key={t}>{c ? `${c.successes}/${c.total} (${fmt(pct(c))})` : "N/A"}</td>; })}
                  <td>{TIERS.reduce((a, t) => a + (r.detail[t]?.timeouts ?? 0), 0)}</td>
                  <td>{TIERS.reduce((a, t) => a + (r.detail[t]?.deadlocks ?? 0), 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h3>Baselines</h3>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Agent</th><th>Eval runs</th>{TIERS.map((t) => <th key={t}>{t}</th>)}</tr></thead>
          <tbody>
            {baselines.map((b) => (
              <tr key={b.agent}>
                <td>{b.agent}</td><td>{b.eval_runs}</td>
                {TIERS.map((t) => { const c = b.metrics[t]; return <td key={t}>{c ? `${c.successes}/${c.total} (${fmt(pct(c))})` : "N/A"}</td>; })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">Success = episode reached WON within max_steps. RL inference never calls the solver; a failed RL attempt is reported as a failure, not replaced by BFS. Difficulty is an algorithmic proxy (BFS path length + search effort), not a human rating.</p>
    </section>
  );
}
