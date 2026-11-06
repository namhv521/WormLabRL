import type { ReactNode } from "react";
import type { Action, ApiError, Tier } from "./api";

export const TIERS: Tier[] = ["easy", "medium", "hard"];

export function TierPicker({ value, onChange, disabled }: { value: Tier; onChange: (t: Tier) => void; disabled?: boolean }) {
  return (
    <div className="seg" role="radiogroup" aria-label="Difficulty">
      {TIERS.map((t) => (
        <button key={t} role="radio" aria-checked={value === t} className={value === t ? "on" : ""}
          onClick={() => onChange(t)} disabled={disabled}>
          {t[0].toUpperCase() + t.slice(1)}
        </button>
      ))}
    </div>
  );
}

const ARROWS: [Action, string, string][] = [["UP", "▲", "up"], ["LEFT", "◀", "left"], ["RIGHT", "▶", "right"], ["DOWN", "▼", "down"]];

export function DPad({ onMove, mask, disabled }: { onMove: (a: Action) => void; mask: boolean[]; disabled?: boolean }) {
  const idx: Record<Action, number> = { UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3 };
  return (
    <div className="dpad" role="group" aria-label="Move the worm">
      {ARROWS.map(([a, glyph, cls]) => (
        <button key={a} className={`${cls}${mask[idx[a]] ? "" : " dim"}`} aria-label={`Move ${cls}`}
          onClick={() => onMove(a)} disabled={disabled}>{glyph}</button>
      ))}
    </div>
  );
}

const FRIENDLY: Record<string, string> = {
  LEVEL_POOL_EMPTY: "No certified level available",
  SESSION_EXPIRED: "Session expired",
  MODEL_INCOMPATIBLE: "This model does not support this ruleset",
  MODEL_NOT_FOUND: "Model not found",
  INFERENCE_TIMEOUT: "Model attempt timed out",
  NETWORK: "Server unreachable",
};

export function ErrorBox({ error, children }: { error: ApiError; children?: ReactNode }) {
  return (
    <div className="error" role="alert">
      <strong>{FRIENDLY[error.code] ?? "Something went wrong"}</strong>
      <span>{children ?? error.message}</span>
      <small><code>{error.code}</code>{error.requestId && <> · request {error.requestId.slice(0, 8)}</>}</small>
    </div>
  );
}
