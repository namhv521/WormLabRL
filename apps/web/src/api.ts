// Typed client for /api/v1 (SPEC-05). The backend is the only source of game truth.

export type Action = "UP" | "RIGHT" | "DOWN" | "LEFT";
export type Tier = "easy" | "medium" | "hard";
export type Coord = [number, number];

export interface Level {
  level_id: string;
  ruleset_id: string;
  grid: { width: number; height: number };
  walls: Coord[];
  worm: { segments: Coord[] };
  apples: Coord[];
  exit: Coord;
  verified: boolean;
  difficulty_label: Tier;
  difficulty_score: number;
  level_hash: string;
  canonical_hash: string;
  seed: number | null;
  generator_version: string;
}

export interface GameState {
  worm: { segments: Coord[] };
  remaining_apples: Coord[];
  step_count: number;
  max_steps: number;
  status: "PLAYING" | "WON" | "LOST" | "TIMEOUT";
  action_mask: boolean[];
}

export interface StepResponse {
  session_id: string;
  state: GameState;
  event: { valid_action: boolean; ate_apple: boolean; won: boolean; deadlock: boolean };
  reward: number;
  terminated: boolean;
  truncated: boolean;
}

export interface SolveResponse {
  level_id: string;
  agent_type: "rl" | "bfs";
  model_id: string | null;
  outcome: "SOLVED" | "FAILED" | "TIMEOUT";
  failure_reason?: string | null;
  actions: Action[];
  steps: number;
  optimal: boolean;
  elapsed_ms: number;
}

export interface Agent {
  model_id: string;
  ruleset_id: string;
  seed: number | null;
  run_id: string | null;
  trained_at: string | null;
  checkpoint_sha256: string;
}

export interface Count { successes: number; total: number }
export type TierMetrics = Record<Tier | "overall", Count | null>;

export interface TrainingRuns {
  runs: { run_id: string; model_id: string; seed: number; train_steps: number; trained_at: string;
          dataset_manifest_hash: string; dataset: string; metrics: TierMetrics;
          detail: Record<string, { mean_steps_solved: number | null; timeouts: number; deadlocks: number } | null> }[];
  baselines: { agent: string; dataset_manifest_hash: string; dataset: string; eval_runs: number; metrics: TierMetrics }[];
}

export class ApiError extends Error {
  constructor(public code: string, message: string, public retryable: boolean, public requestId?: string) {
    super(message);
  }
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api/v1${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError("NETWORK", "Cannot reach the WormLab API. Is the server running?", true);
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const e = data?.error;
    throw new ApiError(e?.code ?? `HTTP_${res.status}`, e?.message ?? res.statusText, !!e?.retryable, e?.request_id);
  }
  return data as T;
}

export const api = {
  generate: (difficulty: Tier, exclude_level_ids: string[] = []) =>
    call<{ level: Level; certification: { verified: boolean; optimal_steps: number } }>(
      "POST", "/levels/generate", { difficulty, exclude_level_ids: exclude_level_ids.slice(-100) }),
  level: (id: string) => call<Level>("GET", `/levels/${encodeURIComponent(id)}`),
  createSession: (level_id: string) =>
    call<{ session_id: string; level_id: string; state: GameState }>("POST", "/sessions", { level_id }),
  act: (sid: string, action: Action) => call<StepResponse>("POST", `/sessions/${sid}/actions`, { action }),
  restart: (sid: string) => call<{ session_id: string; state: GameState }>("POST", `/sessions/${sid}/restart`),
  solve: (level_id: string, agent_type: "rl" | "bfs", model_id?: string) =>
    call<SolveResponse>("POST", "/solve", agent_type === "rl" ? { level_id, agent_type, model_id } : { level_id, agent_type }),
  agents: () => call<Agent[]>("GET", "/agents"),
  trainingRuns: () => call<TrainingRuns>("GET", "/training-runs"),
};
