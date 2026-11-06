import type { Coord, GameState, Level } from "./api";

// Shapes + text, never colour alone, distinguish tiles (SPEC-07 §7).
export function Board({ level, state, shake }: { level: Level; state: GameState; shake?: boolean }) {
  const n = level.grid.width;
  const key = (c: Coord) => `${c[0]},${c[1]}`;
  const walls = new Set(level.walls.map(key));
  const apples = new Set(state.remaining_apples.map(key));
  const exitOpen = state.remaining_apples.length === 0;

  const describe = () => {
    const [hx, hy] = state.worm.segments[0];
    return `Board ${n} by ${n}. Worm head at column ${hx + 1}, row ${hy + 1}, length ${state.worm.segments.length}. ` +
      `${state.remaining_apples.length} apples left. Exit at column ${level.exit[0] + 1}, row ${level.exit[1] + 1}` +
      `${exitOpen ? ", open" : ", locked until all apples are eaten"}.`;
  };

  const cells = [];
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const k = `${x},${y}`;
      const isExit = level.exit[0] === x && level.exit[1] === y;
      cells.push(
        <g key={k} transform={`translate(${x},${y})`}>
          <rect className={walls.has(k) ? "tile wall" : "tile"} width={0.96} height={0.96} x={0.02} y={0.02} rx={0.12} />
          {walls.has(k) && <path className="wall-hatch" d="M0.2 0.8 L0.8 0.2 M0.2 0.5 L0.5 0.2 M0.5 0.8 L0.8 0.5" />}
          {isExit && (
            <g className={exitOpen ? "exit open" : "exit"}>
              <rect x={0.18} y={0.14} width={0.64} height={0.74} rx={0.3} />
              <text x={0.5} y={0.62} textAnchor="middle">{exitOpen ? "EXIT" : "🔒"}</text>
            </g>
          )}
          {apples.has(k) && (
            <g className="apple">
              <circle cx={0.5} cy={0.55} r={0.26} />
              <path d="M0.5 0.3 Q0.56 0.16 0.68 0.14" />
            </g>
          )}
        </g>,
      );
    }
  }

  const segs = state.worm.segments;
  const path = segs.map(([x, y], i) => `${i ? "L" : "M"}${x + 0.5} ${y + 0.5}`).join(" ");
  const [hx, hy] = segs[0];
  const neck = segs[1] ?? segs[0];
  const dx = hx - neck[0], dy = hy - neck[1];

  return (
    <figure className={`board${shake ? " shake" : ""}`} aria-label={describe()} role="img">
      <svg viewBox={`-0.1 -0.1 ${n + 0.2} ${n + 0.2}`} aria-hidden="true">
        {cells}
        <path className="worm-body" d={path} />
        {segs.slice(1).map(([x, y], i) => (
          <circle key={i} className="worm-seg" cx={x + 0.5} cy={y + 0.5} r={0.3 - i * 0.012} />
        ))}
        <g className="worm-head" transform={`translate(${hx + 0.5},${hy + 0.5})`}>
          <circle r={0.36} />
          <circle className="eye" cx={dx * 0.14 - dy * 0.13} cy={dy * 0.14 + dx * 0.13} r={0.07} />
          <circle className="eye" cx={dx * 0.14 + dy * 0.13} cy={dy * 0.14 - dx * 0.13} r={0.07} />
        </g>
      </svg>
      <figcaption className="legend">
        <span><i className="sw head" /> Worm head</span>
        <span><i className="sw apple" /> Apple</span>
        <span><i className="sw exit" /> Exit</span>
        <span><i className="sw wall" /> Wall</span>
      </figcaption>
    </figure>
  );
}
