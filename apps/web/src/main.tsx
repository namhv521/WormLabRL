import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Experiments } from "./Experiments";
import { Lab } from "./Lab";
import { Play } from "./Play";
import "./styles.css";

type Route = "play" | "lab" | "experiments" | "about";
const ROUTES: [Route, string][] = [["play", "Play"], ["lab", "Watch AI"], ["experiments", "Experiments"], ["about", "Rules"]];

function parse(): { route: Route; level?: string } {
  const [, r, level] = window.location.hash.split("/");
  return { route: (ROUTES.find(([k]) => k === r)?.[0] ?? "play"), level };
}

function App() {
  const [loc, setLoc] = useState(parse);
  useEffect(() => {
    const f = () => setLoc(parse());
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);

  return (
    <div className="app">
      <header>
        <a href="#/play" className="brand"><span aria-hidden="true">🪱</span> WormLab RL</a>
        <nav aria-label="Main">
          {ROUTES.map(([k, label]) => (
            <a key={k} href={`#/${k}`} aria-current={loc.route === k ? "page" : undefined}>{label}</a>
          ))}
        </nav>
      </header>
      <main>
        {loc.route === "play" && <Play onWatch={(id) => { window.location.hash = `#/lab/${id}`; }} />}
        {loc.route === "lab" && <Lab initialLevelId={loc.level} />}
        {loc.route === "experiments" && <Experiments />}
        {loc.route === "about" && <About />}
      </main>
    </div>
  );
}

function About() {
  return (
    <section className="about prose">
      <h2>Rules — ruleset <code>worm-grid-v1</code></h2>
      <ul>
        <li>Move the worm one cell per action: up, right, down or left.</li>
        <li>You cannot leave the board, enter a wall, reverse into your own neck, or hit your body. The tail cell is free if the tail moves away this turn.</li>
        <li>Eating an apple makes the worm one cell longer.</li>
        <li>You win when every apple is eaten <em>and</em> the head reaches the exit. Passing the exit earlier does nothing.</li>
        <li>A blocked move still costs a move. Running out of moves or boxing yourself in ends the attempt.</li>
      </ul>
      <h3>What the labels mean</h3>
      <dl>
        <dt>Verified solvable</dt><dd>The exact BFS solver found a path and the game engine replayed it to a win.</dd>
        <dt>Algorithmic difficulty</dt><dd>A score from the shortest-path length and how many states BFS had to explore. It is not a rating from human players.</dd>
        <dt>AI success rate</dt><dd>How many held-out levels — never seen in training — the RL model finished on its own.</dd>
        <dt>New level</dt><dd>A certified level from the pool. Boards are finite, so the supply of distinct puzzles is too.</dd>
      </dl>
    </section>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
