"""Arsenal vs PSG, played out by two reinforcement learning agents.

The idea: a football match is just an MDP with a very short horizon per
possession. The team on the ball is the acting agent, the pitch is a handful of
zones, and the only real reward is a goal. So we can hand the whole thing to
tabular Q-learning, let Arsenal and PSG learn against *each other* by self-play,
and then watch the policies they converge on play the tie out.

Three things happen when you run this file:

  1. train()      - self-play Q-learning, both teams learn at the same time
  2. commentate() - one match narrated minute by minute from the greedy policies
  3. monte_carlo() - the same fixture replayed thousands of times for odds

The state the agent sees
------------------------
    zone      0 own third | 1 middle third | 2 final third | 3 penalty area
    pressure  0 the team has time on the ball | 1 they are being pressed
    stamina   0 fresh (<30') | 1 legs going (<65') | 2 gassed
    diff      -1 behind | 0 level | +1 ahead     (clipped, so it fits in a table)

4 x 2 x 3 x 3 = 72 states, 5 actions. Small enough to solve exactly, big enough
that the two teams end up playing visibly different football.

Team ratings are on a 0-1 scale and are eyeballed for flavour, not scouted -
this is a toy, the rosters and numbers are approximate.

    python football_rl.py                  # train, narrate, then run the odds
    python football_rl.py --seed 7         # a different night at the Emirates
    python football_rl.py --episodes 30000 # let them learn for longer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# State / action space
# --------------------------------------------------------------------------- #

N_ZONES, N_PRESSURE, N_STAMINA, N_DIFF = 4, 2, 3, 3
N_STATES = N_ZONES * N_PRESSURE * N_STAMINA * N_DIFF  # 72
ACTIONS = ("short_pass", "long_ball", "carry", "shoot", "recycle")
N_ACTIONS = len(ACTIONS)

ZONE_NAMES = ("their own third", "midfield", "the final third", "the box")

# Reward shaping, kept deliberately tiny. Real teams give the ball away a hundred
# times a match, so pricing a turnover anywhere near a goal teaches cowardice:
# both agents just recycle in midfield all night. The transition model already
# punishes a bad giveaway by handing the opponent a dangerous position - these
# costs only exist to stop the pathological cases (stalling forever, shooting
# from nowhere) being literally free.
R_GOAL = 1.0
R_CONCEDE = -1.0
R_TURNOVER = -0.05
R_WASTED_SHOT = -0.03   # a shot that goes nowhere is still possession thrown away
R_SIDEWAYS = -0.015     # going backwards is safe, but it cannot be entirely free

BIG_CHANCE_XG = 0.15


def encode(zone: int, pressure: int, stamina: int, diff_bucket: int) -> int:
    """Flatten the four state variables into a single table index."""
    return ((zone * N_PRESSURE + pressure) * N_STAMINA + stamina) * N_DIFF + diff_bucket


ZONE_STRIDE = N_PRESSURE * N_STAMINA * N_DIFF


def zone_of(state: int) -> int:
    return state // ZONE_STRIDE


# Shots are only legal from the final third in. Without this the agents converge
# on something that is optimal in the model and absurd on a pitch: a hopeful
# 40-yarder is a *cheap way to give the ball away*, since it hands the keeper a
# dead ball instead of conceding a live midfield turnover. Rather than keep
# re-tuning the cost of a wasted shot until speculative efforts stop paying, cut
# the action out of the zones where it was never realistic anyway.
LEGAL = np.ones((N_ZONES, N_ACTIONS), dtype=bool)
LEGAL[0, ACTIONS.index("shoot")] = False
LEGAL[1, ACTIONS.index("shoot")] = False


def best_legal(q_row: np.ndarray, zone: int) -> tuple[int, float]:
    """Greedy action and its value, restricted to what is legal in this zone."""
    masked = np.where(LEGAL[zone], q_row, -np.inf)
    return int(np.argmax(masked)), float(masked.max())


def clip(x: float, lo: float = 0.03, hi: float = 0.97) -> float:
    return float(min(hi, max(lo, x)))


# --------------------------------------------------------------------------- #
# Teams
# --------------------------------------------------------------------------- #


@dataclass
class Team:
    name: str
    short: str
    passing: float
    dribbling: float
    finishing: float
    pressing: float
    defending: float
    pace: float
    set_pieces: float
    keeper: float
    keys: dict[str, list[str]] = field(default_factory=dict)

    def pick(self, group: str, rng: np.random.Generator) -> str:
        return str(rng.choice(self.keys[group]))


ARSENAL = Team(
    name="Arsenal",
    short="ARS",
    passing=0.86,      # patient, positional, happy to circulate
    dribbling=0.78,
    finishing=0.80,
    pressing=0.88,     # the identity: swarm the ball high up
    defending=0.88,    # hardest side in the league to score against
    pace=0.80,
    set_pieces=0.94,   # the corner routine is a genuine tactical weapon
    keeper=0.85,
    keys={
        "gk": ["Raya"],
        "def": ["Saliba", "Gabriel", "Timber", "Calafiori"],
        "mid": ["Rice", "Zubimendi", "Ødegaard", "Merino"],
        "att": ["Saka", "Gyökeres", "Martinelli", "Trossard", "Eze", "Madueke"],
    },
)

PSG = Team(
    name="Paris Saint-Germain",
    short="PSG",
    passing=0.90,      # Vitinha's side: relentless one-touch circulation
    dribbling=0.92,    # the best 1v1 front line in Europe
    finishing=0.85,
    pressing=0.85,
    defending=0.83,    # they leave the back door open and don't much care
    pace=0.90,
    set_pieces=0.74,
    keeper=0.87,
    keys={
        "gk": ["Chevalier"],
        "def": ["Hakimi", "Marquinhos", "Pacho", "Nuno Mendes"],
        "mid": ["Vitinha", "João Neves", "Fabián Ruiz", "Zaïre-Emery"],
        "att": ["Dembélé", "Doué", "Kvaratskhelia", "Barcola", "Gonçalo Ramos"],
    },
)


# --------------------------------------------------------------------------- #
# The match engine (the environment)
# --------------------------------------------------------------------------- #


class MatchEngine:
    """A turn-based football MDP.

    Exactly one team is on the ball at a time, and that team is the agent whose
    turn it is to act. `step` resolves one action, advances the clock by a
    fraction of a minute, and hands the ball to whoever ends up with it.
    """

    FULL_TIME = 90.0

    def __init__(self, home: Team, away: Team, rng: np.random.Generator):
        self.teams = (home, away)
        self.rng = rng
        self.reset()

    # -- lifecycle ---------------------------------------------------------- #

    def reset(self) -> None:
        self.minute = 0.0
        self.stoppage = float(self.rng.uniform(3.0, 6.0))
        self.score = [0, 0]
        self.pos = int(self.rng.integers(2))  # who kicks off
        self.zone = 1                         # kickoff starts in midfield
        self.pressure = 0
        self.events: list[tuple[float, str, str, int]] = []
        self.stats = [self._blank_stats(), self._blank_stats()]

    @staticmethod
    def _blank_stats() -> dict[str, float]:
        return {
            "touches": 0.0,
            "time": 0.0,
            "shots": 0.0,
            "on_target": 0.0,
            "xg": 0.0,
            "big_chances": 0.0,
            "turnovers": 0.0,
            "final_third_entries": 0.0,
            "set_pieces": 0.0,
        }

    # -- state -------------------------------------------------------------- #

    def stamina_bucket(self) -> int:
        if self.minute < 30.0:
            return 0
        return 1 if self.minute < 65.0 else 2

    def state(self) -> int:
        diff = self.score[self.pos] - self.score[1 - self.pos]
        return encode(self.zone, self.pressure, self.stamina_bucket(),
                      int(np.clip(diff, -1, 1)) + 1)

    def done(self) -> bool:
        return self.minute >= self.FULL_TIME + self.stoppage

    # -- probability models ------------------------------------------------- #

    def _success(self, action: str, atk: Team, dfn: Team) -> tuple[float, int]:
        """Probability the action comes off, and how many zones it gains.

        The zone penalties are what keep the scorelines honest: space compresses
        as you climb the pitch, so the pass that actually breaks the last line is
        an order of magnitude harder than the one in front of your own keeper.
        """
        p, z = self.pressure, self.zone
        tired = 0.06 * self.stamina_bucket()

        if action == "short_pass":
            prob = (0.62 + 0.40 * atk.passing - 0.34 * dfn.defending
                    - 0.14 * p - 0.11 * z - tired)
            return clip(prob), 1

        if action == "long_ball":
            # Skips a line of pressure entirely, at the cost of accuracy.
            prob = (0.24 + 0.30 * atk.passing + 0.14 * atk.pace
                    - 0.32 * dfn.defending - 0.04 * p - 0.07 * z - tired)
            return clip(prob), 2

        if action == "carry":
            # Beat your man. Punished by pressure, rewarded by dribbling - this
            # is the action PSG's front line is built to win.
            prob = (0.38 + 0.52 * atk.dribbling - 0.38 * dfn.defending
                    - 0.20 * p - 0.06 * z - tired)
            return clip(prob), 1

        if action == "recycle":
            # Pass it backwards, buy a breath, break the press. Nearly free
            # higher up the pitch - but in your own third there is nowhere left
            # to go, and playing out from the back under a press bites.
            if z == 0:
                return clip(0.90 - 0.30 * p - 0.5 * tired), 0
            return clip(0.96 - 0.08 * p - 0.5 * tired), -1

        raise ValueError(action)

    def _shot_xg(self, atk: Team, dfn: Team) -> float:
        base = (0.002, 0.010, 0.075, 0.26)[self.zone]
        xg = base * (0.55 + 0.9 * atk.finishing) * (1.0 - 0.45 * (dfn.keeper - 0.5))
        xg *= 1.0 - 0.34 * self.pressure          # a body in the way
        xg *= 1.0 + 0.05 * self.stamina_bucket()  # tired legs defend worse
        return clip(xg, 0.002, 0.85)

    def _set_piece_xg(self, atk: Team, dfn: Team, kind: str) -> float:
        if kind == "penalty":
            return clip(0.70 + 0.15 * atk.finishing - 0.12 * (dfn.keeper - 0.5))
        if kind == "corner":
            return clip(0.045 + 0.10 * atk.set_pieces - 0.05 * (dfn.keeper - 0.5))
        return clip(0.030 + 0.075 * atk.set_pieces - 0.04 * (dfn.keeper - 0.5))

    def _repress(self, dfn: Team) -> int:
        """Does the defending team get pressure back on the ball?"""
        prob = clip(0.06 + 0.46 * dfn.pressing - 0.08 * self.zone
                    + 0.10 * (self.stamina_bucket() == 2), 0.05, 0.90)
        return int(self.rng.random() < prob)

    # -- transitions -------------------------------------------------------- #

    def _turnover(self) -> None:
        """Ball changes hands, and where you lost it decides how much it costs.

        Robbed in your own third and the opponent inherits the final third; give
        it away in midfield and it stays a midfield turnover; lose it attacking
        and you have merely handed them a goal kick. Never a tap-in.
        """
        self.stats[self.pos]["turnovers"] += 1
        self.zone = int(np.clip(2 - self.zone, 0, 2))
        self.pos = 1 - self.pos
        self.pressure = 0
        if self.zone >= 2:  # won the ball high: an entry, same as passing your way in
            self.stats[self.pos]["final_third_entries"] += 1

    def _restart_after_goal(self, scorer: int) -> None:
        self.pos = 1 - scorer
        self.zone = 1
        self.pressure = 0

    def _keeper_restart(self) -> None:
        """Shot saved or wide: the other keeper has it in his hands."""
        self.pos = 1 - self.pos
        self.zone = 0
        self.pressure = 0

    def _log(self, tag: str, text: str) -> None:
        self.events.append((self.minute, tag, text, self.pos))

    def step(self, action_idx: int) -> tuple[float, int | None, bool]:
        """Resolve one action. Returns (reward to the acting team, scorer, done)."""
        action = ACTIONS[action_idx]
        pos = self.pos
        atk, dfn = self.teams[pos], self.teams[1 - pos]
        rng = self.rng
        dt = float(rng.uniform(0.30, 1.25))
        self.stats[pos]["touches"] += 1
        self.stats[pos]["time"] += dt
        self.minute += dt

        if action == "shoot":
            xg = self._shot_xg(atk, dfn)
            shooter = atk.pick("att" if self.zone >= 2 else "mid", rng)
            self.stats[pos]["shots"] += 1
            self.stats[pos]["xg"] += xg
            if xg > BIG_CHANCE_XG:
                self.stats[pos]["big_chances"] += 1

            if rng.random() < xg:
                self.score[pos] += 1
                self.stats[pos]["on_target"] += 1
                self._log("GOAL", f"{shooter} finishes from {ZONE_NAMES[self.zone]}")
                self._restart_after_goal(pos)
                return R_GOAL, pos, self.done()

            # Missed. On target -> save, otherwise blocked or wide.
            if rng.random() < 0.42:
                self.stats[pos]["on_target"] += 1
                self._log("SAVE", f"{shooter} forces {dfn.pick('gk', rng)} into a save")
            else:
                self._log("MISS", f"{shooter} drags it wide")

            # Only a shot from close range gets deflected behind. Gating this on
            # the final third matters more than it looks: leave corners
            # available from midfield and the agents learn to hit hopeful
            # 40-yarders forever, because the corner is worth more than the shot.
            if self.zone >= 2 and rng.random() < 0.30:
                return self._set_piece(atk, dfn, pos, "corner")

            self._keeper_restart()
            return R_WASTED_SHOT, None, self.done()

        # Everything else is a progression attempt.
        prob, gain = self._success(action, atk, dfn)

        if rng.random() < prob:
            group = ("def", "mid", "att", "att")[self.zone]
            player = atk.pick(group, rng)
            new_zone = int(np.clip(self.zone + gain, 0, 3))
            if new_zone >= 2 > self.zone:
                self.stats[pos]["final_third_entries"] += 1
                self._log("PROG", f"{player} breaks {atk.short} into the final third")
            self.zone = new_zone
            self.pressure = 0 if action == "recycle" else self._repress(dfn)
            return (R_SIDEWAYS if action == "recycle" else 0.0), None, self.done()

        # Failed. Fouls in dangerous areas turn losses into set pieces.
        if action == "carry" and self.zone == 3 and rng.random() < 0.13:
            return self._set_piece(atk, dfn, pos, "penalty")
        if action in ("carry", "short_pass") and self.zone == 2 and rng.random() < 0.16:
            return self._set_piece(atk, dfn, pos, "free_kick")

        loser = atk.pick(("def", "mid", "att", "att")[self.zone], rng)
        if self.zone == 0 and self.pressure == 1:  # a genuine press turnover
            self._log("LOSS", f"{loser} is caught on the ball - {dfn.short} pounce")
        self._turnover()
        return R_TURNOVER, None, self.done()

    def _set_piece(self, atk: Team, dfn: Team, pos: int, kind: str):
        """Corners, free kicks and penalties are resolved immediately."""
        xg = self._set_piece_xg(atk, dfn, kind)
        taker = atk.pick("mid" if kind == "free_kick" else "att", self.rng)
        scorer_group = "def" if kind == "corner" else "att"
        self.stats[pos]["set_pieces"] += 1
        self.stats[pos]["shots"] += 1
        self.stats[pos]["xg"] += xg
        if xg > BIG_CHANCE_XG:
            self.stats[pos]["big_chances"] += 1
        label = {"penalty": "penalty", "corner": "corner", "free_kick": "free kick"}[kind]

        if self.rng.random() < xg:
            finisher = taker if kind != "corner" else atk.pick(scorer_group, self.rng)
            self.stats[pos]["on_target"] += 1
            self.score[pos] += 1
            self._log("GOAL", f"{finisher} converts the {label}")
            self._restart_after_goal(pos)
            return R_GOAL, pos, self.done()

        self._log("SETP", f"{taker}'s {label} comes to nothing")
        self._turnover()
        return 0.0, None, self.done()


# --------------------------------------------------------------------------- #
# Self-play Q-learning
# --------------------------------------------------------------------------- #


def train(home: Team, away: Team, episodes: int = 15000, alpha: float = 0.15,
          gamma: float = 0.96, seed: int = 0, verbose: bool = True) -> list[np.ndarray]:
    """Both teams learn simultaneously, each from its own possessions.

    The wrinkle versus textbook Q-learning: after a team acts, the *opponent*
    plays before that team sees a new state. So each team's update is held open
    until it is next on the ball, and anything that happens in between - most
    importantly conceding - is banked into that pending reward. It is
    semi-Markov Q-learning, one decision per possession-touch.

    gamma sits in a narrow band. Too close to 1 and the value of *still having
    the ball* rivals the value of a good chance, so the agents hold position in
    the box instead of shooting; drop it much below 0.95 and the distant goal is
    discounted so hard that neither side will risk a forward pass at all.
    """
    rng = np.random.default_rng(seed)
    q = [np.zeros((N_STATES, N_ACTIONS)) for _ in range(2)]
    env = MatchEngine(home, away, rng)
    pending: list[list | None] = [None, None]
    history = []

    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.75 * episodes))
        env.reset()
        pending = [None, None]

        while not env.done():
            t = env.pos
            s = env.state()
            z = zone_of(s)
            a_star, v_star = best_legal(q[t][s], z)

            # Flush this team's previous decision now that we can see s'.
            if pending[t] is not None:
                ps, pa, pr = pending[t]
                q[t][ps, pa] += alpha * (pr + gamma * v_star - q[t][ps, pa])

            if rng.random() < eps:
                a = int(rng.choice(np.flatnonzero(LEGAL[z])))
            else:
                a = a_star
            pending[t] = [s, a, 0.0]

            reward, scorer, _ = env.step(a)
            pending[t][2] += reward
            if scorer is not None and pending[1 - scorer] is not None:
                pending[1 - scorer][2] += R_CONCEDE  # you let one in

        for t in (0, 1):  # terminal: no bootstrap past full time
            if pending[t] is not None:
                ps, pa, pr = pending[t]
                q[t][ps, pa] += alpha * (pr - q[t][ps, pa])

        history.append(tuple(env.score))
        if verbose and (ep + 1) % max(1, episodes // 6) == 0:
            recent = np.array(history[-1000:])
            print(f"  episode {ep + 1:>6}/{episodes}  eps={eps:.2f}  "
                  f"last-1000 avg score {home.short} {recent[:, 0].mean():.2f} - "
                  f"{recent[:, 1].mean():.2f} {away.short}")

    return q


def greedy(q: list[np.ndarray], state: int, team: int, rng, eps: float = 0.0) -> int:
    zone = zone_of(state)
    if eps and rng.random() < eps:
        return int(rng.choice(np.flatnonzero(LEGAL[zone])))
    return best_legal(q[team][state], zone)[0]


def play(home: Team, away: Team, q: list[np.ndarray], rng, eps: float = 0.0) -> MatchEngine:
    env = MatchEngine(home, away, rng)
    while not env.done():
        env.step(greedy(q, env.state(), env.pos, rng, eps))
    return env


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

TAG_STYLE = {
    "GOAL": "⚽",
    "SAVE": "🧤",
    "MISS": "↗",
    "SETP": "⛳",
    "PROG": "→",
    "LOSS": "⚠",
}


def describe_policy(team: Team, table: np.ndarray) -> str:
    """What did this team actually learn to do in each zone?"""
    lines = [f"  {team.name} learned:"]
    for z in range(N_ZONES):
        counts = np.zeros(N_ACTIONS)
        for p in range(N_PRESSURE):
            for st in range(N_STAMINA):
                for d in range(N_DIFF):
                    counts[best_legal(table[encode(z, p, st, d)], z)[0]] += 1
        total = counts.sum()
        mix = ", ".join(f"{ACTIONS[a]} {counts[a] / total:.0%}"
                        for a in np.argsort(-counts) if counts[a] > 0)
        lines.append(f"    in {ZONE_NAMES[z]:<16} {mix}")
    return "\n".join(lines)


def commentate(env: MatchEngine, key_only: bool = True) -> None:
    home, away = env.teams
    print(f"\n{'=' * 74}")
    print(f"  {home.name}  vs  {away.name}".center(74))
    print(f"{'=' * 74}\n")

    running = [0, 0]
    half_time_shown = False
    for minute, tag, text, team in env.events:
        if key_only and tag == "PROG":
            continue
        if minute >= 45.0 and not half_time_shown:
            print(f"\n  --- HALF TIME   {home.short} {running[0]} - "
                  f"{running[1]} {away.short} ---\n")
            half_time_shown = True

        stamp = (f"90+{int(minute) - 90}'" if minute > 90.0
                 else f"{max(int(minute), 1):>3}'")
        side = env.teams[team].short
        print(f"  {stamp}  {TAG_STYLE.get(tag, ' ')}  [{side}] {text}")

        if tag == "GOAL":
            running[team] += 1
            print(f"        {'':<4}{home.short} {running[0]} - "
                  f"{running[1]} {away.short}")

    print(f"\n  {'-' * 70}")
    print(f"  FULL TIME    {home.short} {env.score[0]} - {env.score[1]} {away.short}")
    print(f"  {'-' * 70}\n")

    def pct(i: int) -> float:
        total = env.stats[0]["time"] + env.stats[1]["time"]
        return 100.0 * env.stats[i]["time"] / max(total, 1e-9)

    rows = [
        ("Possession", f"{pct(0):.0f}%", f"{pct(1):.0f}%"),
        ("Shots", f"{env.stats[0]['shots']:.0f}", f"{env.stats[1]['shots']:.0f}"),
        ("On target", f"{env.stats[0]['on_target']:.0f}", f"{env.stats[1]['on_target']:.0f}"),
        ("xG", f"{env.stats[0]['xg']:.2f}", f"{env.stats[1]['xg']:.2f}"),
        ("Big chances", f"{env.stats[0]['big_chances']:.0f}", f"{env.stats[1]['big_chances']:.0f}"),
        ("Set-piece attempts", f"{env.stats[0]['set_pieces']:.0f}", f"{env.stats[1]['set_pieces']:.0f}"),
        ("Final-third entries", f"{env.stats[0]['final_third_entries']:.0f}",
         f"{env.stats[1]['final_third_entries']:.0f}"),
        ("Turnovers", f"{env.stats[0]['turnovers']:.0f}", f"{env.stats[1]['turnovers']:.0f}"),
    ]
    print(f"  {'':<22}{home.short:>8}{away.short:>10}")
    for label, a, b in rows:
        print(f"  {label:<22}{a:>8}{b:>10}")

    goals = env.score[0] + env.score[1]
    chances = env.stats[0]["big_chances"] + env.stats[1]["big_chances"]
    fun = min(10.0, 1.8 * goals + 0.7 * chances + 1.5)
    print(f"\n  Entertainment rating: {fun:.1f}/10")


def export_tables(home: Team, away: Team, q: list[np.ndarray], path: str,
                  seed: int, episodes: int) -> None:
    """Dump the trained tables so a frontend can play the same learned policy.

    Ships the Q-values, not just the argmax, so a UI can show how close each
    decision was - the interesting part is often the second-choice action.
    """
    import json

    payload = {
        "meta": {"seed": seed, "episodes": episodes, "actions": list(ACTIONS),
                 "zones": list(ZONE_NAMES), "n_states": N_STATES},
        "legal": LEGAL.tolist(),
        "teams": [
            {"name": t.name, "short": t.short, "keys": t.keys,
             "ratings": {k: getattr(t, k) for k in
                         ("passing", "dribbling", "finishing", "pressing",
                          "defending", "pace", "set_pieces", "keeper")},
             "q": [[round(float(v), 4) for v in row] for row in q[i]]}
            for i, t in enumerate((home, away))
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))


def monte_carlo(home: Team, away: Team, q: list[np.ndarray], n: int, seed: int) -> dict:
    """Replay the fixture n times from a trained pair of tables."""
    rng = np.random.default_rng(seed + 999)
    tally = {"home": 0, "draw": 0, "away": 0}
    scorelines: dict[tuple[int, int], int] = {}
    goals_for = np.zeros(2)

    for _ in range(n):
        env = play(home, away, q, rng, eps=0.05)  # a little noise = a little variance
        h, a = env.score
        goals_for += env.score
        scorelines[(h, a)] = scorelines.get((h, a), 0) + 1
        tally["home" if h > a else "away" if a > h else "draw"] += 1

    return {"n": n, "tally": tally, "scorelines": scorelines, "goals_for": goals_for}


def report_odds(home: Team, away: Team, books: list[dict], runs: int) -> None:
    n = sum(b["n"] for b in books)
    tally = {k: sum(b["tally"][k] for b in books) for k in ("home", "draw", "away")}
    goals_for = sum((b["goals_for"] for b in books), np.zeros(2))
    scorelines: dict[tuple[int, int], int] = {}
    for b in books:
        for k, c in b["scorelines"].items():
            scorelines[k] = scorelines.get(k, 0) + c

    print(f"\n{'=' * 74}")
    header = (f"{n:,} matches from {runs} independently trained self-play runs"
              if runs > 1 else f"{n:,} simulated matches from the trained policies")
    print(header.center(74))
    print(f"{'=' * 74}\n")
    print(f"  {home.name:<24} win   {100 * tally['home'] / n:5.1f}%")
    print(f"  {'Draw':<24}       {100 * tally['draw'] / n:5.1f}%")
    print(f"  {away.name:<24} win   {100 * tally['away'] / n:5.1f}%")
    print(f"\n  Average scoreline   {home.short} {goals_for[0] / n:.2f} - "
          f"{goals_for[1] / n:.2f} {away.short}")
    print("\n  Most likely results")
    for (h, a), c in sorted(scorelines.items(), key=lambda kv: -kv[1])[:6]:
        bar = "█" * int(round(40 * c / n))
        print(f"    {h}-{a}   {100 * c / n:5.1f}%  {bar}")

    if runs > 1:
        # Self-play is non-stationary: both tables chase a moving target, so
        # different seeds settle on different equilibria. Show that spread rather
        # than passing off one run's number as the answer.
        spread = [100 * b["tally"]["home"] / b["n"] for b in books]
        print(f"\n  Per-run {home.short} win %: "
              + ", ".join(f"{s:.0f}%" for s in spread)
              + f"   (spread {max(spread) - min(spread):.0f} points)")


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=12000, help="self-play matches")
    ap.add_argument("--sims", type=int, default=1000, help="matches per run for the odds")
    ap.add_argument("--runs", type=int, default=1,
                    help="independent self-play runs to average the odds over")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export", metavar="PATH",
                    help="write the first run's trained Q-tables to JSON")
    args = ap.parse_args()

    books, first_q = [], None
    for r in range(args.runs):
        seed = args.seed + 1000 * r
        label = f" (run {r + 1}/{args.runs}, seed {seed})" if args.runs > 1 else ""
        print(f"Training Arsenal and PSG against each other by self-play{label}...\n")
        q = train(ARSENAL, PSG, episodes=args.episodes, seed=seed,
                  verbose=(args.runs == 1))
        books.append(monte_carlo(ARSENAL, PSG, q, args.sims, seed))
        if first_q is None:
            first_q = q

    print()
    print(describe_policy(ARSENAL, first_q[0]))
    print()
    print(describe_policy(PSG, first_q[1]))

    rng = np.random.default_rng(args.seed)
    commentate(play(ARSENAL, PSG, first_q, rng, eps=0.05))
    report_odds(ARSENAL, PSG, books, args.runs)

    if args.export:
        export_tables(ARSENAL, PSG, first_q, args.export, args.seed, args.episodes)
        print(f"\n  Wrote trained tables to {args.export}")


if __name__ == "__main__":
    main()
