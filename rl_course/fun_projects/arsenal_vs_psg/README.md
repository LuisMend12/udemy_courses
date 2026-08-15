# Arsenal vs PSG, learned from scratch

Two tabular Q-learning agents that learn to play football against each other,
then play the tie out with commentary.

```bash
python football_rl.py                  # train, narrate a match, print the odds
python football_rl.py --seed 7         # a different night at the Emirates
python football_rl.py --runs 5         # average the odds over 5 training runs
python football_rl.py --episodes 30000 # let them learn longer
```

## The frontend

`frontend/index.html` plays the trained policy in the browser: the ball moves
between zones, commentary streams in, and a side panel shows the Q-values for
the state the agent is looking at right now, so you can watch the table being
read. There is also a Monte Carlo button that replays the fixture 1,000 times.

It is one self-contained file with no dependencies and no server - the trained
tables are baked in, so refresh it from the trainer whenever you retrain:

```bash
python football_rl.py --episodes 12000 --seed 42 --export frontend/policy.json
cd frontend && python build.py policy.json
```

The engine is a direct port of `MatchEngine` to JavaScript. The port is checked
against the Python numbers rather than assumed: 2,000 headless matches give
ARS 1.36 - 0.78 PSG against Python's 1.44 - 0.87 on the same seed, with the same
win/draw/loss split.

There's also a [3D version](frontend-3d/index.html): the same engine and the
same trained tables, but the pitch is a tilted broadcast camera built out of
CSS 3D transforms, and the ball actually lifts off the turf on a long ball or
a shot instead of just sliding between zones. No WebGL, no dependency - the
ground is a flat scene rotated with `rotateX`, and the ball is a billboarded
sphere translated along `translateZ` in an arc sized to the action, with a
shadow that shrinks as it climbs. Retrain and re-export once, then inline the
tables into both frontends:

```bash
python football_rl.py --episodes 12000 --seed 42 --export frontend/policy.json
cd frontend && python build.py policy.json
cd ../frontend-3d && python build.py ../frontend/policy.json
```

## The frontend

`frontend/index.html` plays the **trained** policy in the browser &mdash; open it
directly, no server needed. The engine is ported to JS and the Q-tables are
inlined, so the page is the real learned result rather than a re-simulation:

```bash
python football_rl.py --episodes 12000 --seed 42 --export frontend/policy.json
cd frontend && python build.py policy.json     # inlines the tables into index.html
```

What it shows, all of it real data from the tables:

- a live pitch with the ball moving zone to zone, a fading trail of the last few
  touches, and the active zone framed in the possessing club's colour
- **the agent's Q-values for the current state**, updated every touch, with
  forbidden actions hatched and exploration flagged when it overrides the greedy pick
- streaming commentary, live match stats, and a cumulative **xG race** over 90
  minutes with goals marked
- a full-time verdict comparing the scoreline against the xG
- Monte Carlo odds over 1,000 in-browser matches

The JS port was checked against the Python trainer rather than assumed: 2,000
headless matches give ARS 1.36 &ndash; 0.78 and a 41/42/17 win-draw-loss split,
against Python's 1.44 &ndash; 0.87 and 41/41/18 on the same seed.

## The MDP

The team on the ball is the acting agent. Its state is four variables:

| variable | values |
| --- | --- |
| `zone` | own third / midfield / final third / the box |
| `pressure` | on the ball in space, or being pressed |
| `stamina` | fresh (`<30'`), legs going (`<65'`), gassed |
| `diff` | behind / level / ahead (clipped, so it fits the table) |

That is 4 x 2 x 3 x 3 = **72 states**, and five actions: `short_pass`,
`long_ball`, `carry`, `shoot`, `recycle`. Small enough to solve exactly, big
enough that the two sides end up playing visibly different football.

Reward is basically just the goal: `+1` scoring, `-1` conceding, plus tiny costs
for giving it away, stalling, or shooting from nowhere.

## Self-play with a held-open update

After a team acts, the *opponent* plays before that team sees another state. So
each team's Q-update is held open until it is next on the ball, and everything
that happens in between - most importantly conceding - is banked into that
pending reward. One decision per possession-touch, semi-Markov style.

Both tables train simultaneously, so each side is learning against a moving
target.

## What went wrong on the way (the interesting part)

Every one of these was the agents playing my model correctly while the model was
wrong about football:

- **Midfield pot-shots.** Both teams shot from midfield ~60% of the time,
  because a blocked shot could win a corner and a corner was worth more than the
  shot. Corners are now only available from the final third.
- **Shooting as a safe clearance.** Even with corners gated, hoofing it from
  40 yards gave the keeper a *dead* ball instead of conceding a live midfield
  turnover - so a hopeless shot was the cheapest way to get rid of possession.
  Shots are now illegal outside the final third rather than endlessly re-priced.
- **Recycling out of the box.** With `gamma = 0.97` and possessions running
  dozens of touches, simply *still having the ball* was worth about as much as a
  30% chance to score, so the agents passed backwards instead of shooting.
- **Cowardice.** Dropping `gamma` to 0.90 fixed the box and broke midfield: both
  sides refused any forward pass and PSG stopped scoring altogether (0.04
  goals/match, 60% of matches 0-0). The real culprit was a `-0.15` turnover
  penalty - seven giveaways priced at a goal, when real teams concede possession
  a hundred times a match. Shaping costs went to near-zero and the dynamics do
  the punishing now.

Final calibration lands around **1.7 - 1.2**, with 1-1, 1-0, 2-0 and 0-0 the
most common results - roughly right for a real fixture.

## The result is less certain than one run makes it look

Both tables chase a moving target, so self-play is non-stationary and different
seeds settle on different equilibria. Averaged over five independent runs:

```
Arsenal win   45.4%
Draw          29.6%
PSG win       25.0%      average scoreline  ARS 1.67 - 1.16 PSG
```

But the per-run Arsenal win rate was **42%, 48%, 27%, 44%, 66%** - a 39-point
spread. On seed 7 alone PSG comes out the 52% favourite. So `--runs` exists
because any single training run's odds are mostly seed noise, and the script
prints that spread rather than quietly reporting one run as the answer.

## What they learned

The two policies converge on recognisably different football, which comes purely
from the rating vectors:

```
Arsenal        own third   short_pass 78%      build out from the back
               midfield    short_pass 39%, recycle 33%, long_ball 22%
               final third shoot 39%, short_pass 28%
               box         shoot 67%

PSG            own third   short_pass 67%
               midfield    long_ball 39%       go vertical, skip the press
               final third carry 61%           dribble your man, get in the box
               box         shoot 61%
```

Arsenal circulates and shoots earlier, leaning on set pieces (`set_pieces=0.94`).
PSG goes direct through midfield and then dribbles into the area, which is what
`dribbling=0.92` buys you. Nobody told either side to do that.

Ratings and rosters are eyeballed for flavour, not scouted - this is a toy.
