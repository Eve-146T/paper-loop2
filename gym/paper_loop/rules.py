"""Constants mirrored 1:1 from the Kotlin sim (World.kt / Player.kt / Geom.kt).

Single source of truth for the rule kernel. Unlike Paper Loop 1 (grid-locked, 4
cardinal moves), Paper Loop 2 is **free continuous movement**: the head has a float
position + heading and is steered toward a target heading at a capped turn rate.
The map is a **circle** (ARENA_R inside the GW*GH grid). Territory + capture are
still grid-based (the proven flood-fill), so the rule kernel is: continuous head
integration, a thick trail brush stamped onto the grid, grid capture (BFS flood +
enclosure elimination), continuous collisions (head-to-head, trail-cut, geometric
self-cross), and the circular-wall slide/die.

The render-only layers of World.kt (animated death dissolve, trail retract, audio,
haptics, render interpolation) are deliberately NOT ported — in the gym death is
permanent and territory clears instantly. Respawn is an optional knob (mirrors the
game's RESPAWN toggle); the pretty dissolve is reconstructed only at replay time.
"""
from math import pi

# --- arena / grid (World.kt) ------------------------------------------------
GW = 200                 # grid width  (World.kt:GW)
GH = 200                 # grid height (World.kt:GH)
N_CELLS = GW * GH
ARENA_CX = 100.0         # arena centre x (World.kt:ARENA_CX)
ARENA_CY = 100.0         # arena centre y
ARENA_R = 94.0           # circular playfield radius, cells — the MAP is a circle
MAX_PLAYERS = 12         # Palette team-colour cap

# --- movement (World.kt) ----------------------------------------------------
SUB_DT = 1.0 / 60.0      # fixed simulation sub-step
SPEED = 20.0             # head speed, cells/sec
TURN_RATE = 5.0          # max steering, rad/sec  (min turn radius = SPEED/TURN_RATE = 4 cells)
HEAD_R = 0.95            # head half-extent, cells
TRAIL_R = 0.95           # trail brush radius, cells (>=1-cell band, no fill gaps)
START_R = 10.0           # starting territory disc radius, cells
TRAIL_DECIM = 0.5        # min spacing between recorded polyline points
TRAIL_ROOT = 1.2         # polyline starts this far inside the turf (render attach; kept for parity)
SELF_GRACE = 3.8         # path length behind the head excluded from self-cross tests
SELF_CUT_R = 0.7         # how close the head must come to its own old trail to die
HEAD_KILL = 1.5          # head-to-head crash distance
WALL_MARGIN = 0.6        # head rides the ring this far inside the arena edge
WALL_DIE_DOT = 0.866     # hit the wall steeper than 60deg from tangent (v.n > sin60) -> die
BOT_DECIDE_DT = 0.07     # decision cadence (World.kt BOT_DECIDE_DT)

# One env step = one *decision*; the sim advances SUBSTEPS fixed sub-steps holding
# the chosen target heading (matches on-device: bots decide every BOT_DECIDE_DT and
# hold target heading between decisions). round(0.07 / (1/60)) = 4.
SUBSTEPS = 4
DECISION_DT = SUBSTEPS * SUB_DT          # ~0.0667 s of game time per env step

# --- actions: relative turn buckets ----------------------------------------
# 5 buckets: hard-left / soft-left / straight / soft-right / hard-right. At a
# decision the target heading is set to heading + offset; over the SUBSTEPS the head
# turns toward it (capped at TURN_RATE) so the chosen turn is executed exactly.
# offset(a) = (a - 2) * (MAX_TURN_PER_DECISION / 2); a in 0..4.
N_ACTIONS = 5
MAX_TURN_PER_DECISION = TURN_RATE * DECISION_DT   # = 0.3333 rad ~= 19 deg

# Geometry constants (Geom.kt)
PI_F = float(pi)
TWO_PI = float(2.0 * pi)
HALF_PI = float(pi / 2.0)

# --- per-player INT fields  pi[B, P, NF] -----------------------------------
ALIVE, HAS, AREA, PEAK, KILLS, TLEN, RESPAWN, POLN = range(8)
NF = 8                  # POLN = polyline point count; RESPAWN = steps until respawn (0 = none)

# --- per-player FLOAT fields  pf[B, P, NFF] --------------------------------
FX, FY, FPX, FPY, FHEAD, FTHEAD, FEXITX, FEXITY, FDEATHARC = range(9)
NFF = 9

# --- per-env INT fields  ei[B, ME] -----------------------------------------
NPLAY, ESTEP, TOKEN, EDONE = range(4)
ME = 4

# --- polyline ring (per player) --------------------------------------------
MAXPTS = 1024           # max recorded polyline points (self-cross geometry test)

# --- observation layout -----------------------------------------------------
OBS_R_DEFAULT = 10              # local crop radius -> (2R+1)^2 window
OBS_C = 6                       # wall, own_terr, own_trail, enemy_terr, enemy_trail, enemy_head
OBS_S = 12                      # scalar features
