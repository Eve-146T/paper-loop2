package paper.loop2.game

import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

// --- arena / movement tunables (the "knobs") ---------------------------------
const val GW = 200             // grid width in cells (the circular arena is inscribed in it)
const val GH = 200             // grid height in cells
const val ARENA_CX = 100f      // arena centre x (= GW/2)
const val ARENA_CY = 100f      // arena centre y (= GH/2)
const val ARENA_R = 94f        // circular playfield radius, cells — the MAP is a circle
const val SUB_DT = 1f / 60f    // fixed simulation sub-step (the sim is frame-rate-independent)
const val SPEED = 20f          // head speed, cells/sec (matches Paper Loop 1's on-screen pace)
const val TURN_RATE = 5.0f     // max steering, rad/sec  (min turn radius = SPEED / TURN_RATE)
const val HEAD_R = 0.95f       // head half-extent, cells (render + crash size)
const val TRAIL_R = 0.95f      // trail brush radius, cells (a solid >=1-cell band, no fill gaps)
const val START_R = 10f        // starting territory disc radius, cells (a grounded home base)
const val TRAIL_DECIM = 0.5f   // min spacing between recorded polyline points
const val TRAIL_ROOT = 1.2f    // ribbon starts this far inside the turf; your own turf is redrawn over
                               // the root so it tucks under your edge (clean attach, no gap)
const val SELF_GRACE = 3.8f    // path length behind the head excluded from self-crossing tests
const val SELF_CUT_R = 0.7f    // how close the head must come to its own old trail to die
const val HEAD_KILL = 1.5f     // head-to-head crash distance
const val WALL_MARGIN = 0.6f   // head rides the ring this far inside the circular arena edge
const val WALL_DIE_DOT = 0.866f // hit the wall steeper than 60 deg from tangent (v·n > sin60) -> die
const val BOT_DECIDE_DT = 0.07f// bots re-decide this often (they hold heading between decisions)
const val DECIDE_SUBSTEPS = 4  // neural bots decide every this many sub-steps (~BOT_DECIDE_DT); must
                               // match the gym's SUBSTEPS so the policy sees its training cadence
const val N_ACTIONS = 5        // relative-turn buckets the neural policy chooses from
const val MAX_TURN_PER_DECISION = TURN_RATE * DECIDE_SUBSTEPS * SUB_DT // max heading change per decision
const val TRAIL_RETRACT = 110f // cells/sec a dead player's trail zips back home
const val RIPPLE_SPEED = 32f   // cells/sec the territory then dissolves outward (virus ripple)

/** Result snapshot recorded when the human dies. */
data class DeadStats(val scoreTenths: Int, val rank: Int, val total: Int, val kills: Int)

/**
 * The whole simulation: an ownership grid (territory, the source of truth) + a trail grid + a list
 * of players that move **freely** (continuous position/heading), not on grid steps.
 *
 * Each sub-step every alive head turns toward its target heading (capped), advances, and — while
 * outside its own land — paints a thick brush onto the trail grid and records a polyline. Re-entering
 * its own territory closes the loop: the trail becomes land and a flood-fill claims everything
 * enclosed (identical to Paper Loop — the grid makes capture robust for any free-form shape).
 *
 * Side effects (sound / haptics / logging) go through the [onSound]/[onHaptic]/[onLog] seams so the
 * core has **no Android/libGDX dependency** and runs headless in unit tests. They fire for HUMAN
 * events only (10 bots capturing at once would be a wall of noise) — preserve that.
 */
class World(seed: Long = System.nanoTime()) {
    val owner = ByteArray(GW * GH)   // 0 = empty, else playerId+1
    val trail = ByteArray(GW * GH)   // 0 = none, else playerId+1
    val players = ArrayList<Player>()
    lateinit var human: Player
        private set

    val rng: Random = Random(seed)
    var humanDeadStats: DeadStats? = null; private set
    var respawnEnabled = true
    var stepCount = 0; private set
    var ownerVersion = 0; private set    // bumped on every territory change (renderer rebuild trigger)
    var resetGen = 0; private set        // bumped on reset() so the renderer fully rebuilds (a respawn
                                         // can land on the same area as the dead state, fooling the
                                         // per-player area dirty-check into keeping a stale texture)

    // effect seams (wired to SoundFx / Haptics / Gdx.log by the screen; no-ops in tests)
    var onSound: (String) -> Unit = {}
    var onHaptic: (String) -> Unit = {}
    var onLog: (String) -> Unit = {}
    // bot brain seam: if set, this drives non-human bots (the neural BotNet, wired by GameScreen);
    // null = the scripted Bot. Kept as a plain lambda so World stays free of libGDX/Android deps.
    var neuralDecide: ((World) -> Unit)? = null

    private var acc = 0f
    private val doomed = HashSet<Player>()

    // flood-fill scratch (a stamp token avoids clearing a visited[] every capture)
    private val fillStamp = IntArray(GW * GH)
    private var fillToken = 0
    private val bfs = IntArray(GW * GH)

    private val dissolves = ArrayList<Dissolve>()

    /** One dying player's grid being cleared over time instead of all at once. */
    private class Dissolve(val self: Int) {
        var phase = 0
        var trail = IntArray(0)
        var trailCursor = 0
        var acc = 0f
        var seedX = 0; var seedY = 0
        var terr = IntArray(0)
        var terrDist = FloatArray(0)
        var terrCursor = 0
        var radius = 0f
        var audioPlayed = false
    }

    fun idx(x: Int, y: Int) = y * GW + x
    fun inBounds(x: Int, y: Int) = x in 0 until GW && y in 0 until GH
    fun cellX(fx: Float) = floor(fx).toInt().coerceIn(0, GW - 1)
    fun cellY(fy: Float) = floor(fy).toInt().coerceIn(0, GH - 1)
    fun ownerAt(fx: Float, fy: Float): Int = owner[idx(cellX(fx), cellY(fy))].toInt()
    val totalCells get() = GW * GH

    // --- setup -------------------------------------------------------------

    fun reset(numBots: Int, numColors: Int, humanColor: Int = -1) {
        owner.fill(0); trail.fill(0)
        players.clear(); doomed.clear(); dissolves.clear()
        humanDeadStats = null
        acc = 0f; stepCount = 0; resetGen++

        val total = (numBots + 1).coerceAtMost(numColors)
        val colors = MutableList(numColors) { it }.also { it.shuffle(rng) }
        if (humanColor in 0 until numColors) { colors.remove(humanColor); colors.add(0, humanColor) }
        players.add(Player(0, colors[0], isHuman = true, name = "(You)"))
        for (i in 1 until total) {
            players.add(Player(i, colors[i], isHuman = false, name = BOT_NAMES[(i - 1) % BOT_NAMES.size]))
        }
        human = players[0]
        // Spawn bots first, the human LAST, so an overlapping disc can never overwrite your platform.
        val pairs = players.zip(spawnSpots(total))
        for (i in pairs.indices.reversed()) spawnAt(pairs[i].first, pairs[i].second[0], pairs[i].second[1])
        // Start the human heading INWARD (toward the arena centre) so it never drives straight into
        // the circular wall from a near-edge spawn.
        val a = atan2(ARENA_CY - human.y, ARENA_CX - human.x)
        human.heading = a; human.targetHeading = a
    }

    /** Spawn points spread around a ring inside the circular arena (with a little radius jitter). */
    private fun spawnSpots(total: Int): List<IntArray> {
        val spots = ArrayList<IntArray>()
        val ring = ARENA_R * 0.55f
        val base = rng.nextFloat() * TWO_PI
        val lim = (ARENA_R - START_R - 3f)
        for (i in 0 until total) {
            val ang = base + i * (TWO_PI / total.coerceAtLeast(1))
            val rr = (ring * (0.7f + rng.nextFloat() * 0.35f)).coerceAtMost(lim)
            val cx = (ARENA_CX + cos(ang) * rr).toInt().coerceIn(8, GW - 9)
            val cy = (ARENA_CY + sin(ang) * rr).toInt().coerceIn(8, GH - 9)
            spots.add(intArrayOf(cx, cy))
        }
        spots.shuffle(rng)
        return spots
    }

    private fun spawnAt(p: Player, cx: Int, cy: Int) {
        p.x = cx + 0.5f; p.y = cy + 0.5f
        p.prevX = p.x; p.prevY = p.y
        val d = rng.nextInt(4)
        p.heading = d * HALF_PI; p.targetHeading = p.heading; p.botGoal = p.heading
        p.hasTrail = false; p.trailX.clear(); p.trailY.clear(); p.trailCells.clear()
        p.alive = true; p.deathFx = 0f; p.respawn = 0f; p.dyingFrac = 0f
        p.botPhase = 0; p.botTimer = rng.nextFloat() * BOT_DECIDE_DT; p.botRem = 0f
        // fresh life: clear the (stale, possibly large) owned-cell bbox; the disc below repopulates it
        p.resetBounds()
        // round starting disc
        val r = ceil(START_R).toInt()
        for (yy in cy - r..cy + r) for (xx in cx - r..cx + r) {
            val dx = xx + 0.5f - p.x; val dy = yy + 0.5f - p.y
            if (inBounds(xx, yy) && dx * dx + dy * dy <= START_R * START_R) {
                setOwner(idx(xx, yy), p.id + 1); trail[idx(xx, yy)] = 0
            }
        }
        p.peakArea = p.area
    }

    private fun respawnBot(p: Player) {
        var best = intArrayOf(ARENA_CX.toInt(), ARENA_CY.toInt()); var bestEmpty = -1
        repeat(50) {
            val ang = rng.nextFloat() * TWO_PI; val rr = rng.nextFloat() * (ARENA_R - START_R - 4f)
            val cx = (ARENA_CX + cos(ang) * rr).toInt().coerceIn(8, GW - 9)
            val cy = (ARENA_CY + sin(ang) * rr).toInt().coerceIn(8, GH - 9)
            var empty = 0
            for (yy in cy - 4..cy + 4) for (xx in cx - 4..cx + 4)
                if (inBounds(xx, yy) && owner[idx(xx, yy)].toInt() == 0) empty++
            if (empty > bestEmpty) { bestEmpty = empty; best = intArrayOf(cx, cy) }
        }
        spawnAt(p, best[0], best[1])
    }

    // --- per-frame ---------------------------------------------------------

    fun update(dt: Float) {
        for (p in players) {
            if (p.deathFx > 0f) p.deathFx = (p.deathFx - dt / 0.7f).coerceAtLeast(0f)
            if (p.dyingFrac > 0f) {                       // retract the dead player's trail ribbon
                p.dyingFrac -= dt * TRAIL_RETRACT / p.deathArc
                if (p.dyingFrac <= 0f) { p.dyingFrac = 0f; p.trailX.clear(); p.trailY.clear() }
            }
            if (!p.alive && !p.isHuman && p.respawn > 0f) {
                p.respawn -= dt
                if (p.respawn <= 0f) respawnBot(p)
            }
        }
        acc += dt
        var guard = 0
        while (acc >= SUB_DT && guard < 8) { substep(SUB_DT); acc -= SUB_DT; guard++ }
        if (guard == 8) acc = 0f   // fell behind badly; drop the backlog rather than spiral
        advanceDissolves(dt)
    }

    private fun substep(dt: Float) {
        stepCount++
        // 1) bots decide — the neural brain (wired by GameScreen; runs OFF-thread, once per
        // DECIDE_SUBSTEPS window) if present, else the scripted Bot (throttled; bots hold target
        // heading between decisions).
        val nd = neuralDecide
        if (nd != null) {
            nd(this)
        } else {
            for (p in players) if (p.alive && !p.isHuman) {
                p.botTimer -= dt
                if (p.botTimer <= 0f) { Bot.decide(this, p); p.botTimer = BOT_DECIDE_DT }
            }
        }
        // 2) move every head: steer toward target, advance, wall death
        for (p in players) if (p.alive) {
            p.prevX = p.x; p.prevY = p.y
            val diff = wrapAngle(p.targetHeading - p.heading)
            val maxTurn = TURN_RATE * dt
            p.heading = wrapAngle(p.heading + diff.coerceIn(-maxTurn, maxTurn))
            p.x += cos(p.heading) * SPEED * dt
            p.y += sin(p.heading) * SPEED * dt
            // circular arena. Grazing the ring at a shallow angle (<=30 deg from the tangent) slides
            // along it (clamped to the ring, naturally a touch slower); slamming in steeper kills.
            val ddx = p.x - ARENA_CX; val ddy = p.y - ARENA_CY
            val dist = sqrt(ddx * ddx + ddy * ddy)
            val lim = ARENA_R - WALL_MARGIN
            if (dist > lim) {
                val nx = if (dist > 1e-4f) ddx / dist else cos(p.heading)
                val ny = if (dist > 1e-4f) ddy / dist else sin(p.heading)
                p.x = ARENA_CX + nx * lim; p.y = ARENA_CY + ny * lim    // ride the ring
                val into = cos(p.heading) * nx + sin(p.heading) * ny     // how hard we point outward
                if (into > WALL_DIE_DOT) { doomed.add(p); if (p.isHuman) onLog("HUMAN wall slam") }
            }
        }
        // 3) lay trail (and detect brush-crossings of enemy trails) — no capture yet
        for (p in players) if (p.alive && p !in doomed) layTrail(p)
        // 4) collisions among the moved heads
        resolveCollisions()
        // 5) apply deaths
        if (doomed.isNotEmpty()) { for (p in doomed.toList()) killPlayer(p); doomed.clear() }
        // 6) capture (head back on own land) + start-block deaths
        for (p in players) if (p.alive) {
            if (p.hasTrail && ownerAt(p.x, p.y) == p.id + 1) capture(p)
        }
        for (p in players) if (p.alive && p.hasTrail) {
            if (owner[idx(cellX(p.exitX), cellY(p.exitY))].toInt() != p.id + 1) {
                if (p.isHuman) onLog("HUMAN start-block"); killPlayer(p)
            }
        }
    }

    /** Steer a player toward an absolute heading (used by the bot). */
    fun steerToward(p: Player, ang: Float) { p.targetHeading = ang }

    // --- trail laying + brush rasterisation --------------------------------

    private fun layTrail(p: Player) {
        val insideNow = ownerAt(p.x, p.y) == p.id + 1
        if (p.hasTrail) {
            stampSeg(p, p.prevX, p.prevY, p.x, p.y)
            appendPoint(p, p.x, p.y)
        } else if (!insideNow) {
            // just left our territory: open a trail anchored at the last inside point
            p.hasTrail = true
            p.exitX = p.prevX; p.exitY = p.prevY
            p.trailX.clear(); p.trailY.clear(); p.trailCells.clear()
            // start the visible ribbon a couple cells INSIDE the territory so it attaches cleanly
            // (the ribbon is drawn over the turf; capture uses the cell grid, not this polyline)
            val rootX = p.prevX - cos(p.heading) * TRAIL_ROOT; val rootY = p.prevY - sin(p.heading) * TRAIL_ROOT
            p.trailX.add(rootX); p.trailY.add(rootY)
            p.trailX.add(p.prevX); p.trailY.add(p.prevY)
            p.trMinX = minOf(rootX, p.prevX); p.trMaxX = maxOf(rootX, p.prevX)   // seed the trail bbox
            p.trMinY = minOf(rootY, p.prevY); p.trMaxY = maxOf(rootY, p.prevY)
            stampSeg(p, p.prevX, p.prevY, p.x, p.y)
            appendPoint(p, p.x, p.y)
        }
    }

    private fun appendPoint(p: Player, x: Float, y: Float) {
        if (x < p.trMinX) p.trMinX = x; if (x > p.trMaxX) p.trMaxX = x   // grow the trail bbox (render cull)
        if (y < p.trMinY) p.trMinY = y; if (y > p.trMaxY) p.trMaxY = y
        val n = p.trailX.size
        if (n == 0) { p.trailX.add(x); p.trailY.add(y); return }
        val dx = x - p.trailX[n - 1]; val dy = y - p.trailY[n - 1]
        if (dx * dx + dy * dy >= TRAIL_DECIM * TRAIL_DECIM) { p.trailX.add(x); p.trailY.add(y) }
    }

    /** Stamp a thick segment of trail from A to B; cutting any enemy trail the brush sweeps over. */
    private fun stampSeg(p: Player, ax: Float, ay: Float, bx: Float, by: Float) {
        val dist = hypot(bx - ax, by - ay)
        val n = ceil(dist / (TRAIL_R * 0.6f)).toInt().coerceAtLeast(1)
        for (i in 1..n) {
            val t = i / n.toFloat()
            stampDisc(p, ax + (bx - ax) * t, ay + (by - ay) * t)
        }
    }

    private fun stampDisc(p: Player, cx: Float, cy: Float) {
        val self = p.id + 1
        val x0 = floor(cx - TRAIL_R).toInt(); val x1 = ceil(cx + TRAIL_R).toInt()
        val y0 = floor(cy - TRAIL_R).toInt(); val y1 = ceil(cy + TRAIL_R).toInt()
        val r2 = TRAIL_R * TRAIL_R
        for (yy in y0..y1) for (xx in x0..x1) {
            if (xx < 0 || xx >= GW || yy < 0 || yy >= GH) continue
            val dx = xx + 0.5f - cx; val dy = yy + 0.5f - cy
            if (dx * dx + dy * dy > r2) continue
            val id = idx(xx, yy)
            if (owner[id].toInt() == self) continue          // never lay trail on our own land
            val cur = trail[id].toInt()
            if (cur == self) continue
            if (cur != 0) {                                  // our brush crossed an enemy trail -> cut them
                val victim = players[cur - 1]
                if (victim.alive && victim !== p) { doomed.add(victim); creditKill(p, victim) }
            }
            trail[id] = self.toByte(); p.trailCells.add(id)
        }
    }

    // --- collisions --------------------------------------------------------

    private val liveBuf = ArrayList<Player>(16)
    private fun resolveCollisions() {
        val live = liveBuf
        live.clear()
        for (p in players) if (p.alive && p !in doomed) live.add(p)
        // head-to-head crashes: turf-priority then territory decides the survivor
        for (a in live.indices) for (b in a + 1 until live.size) {
            val p = live[a]; val q = live[b]
            if (p.distTo(q) < HEAD_KILL) {
                when (pickCrashSurvivor(p, q)) {
                    p -> { doomed.add(q); creditKill(p, q); if (q.isHuman) onLog("HUMAN head-crash vs ${p.name}") }
                    q -> { doomed.add(p); creditKill(q, p); if (p.isHuman) onLog("HUMAN head-crash vs ${q.name}") }
                    else -> { doomed.add(p); doomed.add(q); if (p.isHuman || q.isHuman) onLog("HUMAN head-crash tie") }
                }
            }
        }
        // a head with NO trail (cruising its own land) running over an enemy trail cuts them
        for (p in live) if (!p.hasTrail) {
            val t = trail[idx(cellX(p.x), cellY(p.y))].toInt()
            if (t != 0 && t != p.id + 1) {
                val victim = players[t - 1]
                if (victim.alive) { doomed.add(victim); creditKill(p, victim) }
            }
        }
        // crossing your OWN trail (beyond the grace zone near the head) kills you
        for (p in live) if (p.hasTrail && p !in doomed && headCrossesOwnTrail(p)) {
            doomed.add(p); if (p.isHuman) onLog("HUMAN self-cross")
        }
    }

    private fun headCrossesOwnTrail(p: Player): Boolean {
        val n = p.trailX.size
        if (n < 2) return false
        var acc = hypot(p.x - p.trailX[n - 1], p.y - p.trailY[n - 1])
        var i = n - 1
        while (i >= 1) {
            val bx = p.trailX[i]; val by = p.trailY[i]
            val ax = p.trailX[i - 1]; val ay = p.trailY[i - 1]
            val segLen = hypot(bx - ax, by - ay)
            if (acc >= SELF_GRACE && distSqPointSeg(p.x, p.y, ax, ay, bx, by) < SELF_CUT_R * SELF_CUT_R) return true
            acc += segLen; i--
        }
        return false
    }

    /** Head-to-head survivor: standing on your own turf wins; else bigger territory; tie kills both. */
    private fun pickCrashSurvivor(p: Player, q: Player): Player? {
        if (ownerAt(p.x, p.y) == p.id + 1) return p
        if (ownerAt(q.x, q.y) == q.id + 1) return q
        return when {
            p.area > q.area -> p
            q.area > p.area -> q
            else -> null
        }
    }

    private fun creditKill(killer: Player, victim: Player) {
        if (killer === victim || !killer.alive) return
        killer.kills++
        if (killer.isHuman) { onSound("eliminate"); onHaptic("kill") }
    }

    // --- capture (trail -> territory + fill enclosed) ----------------------

    private fun capture(p: Player) {
        val self = p.id + 1
        for (id in p.trailCells) { setOwner(id, self); trail[id] = 0 }
        p.trailCells.clear(); p.trailX.clear(); p.trailY.clear(); p.hasTrail = false

        // Flood "outside" from the border, treating self-owned cells as walls — but only within the
        // player's owned bbox (+1 ring) instead of the whole 200x200 grid. Enclosed cells are
        // surrounded by self, so they lie inside the bbox; the whole region between the +1 ring and the
        // grid border is non-self and connected, so flood-reachability from the ring == from the grid
        // border. Result is identical to a full-grid flood, far cheaper. (The arena is a circle well
        // inside the grid, so the +1 ring is always non-self and a valid seed.)
        val x0 = (p.bbMinX - 1).coerceAtLeast(0); val x1 = (p.bbMaxX + 1).coerceAtMost(GW - 1)
        val y0 = (p.bbMinY - 1).coerceAtLeast(0); val y1 = (p.bbMaxY + 1).coerceAtMost(GH - 1)
        fillToken++
        var tail = 0
        fun visit(x: Int, y: Int) {
            val id = idx(x, y)
            if (owner[id].toInt() != self && fillStamp[id] != fillToken) { fillStamp[id] = fillToken; bfs[tail++] = id }
        }
        for (x in x0..x1) { visit(x, y0); visit(x, y1) }
        for (y in y0..y1) { visit(x0, y); visit(x1, y) }
        var head = 0
        while (head < tail) {
            val cur = bfs[head++]; val cx = cur % GW; val cy = cur / GW
            if (cx > x0) visit(cx - 1, cy); if (cx < x1) visit(cx + 1, cy)
            if (cy > y0) visit(cx, cy - 1); if (cy < y1) visit(cx, cy + 1)
        }
        // any non-self cell inside the bbox the flood didn't reach is enclosed -> claim it
        var eliminated: HashSet<Player>? = null
        for (yy in y0..y1) {
            val base = yy * GW
            for (xx in x0..x1) {
                val i = base + xx
                if (owner[i].toInt() != self && fillStamp[i] != fillToken) {
                    val t = trail[i].toInt()
                    if (t != 0 && t != self) {
                        val vic = players[t - 1]
                        (eliminated ?: HashSet<Player>().also { eliminated = it }).add(vic)
                        trail[i] = 0
                    }
                    setOwner(i, self)
                }
            }
        }
        if (p.isHuman) { onSound("capture"); onHaptic("tick") }
        eliminated?.let { elim ->
            val real = elim.filter { it.alive && it !== p }
            if (p.isHuman && real.isNotEmpty()) { onSound("eliminate"); onHaptic("kill") }
            real.forEach { p.kills++; killPlayer(it) }
        }
    }

    // --- death + animated dissolve ----------------------------------------

    private fun killPlayer(p: Player) {
        if (!p.alive) return
        if (p.isHuman) { onLog("HUMAN dies area=${p.area}"); recordHumanDeath() }
        p.alive = false; p.deathFx = 1f
        startDissolve(p, p.id + 1)
        // Keep the polyline so the ribbon can ZIP BACK (retract) like Paper Loop 1 instead of
        // vanishing; arc length sets the retract speed. The cell grid (trailCells) is handed to the
        // dissolve and cleared here.
        var arc = 0f
        for (i in 1 until p.trailX.size) arc += hypot(p.trailX[i] - p.trailX[i - 1], p.trailY[i] - p.trailY[i - 1])
        p.deathArc = arc.coerceAtLeast(0.001f)
        p.dyingFrac = if (p.trailX.size >= 2) 1f else 0f
        p.trailCells.clear(); p.hasTrail = false
        if (p.isHuman) { onSound("die"); onHaptic("fail") }
        else if (respawnEnabled) p.respawn = 2.2f + rng.nextFloat() * 2.2f
    }

    private fun startDissolve(p: Player, self: Int) {
        val d = Dissolve(self)
        if (p.trailCells.isNotEmpty()) {
            d.trail = p.trailCells.toIntArray()
            d.seedX = cellX(p.exitX); d.seedY = cellY(p.exitY)
        } else { d.seedX = cellX(p.x); d.seedY = cellY(p.y) }
        dissolves.add(d)
    }

    private fun advanceDissolves(dt: Float) {
        if (dissolves.isEmpty()) return
        val hx = if (::human.isInitialized) cellX(human.x) else -1
        val hy = if (::human.isInitialized) cellY(human.y) else -1
        val hAlive = ::human.isInitialized && human.alive
        val it = dissolves.iterator()
        while (it.hasNext()) {
            val d = it.next()
            if (d.phase == 0) {
                d.acc += dt * TRAIL_RETRACT
                while (d.acc >= 1f && d.trailCursor < d.trail.size) {
                    d.acc -= 1f
                    val id = d.trail[d.trail.size - 1 - d.trailCursor]
                    if (trail[id].toInt() == d.self) trail[id] = 0
                    d.trailCursor++
                }
                if (d.trailCursor >= d.trail.size) { startTerritory(d); d.phase = 1 }
            } else {
                d.radius += dt * RIPPLE_SPEED
                while (d.terrCursor < d.terr.size && d.terrDist[d.terrCursor] <= d.radius) {
                    val id = d.terr[d.terrCursor]; d.terrCursor++
                    if (owner[id].toInt() == d.self) {
                        setOwner(id, 0)
                        if (hAlive && !d.audioPlayed && id == idx(hx, hy)) { d.audioPlayed = true; onSound("ripple") }
                    }
                }
                if (d.terrCursor >= d.terr.size) it.remove()
            }
        }
    }

    private fun startTerritory(d: Dissolve) {
        val self = d.self
        val sx = d.seedX + 0.5f; val sy = d.seedY + 0.5f
        val cells = ArrayList<Int>()
        for (i in 0 until GW * GH) if (owner[i].toInt() == self) cells.add(i)
        cells.sortBy { id -> val dx = (id % GW) + 0.5f - sx; val dy = (id / GW) + 0.5f - sy; dx * dx + dy * dy }
        d.terr = IntArray(cells.size) { cells[it] }
        d.terrDist = FloatArray(cells.size) {
            val id = d.terr[it]; val dx = (id % GW) + 0.5f - sx; val dy = (id / GW) + 0.5f - sy
            sqrt(dx * dx + dy * dy)
        }
        d.terrCursor = 0; d.radius = 0f
    }

    private fun recordHumanDeath() {
        val h = human
        val rank = 1 + players.count { it !== h && it.alive && it.area > h.area }
        humanDeadStats = DeadStats(toTenths(h.peakArea), rank, players.size, h.kills)
    }

    // --- helpers -----------------------------------------------------------

    private fun setOwner(id: Int, newId: Int) {
        val old = owner[id].toInt()
        if (old == newId) return
        ownerVersion++                       // tells the renderer to rebuild its territory texture
        if (old != 0) players[old - 1].area--
        owner[id] = newId.toByte()
        if (newId != 0) {
            val p = players[newId - 1]
            p.area++; if (p.area > p.peakArea) p.peakArea = p.area
            val cx = id % GW; val cy = id / GW   // expand the owner's bbox to include this cell
            if (cx < p.bbMinX) p.bbMinX = cx; if (cx > p.bbMaxX) p.bbMaxX = cx
            if (cy < p.bbMinY) p.bbMinY = cy; if (cy > p.bbMaxY) p.bbMaxY = cy
        }
    }

    /** Flood-fill from cell (sx,sy): reachable count capped at [cap]; own trail + walls block.
     *  Bots use this to avoid steering into a pocket too small to escape. */
    fun reachFrom(sx: Int, sy: Int, selfId: Int, cap: Int): Int {
        if (!inBounds(sx, sy)) return 0
        fillToken++; val tok = fillToken
        var head = 0; var tail = 0; var cnt = 0
        fillStamp[idx(sx, sy)] = tok; bfs[tail++] = idx(sx, sy)
        while (head < tail && cnt < cap) {
            val c = bfs[head++]; cnt++
            val cx = c % GW; val cy = c / GW
            if (cx > 0) { val cc = c - 1; if (fillStamp[cc] != tok && trail[cc].toInt() != selfId) { fillStamp[cc] = tok; if (tail < bfs.size) bfs[tail++] = cc } }
            if (cx < GW - 1) { val cc = c + 1; if (fillStamp[cc] != tok && trail[cc].toInt() != selfId) { fillStamp[cc] = tok; if (tail < bfs.size) bfs[tail++] = cc } }
            if (cy > 0) { val cc = c - GW; if (fillStamp[cc] != tok && trail[cc].toInt() != selfId) { fillStamp[cc] = tok; if (tail < bfs.size) bfs[tail++] = cc } }
            if (cy < GH - 1) { val cc = c + GW; if (fillStamp[cc] != tok && trail[cc].toInt() != selfId) { fillStamp[cc] = tok; if (tail < bfs.size) bfs[tail++] = cc } }
        }
        return cnt
    }

    fun toTenths(area: Int): Int = ((area * 1000f) / (GW * GH)).roundToInt()
    fun percentTenths(p: Player): Int = toTenths(p.area)

    companion object {
        private val BOT_NAMES = arrayOf(
            "Nova", "Zigzag", "Pixel", "Blitz", "Rogue", "Echo",
            "Vortex", "Comet", "Hex", "Riot", "Glyph", "Quark",
        )
    }
}
