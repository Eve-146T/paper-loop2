package paper.loop2.game

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * Scripted free-movement bot (no neural net yet). It runs a small loop: head **out** into open
 * space, **arc** around to start enclosing, then run **home** to bank the loop — banking the moment
 * its head is back on its own land (handled by [World.capture]). A reactive safety filter steers it
 * away from the wall, away from crossing its own trail, and out of pockets too small to escape.
 *
 * Pure heading logic — it only ever sets the player's target heading; the world integrates motion.
 */
object Bot {
    private const val OUT_MIN = 0.7f       // seconds on the outward leg, min ...
    private const val OUT_RND = 1.0f       // ... plus rng
    private const val SIDE_MIN = 0.6f      // seconds on the perpendicular leg
    private const val SIDE_RND = 0.8f
    private const val MAX_TRAIL = 78       // bank once this many trail cells are out
    private const val DANGER_DIST = 6f     // an enemy head this close -> run home
    private const val LOOK = 3.4f          // self-trail / pocket lookahead, cells
    private const val WALL_LOOK = 5.5f     // wall lookahead, cells
    private const val NEEDFREE = 16        // never steer into a pocket smaller than this

    // candidate heading offsets from the current heading (reactive avoidance)
    private val OFFS = floatArrayOf(0f, 0.22f, -0.22f, 0.48f, -0.48f, 0.8f, -0.8f, 1.2f, -1.2f, 1.7f, -1.7f)

    fun decide(w: World, p: Player) {
        val self = p.id + 1
        val onLand = w.ownerAt(p.x, p.y) == self

        // start a fresh venture whenever we're safely home with no trail
        if (onLand && !p.hasTrail) {
            p.botPhase = 0
            p.botRem = OUT_MIN + w.rng.nextFloat() * OUT_RND
            p.botTurnSign = if (w.rng.nextBoolean()) 1f else -1f
            p.botGoal = outwardHeading(w, p, self)
        }
        p.botRem -= BOT_DECIDE_DT

        // bail home early if the trail is long or an enemy head is near
        var nearest = Float.MAX_VALUE
        for (o in w.players) if (o !== p && o.alive) { val d = p.distTo(o); if (d < nearest) nearest = d }
        if (p.hasTrail && (p.trailCells.size >= MAX_TRAIL || nearest < DANGER_DIST)) p.botPhase = 2

        if (p.botRem <= 0f) when (p.botPhase) {
            0 -> { p.botPhase = 1; p.botRem = SIDE_MIN + w.rng.nextFloat() * SIDE_RND }
            1 -> { p.botPhase = 2; p.botRem = 999f }
            else -> {}
        }

        // A definite rectangle/triangle venture: out leg, a 90° side leg, then beeline back to the
        // exit anchor (re-entering territory banks the loop). botGoal holds the outward heading.
        val desired = when (p.botPhase) {
            0 -> p.botGoal
            1 -> wrapAngle(p.botGoal + p.botTurnSign * HALF_PI)
            else -> atan2(p.exitY - p.y, p.exitX - p.x)   // home anchor
        }

        w.steerToward(p, bestHeading(w, p, self, desired))
    }

    /** Pick the achievable heading closest to [desired] that avoids the wall, our own trail, pockets. */
    private fun bestHeading(w: World, p: Player, self: Int, desired: Float): Float {
        var best = desired; var bestSc = -1e30f
        for (off in OFFS) {
            val ang = wrapAngle(p.heading + off)
            var sc = -abs(wrapAngle(ang - desired)) * 1.3f                 // prefer aligned with intent
            val wall = clearWall(w, p, ang)
            if (wall < 1.3f) sc -= 200f else sc += minOf(wall, WALL_LOOK) * 0.22f
            val st = selfTrailAhead(w, p, self, ang)
            if (st < LOOK) sc -= 80f * (1f - st / LOOK)                    // would re-cross our trail
            val lx = p.x + cos(ang) * LOOK; val ly = p.y + sin(ang) * LOOK
            if (inBounds(lx, ly)) {
                if (w.reachFrom(w.cellX(lx), w.cellY(ly), self, NEEDFREE + 6) < NEEDFREE) sc -= 60f
            } else sc -= 80f
            if (sc > bestSc) { bestSc = sc; best = ang }
        }
        return best
    }

    /** Best heading to begin a venture: most open ground, well clear of the wall. */
    private fun outwardHeading(w: World, p: Player, self: Int): Float {
        var best = p.heading; var bs = -1e30f
        for (k in 0 until 12) {
            val ang = k * (TWO_PI / 12f)
            val wall = clearWall(w, p, ang)
            if (wall < 3f) continue
            var empty = 0
            for (s in 1..5) {
                val px = p.x + cos(ang) * s; val py = p.y + sin(ang) * s
                if (inBounds(px, py) && w.ownerAt(px, py) != self) empty++
            }
            val sc = empty + wall * 0.1f + w.rng.nextFloat() * 0.6f
            if (sc > bs) { bs = sc; best = ang }
        }
        return best
    }

    /** Distance ahead along [ang] before the head would leave the circular arena (capped at WALL_LOOK). */
    private fun clearWall(w: World, p: Player, ang: Float): Float {
        val dx = cos(ang); val dy = sin(ang)
        var d = 0.5f
        while (d <= WALL_LOOK) {
            if (!inBounds(p.x + dx * d, p.y + dy * d)) return d
            d += 0.5f
        }
        return WALL_LOOK
    }

    /** Distance ahead along [ang] to our own trail (capped at LOOK); LOOK if clear. Starts past the
     *  head so the freshly-laid trail right behind us never counts. */
    private fun selfTrailAhead(w: World, p: Player, self: Int, ang: Float): Float {
        if (!p.hasTrail) return LOOK
        val dx = cos(ang); val dy = sin(ang)
        var d = 1.1f
        while (d <= LOOK) {
            val x = p.x + dx * d; val y = p.y + dy * d
            if (inBounds(x, y) && w.trail[w.idx(w.cellX(x), w.cellY(y))].toInt() == self) return d
            d += 0.5f
        }
        return LOOK
    }

    private fun inBounds(x: Float, y: Float): Boolean {
        val dx = x - ARENA_CX; val dy = y - ARENA_CY
        val lim = ARENA_R - WALL_MARGIN
        return dx * dx + dy * dy < lim * lim
    }
}
