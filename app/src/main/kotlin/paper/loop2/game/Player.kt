package paper.loop2.game

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * One player (human or bot). Unlike Paper Loop (grid-locked), the head moves **freely**: a
 * continuous float position + heading angle, steered toward [targetHeading] at a capped turn rate.
 * The trail is a polyline (for the smooth ribbon) plus the set of grid cells it has stamped
 * (for capture + collision). Grid ownership is still the source of truth for territory.
 */
class Player(
    val id: Int,            // 0-based; cell ownership is stored as id+1
    val colorIdx: Int,
    val isHuman: Boolean,
    val name: String,
) {
    var alive = false

    // continuous head state (cell coordinates, y up)
    var x = 0f
    var y = 0f
    var prevX = 0f                  // head position last sub-step (brush is laid prev -> current)
    var prevY = 0f
    var heading = HALF_PI           // radians; default faces "up"
    var targetHeading = HALF_PI

    // trail: the path drawn while outside own territory
    var hasTrail = false
    var exitX = 0f                  // territory point the trail left from (home anchor + dissolve seed)
    var exitY = 0f
    val trailX = FloatArrayList()   // decimated polyline (render ribbon + self-cross test)
    val trailY = FloatArrayList()
    val trailCells = ArrayList<Int>()   // grid cells stamped by the brush, in lay order

    var area = 0                    // owned cell count, kept incrementally
    var peakArea = 0                // high-water mark (scoring)
    var kills = 0

    // owned-cell bounding box (cells), maintained incrementally in World.setOwner and reset at spawn.
    // EXPAND-ONLY (removals don't shrink it) so it always *contains* every owned cell — an
    // over-approximation that's still correct for limiting the capture flood and the render rebuild
    // to a sub-rect instead of scanning the whole 200x200 grid. Empty = bbMaxX < bbMinX.
    var bbMinX = GW; var bbMinY = GH; var bbMaxX = -1; var bbMaxY = -1
    fun resetBounds() { bbMinX = GW; bbMinY = GH; bbMaxX = -1; bbMaxY = -1 }

    // trail bbox (world coords), set when the trail opens and expanded as it grows — lets the renderer
    // cull off-screen trails without walking the polyline every frame.
    var trMinX = 0f; var trMinY = 0f; var trMaxX = 0f; var trMaxY = 0f

    // scripted-bot brain (continuous): see Bot.kt
    var botTimer = 0f               // seconds until the next decision
    var botPhase = 0                // 0 out · 1 arc · 2 home
    var botRem = 0f                 // seconds left in the current phase
    var botTurnSign = 1f            // which way this loop curls
    var botGoal = HALF_PI           // a heading the bot is steering toward

    // render / fx
    var deathFx = 0f                // 1 -> 0 death burst
    var respawn = 0f                // seconds until a dead bot respawns (or looks for a site again)
    // Respawn telegraph: seconds left before this bot lands on the reserved site (0 = none pending),
    // and the cell it will land on. The renderer/minimap draw it so a spawn is never a surprise;
    // World.siteUsable re-checks it every frame and drops it if anyone comes close.
    var spawnWarn = 0f
    var spawnCx = 0
    var spawnCy = 0
    var dyingFrac = 0f             // 1 -> 0: on death the trail ribbon zips back (kept for rendering)
    var deathArc = 1f              // trail arc length at death (drives the retract speed)

    val dirX: Float get() = cos(heading)
    val dirY: Float get() = sin(heading)

    /** Head distance to another head (for head-to-head crash tests). */
    fun distTo(o: Player): Float {
        val dx = x - o.x; val dy = y - o.y
        return sqrt(dx * dx + dy * dy)
    }
}
