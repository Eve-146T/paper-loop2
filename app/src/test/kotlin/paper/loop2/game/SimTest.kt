package paper.loop2.game

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Headless tests for the pure-Kotlin simulation (no libGDX / Android). They drive [World] with a
 * fixed sub-step so behaviour is deterministic, and check the fundamentals: capture grows territory,
 * the incremental area count matches the grid, the wall and self-crossing kill, and a long mixed
 * bot run never throws or corrupts state.
 */
class SimTest {
    private fun tick(w: World, seconds: Float) {
        repeat((seconds / SUB_DT).toInt()) { w.update(SUB_DT) }
    }

    private fun countOwned(w: World, id: Int): Int {
        var c = 0; for (v in w.owner) if (v.toInt() == id + 1) c++; return c
    }

    @Test
    fun captureGrowsTerritory() {
        val w = World(seed = 42L)
        w.reset(numBots = 0, numColors = 12)
        val h = w.human
        val start = h.area
        assertTrue(start > 0, "starting disc should own cells")
        // a modest loop out and back into the home disc: right, up, left, down (kept small so it
        // closes regardless of where on the circular arena the player spawned)
        h.targetHeading = 0f;        tick(w, 0.8f)
        h.targetHeading = HALF_PI;   tick(w, 0.8f)
        h.targetHeading = PI_F;      tick(w, 0.9f)
        h.targetHeading = -HALF_PI;  tick(w, 1.0f)
        assertTrue(h.alive, "human should survive a clean loop")
        assertTrue(h.area > start, "closing the loop should claim enclosed area: ${h.area} vs $start")
    }

    @Test
    fun areaCountMatchesGrid() {
        val w = World(seed = 7L)
        w.reset(numBots = 7, numColors = 12)
        tick(w, 10f)
        for (p in w.players) assertEquals(countOwned(w, p.id), p.area, "area drifted for ${p.name}")
    }

    @Test
    fun longBotRunIsStableAndProductive() {
        val w = World(seed = 99L)
        w.reset(numBots = 7, numColors = 12)
        tick(w, 40f)                                   // must not throw
        assertTrue(w.players.any { it.peakArea > 120 }, "some bot should capture territory")
        // every cell owner is a valid player id
        for (v in w.owner) assertTrue(v.toInt() in 0..w.players.size, "stray owner byte ${v.toInt()}")
    }

    @Test
    fun wallKills() {
        val w = World(seed = 3L)
        w.reset(numBots = 0, numColors = 12)
        val h = w.human
        h.targetHeading = -HALF_PI                     // straight down into the wall
        tick(w, 12f)
        assertFalse(h.alive, "running into the wall should kill the human")
    }

    @Test
    fun wallGrazeSlidesButSlamKills() {
        // grazing the ring tangentially should slide (survive) and ride near the ring
        val w = World(seed = 11L); w.reset(numBots = 0, numColors = 12)
        val h = w.human
        val hx = ARENA_CX + (ARENA_R - WALL_MARGIN) - 1f; val hy = ARENA_CY
        // give the human turf around the spot so leaving it can't trigger the start-block rule
        val cxI = hx.toInt(); val cyI = hy.toInt()
        for (yy in cyI - 24..cyI + 24) for (xx in cxI - 24..cxI + 24) {
            val dx = xx - hx; val dy = yy - hy
            if (w.inBounds(xx, yy) && dx * dx + dy * dy <= 24 * 24) w.owner[w.idx(xx, yy)] = 1
        }
        h.x = hx; h.y = hy; h.prevX = hx; h.prevY = hy
        h.heading = HALF_PI; h.targetHeading = HALF_PI       // up = tangent at the right edge
        tick(w, 1.0f)
        assertTrue(h.alive, "grazing the wall tangentially must not kill")
        val d = Math.hypot((h.x - ARENA_CX).toDouble(), (h.y - ARENA_CY).toDouble()).toFloat()
        assertTrue(d in (ARENA_R - 6f)..ARENA_R, "should ride along the ring, was $d")

        // slamming straight out (90 deg into the wall) should kill
        val w2 = World(seed = 12L); w2.reset(numBots = 0, numColors = 12)
        val h2 = w2.human
        h2.x = ARENA_CX + ARENA_R - 3f; h2.y = ARENA_CY
        h2.heading = 0f; h2.targetHeading = 0f               // straight into the wall
        tick(w2, 1.0f)
        assertFalse(h2.alive, "slamming into the wall must kill")
    }

    @Test
    fun selfCrossKills() {
        val w = World(seed = 5L)
        w.reset(numBots = 0, numColors = 12)
        val h = w.human
        h.targetHeading = 0f; tick(w, 1.6f)            // run well clear of home, laying trail
        // a tight pure circle (always steering past the current heading) re-crosses that trail
        repeat((2.2f / SUB_DT).toInt()) { h.targetHeading = h.heading - 2.0f; w.update(SUB_DT) }
        assertFalse(h.alive, "looping back across our own trail should kill")
    }
}
