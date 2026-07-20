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
    fun fullArenaReadsHundredPercent() {
        // The score % is of the circular arena, not the square grid — so owning the whole circle
        // must read 100.0%, not ~70% (the old GW*GH denominator bug).
        assertTrue(ARENA_CELLS < GW * GH, "arena circle must be smaller than the square grid")
        val ratio = ARENA_CELLS.toFloat() / (GW * GH)
        assertTrue(ratio in 0.6f..0.8f, "sanity: circle is ~70% of the grid, was $ratio")
        val w = World(seed = 1L)
        w.reset(numBots = 0, numColors = 12)
        assertEquals(1000, w.toTenths(ARENA_CELLS), "owning every arena cell must read 100.0%")
        assertEquals(1000, w.toTenths(ARENA_CELLS + 500), "a hair past the rim is clamped to 100.0%")
        assertTrue(w.percentTenths(w.human) < 100, "a fresh starting disc is a small single-digit %")
    }

    /** Give player [id] the whole circular arena as territory (everyone else keeps nothing). */
    private fun fillArena(w: World, id: Int) {
        for (i in w.owner.indices) w.owner[i] = 0
        for (yy in 0 until GH) for (xx in 0 until GW) {
            val dx = xx + 0.5f - ARENA_CX; val dy = yy + 0.5f - ARENA_CY
            if (dx * dx + dy * dy <= ARENA_R * ARENA_R) w.owner[w.idx(xx, yy)] = (id + 1).toByte()
        }
    }

    @Test
    fun respawnNeverLandsOnTurf() {
        // Fill the whole arena with player 1's territory and let a dead bot try to come back: there
        // is no clear ground left, so it must stay dead rather than carve into the turf.
        val w = World(seed = 5L)
        w.reset(numBots = 2, numColors = 12)
        val filler = w.players[1]
        fillArena(w, filler.id)
        val victim = w.players[2]
        victim.alive = false; victim.respawn = 0.01f
        tick(w, 3f)                                    // plenty of respawn attempts
        assertFalse(victim.alive, "bot must not respawn when the arena is full of another's turf")
        assertEquals(0f, victim.spawnWarn, "no site should even be telegraphed")
        for (yy in 0 until GH) for (xx in 0 until GW) {
            val dx = xx + 0.5f - ARENA_CX; val dy = yy + 0.5f - ARENA_CY
            if (dx * dx + dy * dy <= ARENA_R * ARENA_R)
                assertEquals(filler.id + 1, w.owner[w.idx(xx, yy)].toInt(), "turf was overwritten")
        }
    }

    @Test
    fun respawnTelegraphsAndKeepsItsDistance() {
        // The fix for "spawns too sudden": a returning bot first shows its site for SPAWN_WARN
        // seconds, and the site is never close to a living head.
        val w = World(seed = 17L)
        w.reset(numBots = 3, numColors = 12)
        val victim = w.players[3]
        victim.alive = false; victim.respawn = 0.01f
        tick(w, 0.2f)
        assertFalse(victim.alive, "the bot must not pop in the instant its timer expires")
        assertTrue(victim.spawnWarn > 0f, "its spawn site should be telegraphed first")
        for (q in w.players) if (q.alive) {
            val d = Math.hypot((q.x - (victim.spawnCx + 0.5f)).toDouble(), (q.y - (victim.spawnCy + 0.5f)).toDouble())
            assertTrue(d >= SPAWN_CLEAR, "site is only $d cells from ${q.name}, want >= $SPAWN_CLEAR")
        }
        tick(w, SPAWN_WARN + 0.3f)
        assertTrue(victim.alive, "the bot should land once the telegraph runs out")
        assertEquals(0f, victim.spawnWarn, "the telegraph clears on landing")
    }

    @Test
    fun respawnOffKeepsBotsDead() {
        val w = World(seed = 5L)
        w.respawnEnabled = false
        w.reset(numBots = 3, numColors = 12)
        val victim = w.players[2]
        victim.alive = false; victim.respawn = 0.01f
        tick(w, 5f)
        assertFalse(victim.alive, "OFF must leave eliminated bots dead")
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
