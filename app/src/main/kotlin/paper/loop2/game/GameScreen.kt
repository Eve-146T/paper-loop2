package paper.loop2.game

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.InputAdapter
import com.badlogic.gdx.ScreenAdapter
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.glutils.ShapeRenderer
import com.badlogic.gdx.math.MathUtils
import com.badlogic.gdx.math.Rectangle
import paper.loop2.core.Haptics
import paper.loop2.core.Palette
import paper.loop2.core.Scores
import paper.loop2.core.Settings
import paper.loop2.core.SoundFx
import kotlin.math.atan2

/** Single screen for the whole game: start menu, play, and the game-over card. */
class GameScreen(private val game: PaperLoopGame) : ScreenAdapter() {
    private enum class Phase { READY, PLAYING, OVER }

    private val world = World()
    private val renderer = WorldRenderer()
    private val hud = Hud(game.ui)
    private val botNet = BotNet()
    private val ui get() = game.ui

    private var phase = Phase.READY
    private var anim = 0f
    private var deadTimer = 0f
    private var smoothDelta = 1f / 60f
    private var stats: DeadStats? = null

    // --- frame profiler (debug only; logs to logcat tag "PERF"; gated by PROFILE). Times update /
    //     glClear (where a GPU/vsync stall surfaces) / render / hud separately so the bottleneck is
    //     unambiguous. Left in (compiled out when PROFILE=false) so re-benchmarking is one flag away.
    private var pfN = 0
    private var pfUpdNs = 0L; private var pfClrNs = 0L; private var pfRenderNs = 0L; private var pfHudNs = 0L
    private var pfRebuilds = 0
    private val pfWallMs = FloatArray(PROFILE_WINDOW)
    private val pfCpuMs = FloatArray(PROFILE_WINDOW)
    private var pfBenched = false

    // shown on the start menu
    private var best = 0
    private var newBest = false

    // start-menu bot settings (persisted)
    private var botMode = Settings.botMode        // 0 = AI HARD, 1 = EASY SCRIPTED, 2 = NO BOTS
    private var respawnOn = Settings.respawnBots
    private val botRect = Rectangle()
    private val respawnRect = Rectangle()

    // AI HARD drives bots with the trained net (crash-safe: any inference error reverts to scripted for
    // the run); null if the net failed to load → AI HARD then falls back to the scripted Bot as well.
    private val neuralLambda: ((World) -> Unit)? = if (botNet.ready) {
        { w ->
            try { botNet.actAll(w) } catch (e: Throwable) {
                Gdx.app.error("BotNet", "inference failed; reverting to scripted bots: $e")
                w.neuralDecide = null
            }
        }
    } else null

    // floating joystick (screen pixels, y-down)
    private var joyActive = false
    private var joyPtr = -1
    private var ox = 0f; private var oy = 0f; private var kx = 0f; private var ky = 0f

    private val restartRect = Rectangle()
    private val exitRect = Rectangle()
    private val dayNightRect = Rectangle()
    private val tmpC = Color()

    init {
        // Route the sim's side effects to the platform; sound respects the setting, haptics self-gate.
        world.onSound = { if (Settings.soundOn) SoundFx.play(it) }
        world.onHaptic = {
            when (it) {
                "kill" -> Haptics.kill(); "fail" -> Haptics.fail(); "success" -> Haptics.success()
                "tick" -> Haptics.tick(); else -> Haptics.click()
            }
        }
        world.onLog = { Gdx.app.log("Death", it) }
        // The bot brain is wired per mode in applyBotFlags(): AI HARD = neural net, EASY SCRIPTED = Bot.
        Gdx.app.log("BotNet", if (botNet.ready) "net ready (AI HARD active)" else "no net; AI HARD falls back to scripted")
    }

    private val input = object : InputAdapter() {
        override fun touchDown(sx: Int, sy: Int, pointer: Int, button: Int): Boolean {
            if (phase == Phase.READY && dayNightRect.contains(sx.toFloat(), ui.h - sy.toFloat())) { toggleTheme(); return true }
            when (phase) {
                Phase.READY -> {
                    val x = sx.toFloat(); val y = ui.h - sy.toFloat()
                    if (game.assets.loaded && botRect.contains(x, y)) { cycleBotMode(); return true }
                    if (game.assets.loaded && respawnRect.contains(x, y)) { if (botMode != 2) toggleRespawn(); return true }
                    phase = Phase.PLAYING; SoundFx.play("start"); beginJoy(sx, sy, pointer)
                }
                Phase.PLAYING -> beginJoy(sx, sy, pointer)
                Phase.OVER -> {
                    val x = sx.toFloat(); val y = ui.h - sy.toFloat()
                    if (restartRect.contains(x, y)) restart()
                    else if (exitRect.contains(x, y)) Gdx.app.exit()
                }
            }
            return true
        }
        override fun touchDragged(sx: Int, sy: Int, pointer: Int): Boolean {
            if (phase == Phase.PLAYING && joyActive && pointer == joyPtr) { kx = sx.toFloat(); ky = sy.toFloat(); steer() }
            return true
        }
        override fun touchUp(sx: Int, sy: Int, pointer: Int, button: Int): Boolean {
            if (pointer == joyPtr) { joyActive = false; joyPtr = -1 }
            return true
        }
    }

    private fun toggleTheme() {
        val v = !Palette.dark
        Palette.setDark(v); Settings.setDarkMode(v)
        SoundFx.play("tap"); Haptics.click()
    }

    private fun beginJoy(sx: Int, sy: Int, pointer: Int) {
        if (!joyActive) { joyActive = true; joyPtr = pointer; ox = sx.toFloat(); oy = sy.toFloat(); kx = ox; ky = oy }
    }

    /** Joystick angle sets the human's ABSOLUTE target heading; the world turns toward it (capped). */
    private fun steer() {
        val dx = kx - ox; val dy = ky - oy
        val dz = minOf(ui.w, ui.h) * 0.03f
        if (dx * dx + dy * dy < dz * dz) return         // deadzone: hold current heading
        world.human.targetHeading = atan2(-dy, dx)      // screen y is down -> invert for world y-up
    }

    private fun loadStats() { best = Scores.best(currentScoreId()) }

    /** Best is bucketed per setting: AI HARD / EASY SCRIPTED × (RESPAWN / NO RESPAWN), plus a single
     *  NO BOTS bucket (respawn is irrelevant with no opponents, so it collapses). */
    private fun currentScoreId(): String = when (botMode) {
        2 -> "${game.scoreId}_nobots"
        else -> "${game.scoreId}_${if (botMode == 0) "hard" else "easy"}_${if (respawnOn) "respawn" else "norespawn"}"
    }

    private fun restart() {
        SoundFx.play("tap"); Haptics.click()
        world.reset(botCountFor(botMode), Palette.TEAM.size)
        applyBotFlags()
        renderer.snap(world)
        phase = Phase.READY
        deadTimer = 0f; stats = null; joyActive = false; joyPtr = -1
        loadStats()
    }

    // --- bot settings -----------------------------------------------------

    private fun botCountFor(mode: Int): Int = if (mode == 2) 0 else game.numBots
    private fun applyBotFlags() {
        world.respawnEnabled = respawnOn
        // AI HARD uses the net (scripted fallback if it failed to load); EASY SCRIPTED / NO BOTS don't.
        world.neuralDecide = if (botMode == 0) neuralLambda else null
    }
    private fun botModeLabel(): String = when (botMode) {
        0 -> "AI HARD"; 1 -> "EASY SCRIPTED"; else -> "NO BOTS"
    }

    /** Cycle AI HARD -> EASY SCRIPTED -> NO BOTS. Re-seed only when the opponent COUNT changes
     *  (AI HARD <-> EASY SCRIPTED keeps the same bots — it just swaps their brain). Pin the human's
     *  colour so a settings change doesn't recolour the player. */
    private fun cycleBotMode() {
        val old = botMode
        botMode = (botMode + 1) % 3
        Settings.setBotMode(botMode)
        SoundFx.play("tap"); Haptics.click()
        if (botCountFor(botMode) != botCountFor(old)) {
            world.reset(botCountFor(botMode), Palette.TEAM.size, world.human.colorIdx); renderer.snap(world)
        }
        applyBotFlags(); loadStats()
    }

    private fun toggleRespawn() {
        respawnOn = !respawnOn
        Settings.setRespawnBots(respawnOn); world.respawnEnabled = respawnOn
        SoundFx.play("tap"); Haptics.click(); loadStats()
    }

    override fun show() {
        botMode = Settings.botMode
        respawnOn = Settings.respawnBots
        world.reset(botCountFor(botMode), Palette.TEAM.size)
        applyBotFlags()
        ui.resize(Gdx.graphics.width, Gdx.graphics.height)
        renderer.resize(Gdx.graphics.width, Gdx.graphics.height)
        renderer.snap(world)
        phase = Phase.READY
        loadStats()
        Gdx.input.inputProcessor = input
    }

    override fun render(delta: Float) {
        anim += delta
        // Low-pass the frame time for uniform motion, but clamp generously (up to 10 fps) so the
        // sim stays REAL-TIME on a slow device instead of going slow-motion. The fixed sub-step
        // accumulator in World.update then catches up with extra sub-steps — frame-rate independent.
        smoothDelta += (delta - smoothDelta) * 0.25f
        val dt = smoothDelta.coerceIn(0.004f, 0.1f)
        val t0 = if (PROFILE) System.nanoTime() else 0L
        when (phase) {
            Phase.PLAYING -> { world.update(dt); renderer.follow(world) }
            Phase.READY -> renderer.follow(world)
            // keep simulating behind the game-over card so the death dissolve finishes (like v1)
            Phase.OVER -> { world.update(dt); renderer.follow(world) }
        }
        val tUpd = if (PROFILE) System.nanoTime() else 0L

        Gdx.gl.glClearColor(Palette.VOID.r, Palette.VOID.g, Palette.VOID.b, 1f)
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT)
        val tClr = if (PROFILE) System.nanoTime() else 0L
        renderer.render(world)
        val tRnd = if (PROFILE) System.nanoTime() else 0L

        val loaded = game.assets.loaded
        when (phase) {
            Phase.READY -> { if (loaded) drawMenu() else drawMenuLoading(); drawDayNight() }
            Phase.PLAYING -> {
                hud.render(world, renderer.cam, best, joyActive, ox, oy, kx, ky)
                if (!world.human.alive) { deadTimer += delta; if (deadTimer > 0.7f) enterOver() }
            }
            Phase.OVER -> if (loaded) drawOverCard()
        }
        if (PROFILE) profileFrame(delta, t0, tUpd, tClr, tRnd, System.nanoTime())
    }

    /** Accumulate per-frame timings while PLAYING; every [PROFILE_WINDOW] frames log a summary and a
     *  one-shot BotNet latency benchmark. update/clear/render/hud are timed separately so a GPU
     *  back-pressure stall (which surfaces at the glClear buffer-dequeue) isn't blamed on the sim. */
    private fun profileFrame(delta: Float, t0: Long, tUpd: Long, tClr: Long, tRnd: Long, tEnd: Long) {
        if (phase != Phase.PLAYING) { pfN = 0; pfUpdNs = 0; pfClrNs = 0; pfRenderNs = 0; pfHudNs = 0; pfRebuilds = 0; return }
        if (BENCH_ON_START && !pfBenched) { pfBenched = true; Thread { botNet.benchmark() }.start() }
        pfUpdNs += tUpd - t0
        pfClrNs += tClr - tUpd
        pfRenderNs += tRnd - tClr
        pfHudNs += tEnd - tRnd
        pfRebuilds += renderer.lastRebuilds
        pfWallMs[pfN] = delta * 1000f
        pfCpuMs[pfN] = (tEnd - t0) / 1e6f
        pfN++
        if (pfN < PROFILE_WINDOW) return
        val wall = pfWallMs.copyOf(); wall.sort()
        val cpu = pfCpuMs.copyOf(); cpu.sort()
        fun p(a: FloatArray, q: Float) = a[(q * (a.size - 1)).toInt()]
        val meanWall = wall.average().toFloat()
        Gdx.app.error("PERF",
            "fps=%.1f wall mean=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f | cpu mean=%.2f | upd=%.2f clr=%.2f rnd=%.2f hud=%.2f ms/f rebuild=%.2f/f"
                .format(1000f / meanWall, meanWall, p(wall, .5f), p(wall, .9f), p(wall, .99f), wall.last(),
                    cpu.average().toFloat(),
                    pfUpdNs / 1e6f / pfN, pfClrNs / 1e6f / pfN, pfRenderNs / 1e6f / pfN, pfHudNs / 1e6f / pfN, pfRebuilds.toFloat() / pfN))
        pfN = 0; pfUpdNs = 0; pfClrNs = 0; pfRenderNs = 0; pfHudNs = 0; pfRebuilds = 0
    }

    // --- start menu --------------------------------------------------------

    private fun drawMenu() {
        val w = ui.w; val h = ui.h
        Gdx.gl.glEnable(GL20.GL_BLEND)
        Gdx.gl.glBlendFunc(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA)
        val accent = Palette.team(world.human.colorIdx)

        val titleY = h * 0.86f
        val headCY = h * 0.350f
        val botCY = h * 0.295f
        val respCY = h * 0.215f
        val botLabel = botModeLabel()
        val respLabel = if (respawnOn) "RESPAWN" else "NO RESPAWN"
        val botLw = ui.textWidth(ui.assets.med, botLabel)
        val respLw = ui.textWidth(ui.assets.med, respLabel)
        val gap = w * 0.045f
        val aw = h * 0.012f
        val ah = h * 0.019f
        val rowH = h * 0.072f
        val halfBot = botLw / 2f + gap + 2 * aw + w * 0.05f
        val halfResp = respLw / 2f + gap + 2 * aw + w * 0.05f
        botRect.set(w / 2f - halfBot, botCY - rowH / 2f, halfBot * 2f, rowH)
        respawnRect.set(w / 2f - halfResp, respCY - rowH / 2f, halfResp * 2f, rowH)

        val respDisabled = botMode == 2

        ui.shapes.projectionMatrix = ui.cam.combined
        ui.shapes.begin(ShapeRenderer.ShapeType.Filled)
        ui.shapes.setColor(accent.r, accent.g, accent.b, 0.85f)
        drawChevrons(w / 2f, botCY, botLw, gap, aw, ah)
        if (respDisabled) ui.shapes.setColor(Palette.MUTED.r, Palette.MUTED.g, Palette.MUTED.b, 0.28f)
        drawChevrons(w / 2f, respCY, respLw, gap, aw, ah)
        ui.shapes.end()

        ui.batch.projectionMatrix = ui.cam.combined
        ui.batch.begin()
        // auto-fit the title to the screen width (it overflows on narrow / low-dpi phones)
        val title = "PAPER LOOP 2"
        val tScale = (w * 0.9f / ui.textWidth(ui.assets.huge, title)).coerceAtMost(1f)
        ui.assets.huge.data.setScale(tScale)
        ui.textCentered(ui.assets.huge, title, w / 2f, titleY, Palette.INK)
        ui.assets.huge.data.setScale(1f)
        ui.textCentered(ui.assets.med, "best  ${pct(best)}", w / 2f, titleY - ui.assets.huge.lineHeight * 0.72f, Palette.MUTED)
        ui.textCenteredIn(ui.assets.small, "Bots", w / 2f, headCY, Palette.MUTED)
        ui.textCenteredIn(ui.assets.med, botLabel, w / 2f, botCY, Palette.INK)
        if (respDisabled) { tmpC.set(Palette.MUTED); tmpC.a = 0.4f; ui.textCenteredIn(ui.assets.med, respLabel, w / 2f, respCY, tmpC) }
        else ui.textCenteredIn(ui.assets.med, respLabel, w / 2f, respCY, Palette.INK)
        val a = 0.34f + 0.20f * (0.5f + 0.5f * MathUtils.sin(anim * 2.4f))
        tmpC.set(Palette.MUTED); tmpC.a = a
        ui.textCentered(ui.assets.med, "tap to begin", w / 2f, h * 0.125f, tmpC)
        ui.batch.end()
    }

    private fun drawChevrons(cx: Float, cy: Float, lw: Float, gap: Float, aw: Float, ah: Float) {
        val lBaseX = cx - lw / 2f - gap
        val rBaseX = cx + lw / 2f + gap
        ui.shapes.triangle(lBaseX, cy + ah, lBaseX, cy - ah, lBaseX - 2 * aw, cy)
        ui.shapes.triangle(rBaseX, cy + ah, rBaseX, cy - ah, rBaseX + 2 * aw, cy)
    }

    private fun drawMenuLoading() {
        val w = ui.w; val h = ui.h
        val accent = Palette.team(world.human.colorIdx)
        val r = w * 0.012f
        val a = 0.25f + 0.30f * (0.5f + 0.5f * MathUtils.sin(anim * 3f))
        Gdx.gl.glEnable(GL20.GL_BLEND)
        Gdx.gl.glBlendFunc(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA)
        ui.shapes.projectionMatrix = ui.cam.combined
        ui.shapes.begin(ShapeRenderer.ShapeType.Filled)
        ui.shapes.setColor(accent.r, accent.g, accent.b, a)
        var i = -1
        while (i <= 1) { ui.shapes.circle(w / 2f + i * r * 4f, h * 0.13f, r, 18); i++ }
        ui.shapes.end()
    }

    // --- day / night toggle (start screen only) ----------------------------

    private fun drawDayNight() {
        val w = ui.w; val h = ui.h
        val m = w * 0.06f
        val r = (w * 0.042f).coerceAtMost(h * 0.032f)
        val cx = w - m - r; val cy = h - m - r
        val pad = r * 1.5f
        dayNightRect.set(cx - pad, cy - pad, 2 * pad, 2 * pad)
        val dark = Palette.dark
        Gdx.gl.glEnable(GL20.GL_BLEND)
        Gdx.gl.glBlendFunc(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA)
        ui.shapes.projectionMatrix = ui.cam.combined
        ui.shapes.begin(ShapeRenderer.ShapeType.Filled)
        if (dark) {
            ui.shapes.color = SUN
            ui.shapes.circle(cx, cy, r * 0.60f, 36)
            var i = 0
            while (i < 8) {
                val ang = i * MathUtils.PI / 4f
                val dxr = MathUtils.cos(ang); val dyr = MathUtils.sin(ang)
                ui.shapes.rectLine(cx + dxr * r * 0.86f, cy + dyr * r * 0.86f, cx + dxr * r * 1.18f, cy + dyr * r * 1.18f, r * 0.11f)
                i++
            }
        } else {
            ui.shapes.color = MOON
            ui.shapes.circle(cx, cy, r, 36)
            ui.shapes.color = Palette.ARENA
            ui.shapes.circle(cx + r * 0.42f, cy + r * 0.36f, r * 0.90f, 36)
        }
        ui.shapes.end()
    }

    // --- game over ---------------------------------------------------------

    private fun enterOver() {
        phase = Phase.OVER
        stats = world.humanDeadStats
        val sc = stats?.scoreTenths ?: 0
        newBest = Scores.submit(currentScoreId(), sc)
        loadStats()
        if (newBest) Haptics.success()
    }

    private fun drawOverCard() {
        val w = ui.w; val h = ui.h
        val accent = Palette.team(world.human.colorIdx)
        val cardFill = if (Palette.dark) CARD_DARK else CARD_LIGHT
        val cardW = w * 0.82f; val cardH = h * 0.56f
        val cardX = (w - cardW) / 2f; val cardY = (h - cardH) / 2f
        val bw = cardW * 0.74f; val bh = h * 0.078f; val bx = cardX + (cardW - bw) / 2f
        exitRect.set(bx, cardY + cardH * 0.055f, bw, bh)
        restartRect.set(bx, exitRect.y + bh + h * 0.020f, bw, bh)
        val st = h * 0.0035f

        Gdx.gl.glEnable(GL20.GL_BLEND)
        Gdx.gl.glBlendFunc(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA)
        ui.shapes.projectionMatrix = ui.cam.combined
        ui.shapes.begin(ShapeRenderer.ShapeType.Filled)
        ui.shapes.setColor(0f, 0f, 0f, 0.55f); ui.shapes.rect(0f, 0f, w, h)
        ui.shapes.setColor(accent.r, accent.g, accent.b, 0.9f)
        Draw.roundRect(ui.shapes, cardX - st, cardY - st, cardW + 2 * st, cardH + 2 * st, h * 0.026f)
        ui.roundRect(cardX, cardY, cardW, cardH, h * 0.024f, cardFill)
        ui.shapes.color = accent
        Draw.roundRect(ui.shapes, restartRect.x, restartRect.y, restartRect.width, restartRect.height, restartRect.height * 0.5f)
        ui.shapes.setColor(Palette.INK.r, Palette.INK.g, Palette.INK.b, 0.10f)
        Draw.roundRect(ui.shapes, exitRect.x, exitRect.y, exitRect.width, exitRect.height, exitRect.height * 0.5f)
        ui.shapes.end()

        val sc = stats?.scoreTenths ?: 0
        val rank = stats?.rank ?: 0
        val total = stats?.total ?: 0
        val k = stats?.kills ?: 0
        val bestY = restartRect.y + restartRect.height + h * 0.050f
        val killsY = bestY + h * 0.052f
        val rankY = killsY + h * 0.052f
        val scoreY = rankY + h * 0.090f
        val headY = scoreY + h * 0.064f

        tmpC.set(accent); if (!Palette.dark) tmpC.mul(0.6f, 0.6f, 0.6f, 1f)

        ui.batch.projectionMatrix = ui.cam.combined
        ui.batch.begin()
        ui.textCentered(ui.assets.med, "GAME OVER", w / 2f, headY, tmpC)
        ui.textCentered(ui.assets.huge, pct(sc), w / 2f, scoreY, Palette.INK)
        ui.textCentered(ui.assets.med, "RANK  #$rank / $total", w / 2f, rankY, Palette.INK)
        ui.textCentered(ui.assets.med, "$k kills", w / 2f, killsY, Palette.MUTED)
        if (newBest) {
            val f = ui.assets.small
            val s = 1f + 0.10f * (0.5f + 0.5f * MathUtils.sin(anim * 9f))
            f.data.setScale(s)
            ui.textCentered(f, "NEW BEST", w / 2f, bestY, tmpC)
            f.data.setScale(1f)
        } else {
            ui.textCentered(ui.assets.small, "best  ${pct(best)}", w / 2f, bestY, Palette.MUTED)
        }
        ui.textCenteredIn(ui.assets.big, "PLAY AGAIN", w / 2f, restartRect.y + restartRect.height / 2f, inkOn(accent))
        ui.textCenteredIn(ui.assets.med, "EXIT", w / 2f, exitRect.y + exitRect.height / 2f, Palette.INK)
        ui.batch.end()
    }

    private fun pct(tenths: Int): String = "${tenths / 10}.${tenths % 10}%"

    private fun inkOn(c: Color): Color =
        if (0.299f * c.r + 0.587f * c.g + 0.114f * c.b > 0.62f) ON_LIGHT else Color.WHITE

    override fun resize(width: Int, height: Int) {
        ui.resize(width, height)
        renderer.resize(width, height)
    }

    override fun dispose() {
        renderer.dispose()
        hud.dispose()
        botNet.dispose()
    }

    private companion object {
        const val PROFILE = false             // debug frame profiler -> logcat tag "PERF" (flip true to re-benchmark)
        const val BENCH_ON_START = false      // run the one-shot BotNet latency probe on first frame (debug)
        const val PROFILE_WINDOW = 120        // frames per logged summary
        val SUN = Color(1f, 1f, 1f, 1f)
        val MOON = Color(0.44f, 0.45f, 0.52f, 1f)
        val CARD_LIGHT = Color(0.99f, 0.97f, 0.91f, 1f)
        val CARD_DARK = Color(0.16f, 0.17f, 0.20f, 1f)
        val ON_LIGHT = Color(0.12f, 0.12f, 0.14f, 1f)
    }
}
