package paper.loop2.game

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.OrthographicCamera
import com.badlogic.gdx.graphics.Pixmap
import com.badlogic.gdx.graphics.Texture
import com.badlogic.gdx.graphics.glutils.ShapeRenderer
import com.badlogic.gdx.math.Vector3
import paper.loop2.core.Palette
import kotlin.math.sqrt

/** Clean, low-key HUD: a small top-left %, subtle floating names, a small minimap and joystick. */
class Hud(private val ui: Ui) {
    private val mmPix = Pixmap(GW, GH, Pixmap.Format.RGBA8888)
    private val mmTex = Texture(mmPix)
    private val mmArr = IntArray(GW * GH)   // packed minimap pixels; bulk-uploaded (faster than drawPixel)
    private var mmTimer = 0f
    private val teamRgba = IntArray(Palette.TEAM.size) { Color.rgba8888(Palette.team(it)) }
    // Trails on the minimap: the team colour lightened, so a trail reads as "the same player,
    // reaching out" without being mistaken for claimed territory.
    private val trailRgba = IntArray(Palette.TEAM.size) {
        Color.rgba8888(Color(Palette.team(it)).lerp(Color.WHITE, 0.40f))
    }
    private val v = Vector3()
    private val tmpC = Color()

    init { mmPix.blending = Pixmap.Blending.None }

    fun render(world: World, cam: OrthographicCamera, best: Int, joyActive: Boolean, ox: Float, oy: Float, kx: Float, ky: Float) {
        mmTimer -= Gdx.graphics.deltaTime
        if (mmTimer <= 0f) { updateMinimap(world); mmTimer = 0.2f }

        val w = ui.w; val h = ui.h
        val m = w * 0.045f
        val mmSize = (w * 0.24f).coerceAtMost(h * 0.16f)
        val pad = mmSize * 0.07f
        val accent = Palette.team(world.human.colorIdx)

        Gdx.gl.glEnable(GL20.GL_BLEND)
        Gdx.gl.glBlendFunc(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA)

        // ---- shapes: minimap panel + joystick ----
        ui.shapes.projectionMatrix = ui.cam.combined
        ui.shapes.begin(ShapeRenderer.ShapeType.Filled)
        ui.roundRect(m, m, mmSize, mmSize, mmSize * 0.08f, Palette.PANEL)
        if (joyActive) {
            val jx = ox; val jy = h - oy
            val baseR = minOf(w, h) * 0.11f
            var dx = kx - ox; var dy = (h - ky) - jy
            val len = sqrt(dx * dx + dy * dy)
            if (len > baseR) { dx = dx / len * baseR; dy = dy / len * baseR }
            ui.shapes.setColor(0f, 0f, 0f, 0.08f); ui.shapes.circle(jx, jy, baseR, 36)
            ui.shapes.setColor(accent.r, accent.g, accent.b, 0.65f); ui.shapes.circle(jx + dx, jy + dy, baseR * 0.42f, 26)
        }
        ui.shapes.end()

        // ---- batch: minimap + dots + names + small score ----
        ui.batch.projectionMatrix = ui.cam.combined
        ui.batch.begin()
        ui.batch.setColor(Color.WHITE)
        val inX = m + pad; val inY = m + pad; val inS = mmSize - 2 * pad
        ui.batch.draw(mmTex, inX, inY, inS, inS)
        for (p in world.players) if (p.alive) {
            val px = inX + p.x / GW * inS
            val py = inY + p.y / GH * inS
            val dr = if (p.isHuman) inS * 0.05f else inS * 0.026f
            ui.batch.color = if (p.isHuman) Palette.INK else Palette.team(p.colorIdx)
            ui.batch.draw(ui.assets.white, px - dr, py - dr, dr * 2, dr * 2)
        }
        ui.batch.setColor(Color.WHITE)

        // Text needs the fonts, which load just after the first frame; until then draw only the
        // shapes/minimap above so play can start immediately without waiting on FreeType.
        if (ui.assets.loaded) {
            // subtle floating names: anchored just above the head's projected top edge with a small
            // pixel gap, so the spacing is the same on any screen and never clips into the square.
            val nameGap = h * 0.006f
            for (p in world.players) if (p.alive) {
                v.set(p.x, p.y + HEAD_R + 0.2f, 0f)   // the head's top edge in world space
                cam.project(v)
                if (v.x in 0f..w && v.y in 0f..h) {
                    tmpC.set(if (p.isHuman) Palette.INK else Palette.team(p.colorIdx)); tmpC.a = 0.8f
                    ui.textCenteredAbove(ui.assets.small, if (p.isHuman) "(You)" else p.name, v.x, v.y + nameGap, tmpC)
                }
            }

            // low-key score, top-left
            val human = world.human
            val rank = 1 + world.players.count { it !== human && it.alive && it.area > human.area }
            val scoreY = h - m * 0.7f
            ui.text(ui.assets.big, pct(world.percentTenths(human)), m, scoreY, if (human.alive) Palette.INK else Palette.MUTED)
            ui.text(ui.assets.small, "#$rank / ${world.players.size}   ·   best ${pct(best)}",
                m, scoreY - ui.assets.big.lineHeight * 0.86f, Palette.MUTED)
        }
        ui.batch.end()
    }

    private fun updateMinimap(world: World) {
        val arenaRgba = Color.rgba8888(Palette.ARENA)   // re-read so the theme toggle is reflected
        val r2 = ARENA_R * ARENA_R
        val arr = mmArr; val owner = world.owner; val trail = world.trail; val players = world.players
        for (y in 0 until GH) {
            val rowBase = (GH - 1 - y) * GW   // minimap is y-flipped vs the world grid
            val wyBase = y * GW
            val dy = y + 0.5f - ARENA_CY
            for (x in 0 until GW) {
                // outside the circular arena -> transparent, so the minimap reads as a disc
                val dx = x + 0.5f - ARENA_CX
                if (dx * dx + dy * dy > r2) { arr[rowBase + x] = 0; continue }
                val id = wyBase + x
                val o = owner[id].toInt()
                arr[rowBase + x] = if (o != 0) teamRgba[players[o - 1].colorIdx % teamRgba.size]
                else { val tr = trail[id].toInt(); if (tr != 0) trailRgba[players[tr - 1].colorIdx % trailRgba.size] else arenaRgba }
            }
        }
        val ib = mmPix.pixels.asIntBuffer(); ib.position(0); ib.put(arr)
        mmPix.pixels.position(0)
        mmTex.draw(mmPix, 0, 0)
    }

    private fun pct(tenths: Int): String = "${tenths / 10}.${tenths % 10}%"

    fun dispose() { mmTex.dispose(); mmPix.dispose() }
}
