package paper.loop2.game

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.OrthographicCamera
import com.badlogic.gdx.graphics.Pixmap
import com.badlogic.gdx.graphics.Texture
import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.badlogic.gdx.graphics.glutils.ShaderProgram
import com.badlogic.gdx.graphics.glutils.ShapeRenderer
import paper.loop2.core.Palette
import kotlin.math.cos
import kotlin.math.sin

/**
 * Free-movement Paper.io-2 look. The **map is a circle**; territories are genuinely **smooth blobs**.
 *
 * ## Smooth territory, smooth boundaries
 * Each player's territory is rendered from its **own** coverage texture (one texel per cell: alpha =
 * "this player owns it"). Per rebuild we box-blur that coverage so the rasterised cog edge reads as a
 * smooth disc, and a shader does `smoothstep` on the **bilinear** coverage (its 0.5 iso-line is a
 * smooth curve, snapped crisply). Because every territory has its own anti-aliased edge, two turfs
 * meeting blend smoothly instead of showing a jagged per-cell colour boundary. Textures rebuild only
 * for players whose area changed (cheap: idle = nothing, a capture = 1-2 players, a dissolve = 1/frame).
 * Each is drawn twice — a darker downward-shifted copy (the 3D bottom rim), then the top face.
 *
 * Trails (lighter-than-turf ribbon from each polyline) and heads are drawn on top with the ShapeRenderer.
 */
class WorldRenderer {
    val cam = OrthographicCamera()
    private val sr = ShapeRenderer()
    private val batch = SpriteBatch()
    private val shader = buildShader()

    private var camX = ARENA_CX
    private var camY = ARENA_CY
    private var viewCells = 26f
    private val tmp = Color()
    private val tmp2 = Color()
    private val headBrighten = 0.30f

    // per-player coverage textures (alpha = coverage; colour comes from the batch vertex colour)
    private val buildPix = Pixmap(GW, GH, Pixmap.Format.RGBA8888)
    private val pixArr = IntArray(GW * GH)
    private val cov = FloatArray(GW * GH)
    private val covT = FloatArray(GW * GH)
    private var texP: Array<Texture?> = arrayOfNulls(0)
    private var lastArea = IntArray(0)
    // last uploaded sub-rect per player (texels outside it are stale); used to detect a region
    // *shrink* (respawn / new game) and wipe the texture so old turf doesn't linger. Empty = x1 < x0.
    private var lastUx0 = IntArray(0); private var lastUy0 = IntArray(0)
    private var lastUx1 = IntArray(0); private var lastUy1 = IntArray(0)
    private var lastResetGen = -1
    var lastRebuilds = 0; private set    // coverage-texture rebuilds in the most recent render() (profiler)

    private val CULL_MARGIN = 5f   // view-rect padding (cells) so a turf's AA fringe is never culled early
    private val blurR = 3          // coverage box-blur radius (rounds the cog into a disc)
    private val edge = 0.0009f     // smoothstep half-width in texcoords (crisp but smooth)
    private val rimDepth = 0.7f    // darker slab side peek-below, cells
    private val rimDark = 0.66f    // bottom-rim darkening

    init { buildPix.blending = Pixmap.Blending.None }

    fun resize(w: Int, h: Int) {
        cam.viewportWidth = viewCells
        cam.viewportHeight = viewCells * h / w
        cam.update()
    }

    fun snap(world: World) { camX = world.human.x; camY = world.human.y }
    fun follow(world: World) { camX = world.human.x; camY = world.human.y }

    fun render(world: World) {
        cam.position.set(camX, camY, 0f)
        cam.update()
        Gdx.gl.glEnable(GL20.GL_BLEND)
        Gdx.gl.glBlendFunc(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA)

        // 1) circular field + wall ring
        sr.projectionMatrix = cam.combined
        sr.begin(ShapeRenderer.ShapeType.Filled)
        val seg = 90
        sr.setColor(Palette.WALL); sr.circle(ARENA_CX, ARENA_CY, ARENA_R + 1.8f, seg)
        sr.setColor(Palette.ARENA); sr.circle(ARENA_CX, ARENA_CY, ARENA_R, seg)
        sr.end()

        // camera view rect (world cells, +margin for the AA fringe) — used both to cull off-screen
        // turfs/trails/heads AND to skip rebuilding turfs that aren't on screen
        val vx0 = camX - cam.viewportWidth * 0.5f - CULL_MARGIN; val vx1 = camX + cam.viewportWidth * 0.5f + CULL_MARGIN
        val vy0 = camY - cam.viewportHeight * 0.5f - CULL_MARGIN; val vy1 = camY + cam.viewportHeight * 0.5f + CULL_MARGIN

        // 2) per-player smooth territories — rebuild only turfs that changed AND are on screen. An
        //    off-screen turf is left dirty (lastArea stale) and rebuilds the frame it scrolls into
        //    view, so a flurry of off-screen captures costs one rebuild on entry, not one each.
        ensureTextures(world.players.size)
        if (world.resetGen != lastResetGen) { lastResetGen = world.resetGen; lastArea.fill(-1) }
        lastRebuilds = 0
        for (p in world.players) if (p.area != lastArea[p.id] || texP[p.id] == null) {
            if (turfVisible(p, vx0, vx1, vy0, vy1)) {
                rebuildPlayer(world, p); lastArea[p.id] = p.area; lastRebuilds++
            }
        }
        batch.projectionMatrix = cam.combined
        batch.shader = shader
        batch.begin()
        shader.setUniformf("u_edge", edge)
        // all rims first, so a player's rim never darkens over another's top face
        for (p in world.players) if (turfVisible(p, vx0, vx1, vy0, vy1)) {
            val c = Palette.team(p.colorIdx)
            batch.setColor(c.r * rimDark, c.g * rimDark, c.b * rimDark, 1f)
            batch.draw(texP[p.id], 0f, -rimDepth, GW.toFloat(), GH.toFloat())
        }
        // tops: bots first, then the human last (cleanest edges on your own turf)
        for (p in world.players) if (!p.isHuman && turfVisible(p, vx0, vx1, vy0, vy1)) { batch.color = Palette.team(p.colorIdx); batch.draw(texP[p.id], 0f, 0f, GW.toFloat(), GH.toFloat()) }
        world.human.let { if (it.area > 0) { batch.color = Palette.team(it.colorIdx); batch.draw(texP[it.id], 0f, 0f, GW.toFloat(), GH.toFloat()) } }
        batch.end()
        batch.shader = null

        // 3+4) YOUR trail on top of all turfs, then your turf redrawn over it so its root tucks UNDER
        //    your own edge. Only needed when you actually have a ribbon (alive+trailing, or the dying
        //    retract) — skipped entirely (a ShapeRenderer pass + a full-grid turf redraw) otherwise.
        val h = world.human
        if ((h.alive && h.hasTrail) || (!h.alive && h.dyingFrac > 0f && h.trailX.size >= 2)) {
            sr.projectionMatrix = cam.combined
            sr.begin(ShapeRenderer.ShapeType.Filled)
            drawTrailIfAny(h)
            sr.end()
            if (h.area > 0) {
                batch.shader = shader
                batch.begin()
                shader.setUniformf("u_edge", edge)
                batch.color = Palette.team(h.colorIdx); batch.draw(texP[h.id], 0f, 0f, GW.toFloat(), GH.toFloat())
                batch.end()
                batch.shader = null
            }
        }

        // 5) bot trails (above all turfs), pending-respawn telegraphs, then every head on top.
        //    SpriteBatch.end() turns GL blending back OFF, so re-enable it here or the telegraph's
        //    translucent disc would composite as solid colour.
        Gdx.gl.glEnable(GL20.GL_BLEND)
        Gdx.gl.glBlendFunc(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA)
        sr.begin(ShapeRenderer.ShapeType.Filled)
        for (p in world.players) if (!p.isHuman && trailVisible(p, vx0, vx1, vy0, vy1)) drawTrailIfAny(p)
        for (p in world.players) if (p.spawnWarn > 0f && spawnVisible(p, vx0, vx1, vy0, vy1)) drawSpawnMarker(p)
        for (p in world.players) if (p.alive && headVisible(p, vx0, vx1, vy0, vy1)) drawHead(p)
        sr.end()
    }

    /** Telegraph for a pending bot respawn: a translucent disc in the bot's colour with
     *  a ring closing in on it, so a spawn can be seen coming and steered around (issue #2). Animated
     *  off the countdown itself, so the renderer needs no clock of its own. */
    private fun drawSpawnMarker(p: Player) {
        val t = (1f - p.spawnWarn / SPAWN_WARN).coerceIn(0f, 1f)   // 0 -> 1 as the spawn approaches
        val c = Palette.team(p.colorIdx)
        val cx = p.spawnCx + 0.5f; val cy = p.spawnCy + 0.5f
        sr.setColor(c.r, c.g, c.b, 0.07f + 0.13f * t)
        sr.circle(cx, cy, START_R, 48)
        sr.setColor(c.r, c.g, c.b, 0.25f + 0.45f * t)
        ring(cx, cy, START_R, 0.45f)
        sr.setColor(c.r, c.g, c.b, 0.35f * (1f - t))              // an outer ring closing in on it
        ring(cx, cy, START_R * (2f - t), 0.35f)
    }

    /** A circle outline in world units (ShapeRenderer's Line mode can't do a world-space thickness). */
    private fun ring(cx: Float, cy: Float, r: Float, thick: Float) {
        val seg = 48
        var px = cx + r; var py = cy
        for (i in 1..seg) {
            val a = i * TWO_PI / seg
            val nx = cx + cos(a) * r; val ny = cy + sin(a) * r
            sr.rectLine(px, py, nx, ny, thick)
            px = nx; py = ny
        }
    }

    private fun drawTrailIfAny(p: Player) {
        if (p.alive && p.hasTrail) drawRibbon(p, false)
        else if (!p.alive && p.dyingFrac > 0f && p.trailX.size >= 2) drawRibbon(p, true)
    }

    /** Does player [p]'s turf (owned-cell bbox, +rim/blur fringe) intersect the camera view rect? */
    private fun turfVisible(p: Player, vx0: Float, vx1: Float, vy0: Float, vy1: Float): Boolean {
        if (p.area <= 0 || p.bbMaxX < p.bbMinX) return false
        return !((p.bbMaxX + 1) < vx0 || p.bbMinX > vx1 || (p.bbMaxY + 1) < vy0 || (p.bbMinY - rimDepth) > vy1)
    }

    /** Does player [p]'s trail bbox (+ribbon half-width) intersect the camera view rect? */
    private fun trailVisible(p: Player, vx0: Float, vx1: Float, vy0: Float, vy1: Float): Boolean {
        if (!p.hasTrail && !(p.dyingFrac > 0f && p.trailX.size >= 2)) return false
        val m = TRAIL_R + 1f
        return !(p.trMaxX + m < vx0 || p.trMinX - m > vx1 || p.trMaxY + m < vy0 || p.trMinY - m > vy1)
    }

    /** Is player [p]'s pending spawn telegraph (its widest ring) within the camera view rect? */
    private fun spawnVisible(p: Player, vx0: Float, vx1: Float, vy0: Float, vy1: Float): Boolean {
        val m = START_R * 2f
        val cx = p.spawnCx + 0.5f; val cy = p.spawnCy + 0.5f
        return !(cx + m < vx0 || cx - m > vx1 || cy + m < vy0 || cy - m > vy1)
    }

    /** Is player [p]'s head within the camera view rect (+ its half-extent)? */
    private fun headVisible(p: Player, vx0: Float, vx1: Float, vy0: Float, vy1: Float): Boolean {
        val m = HEAD_R + 0.5f
        return !(p.x + m < vx0 || p.x - m > vx1 || p.y + m < vy0 || p.y - m > vy1)
    }

    // --- per-player coverage texture ---------------------------------------

    private fun ensureTextures(count: Int) {
        if (texP.size == count) return
        texP.forEach { it?.dispose() }
        texP = arrayOfNulls(count)
        lastArea = IntArray(count) { -1 }
        lastUx0 = IntArray(count); lastUy0 = IntArray(count)
        lastUx1 = IntArray(count) { -1 }; lastUy1 = IntArray(count) { -1 }
    }

    /**
     * Rebuild a player's coverage texture, touching **only the player's owned bbox + blur fringe**
     * instead of the whole 200x200 grid (the old cost: 5 full-grid passes + a 160 KB upload, ~12 ms on
     * a weak core). We box-blur a windowed region and upload just that sub-rect via glTexSubImage2D.
     * The owned bbox is expand-only, so the uploaded region never shrinks between rebuilds *except* on
     * respawn / new game (bbox reset) — detected here as a "shrink" and the texture is wiped first so
     * stale turf from a previous life can't linger outside the new, smaller region.
     */
    private fun rebuildPlayer(world: World, p: Player) {
        val self = p.id + 1
        val owner = world.owner
        var t = texP[p.id]
        if (t == null) {
            t = Texture(GW, GH, Pixmap.Format.RGBA8888).also { it.setFilter(Texture.TextureFilter.Linear, Texture.TextureFilter.Linear) }
            texP[p.id] = t
            clearTexture(t)                                  // texels outside the dirty rect must start at 0
            lastUx1[p.id] = -1                               // no prior region
        }
        if (p.area <= 0 || p.bbMaxX < p.bbMinX) {            // emptied (fully dissolved): wipe any prior turf
            if (lastUx1[p.id] >= lastUx0[p.id]) { clearTexture(t!!); lastUx1[p.id] = -1 }
            return
        }
        val r = blurR
        // upload region U = territory bbox + blur fringe; coverage scratch C = U + r (the blur's input halo)
        val ux0 = (p.bbMinX - r).coerceAtLeast(0); val ux1 = (p.bbMaxX + r).coerceAtMost(GW - 1)
        val uy0 = (p.bbMinY - r).coerceAtLeast(0); val uy1 = (p.bbMaxY + r).coerceAtMost(GH - 1)
        val cx0 = (ux0 - r).coerceAtLeast(0); val cx1 = (ux1 + r).coerceAtMost(GW - 1)
        val cy0 = (uy0 - r).coerceAtLeast(0); val cy1 = (uy1 + r).coerceAtMost(GH - 1)
        // if the region shrank vs last upload, old texels poke outside the new region -> wipe first
        if (lastUx1[p.id] >= lastUx0[p.id] &&
            (lastUx0[p.id] < ux0 || lastUy0[p.id] < uy0 || lastUx1[p.id] > ux1 || lastUy1[p.id] > uy1)) {
            clearTexture(t!!)
        }
        // raw coverage over C, then a windowed separable box-blur into the upload region U
        for (y in cy0..cy1) { val b = y * GW; for (x in cx0..cx1) cov[b + x] = if (owner[b + x].toInt() == self) 1f else 0f }
        blurHWin(cov, covT, r, ux0, ux1, cy0, cy1, cx0, cx1)
        blurVWin(covT, cov, r, ux0, ux1, uy0, uy1, cy0, cy1)
        // pack U (top world-row first, matching the texture's y-flip) and upload just the sub-rect
        val rw = ux1 - ux0 + 1; val rh = uy1 - uy0 + 1
        val ib = buildPix.pixels.asIntBuffer(); ib.position(0)
        for (ty in 0 until rh) {
            val wy = uy1 - ty; val b = wy * GW
            for (xx in 0 until rw) {
                val a = (cov[b + ux0 + xx] * 255f).toInt().coerceIn(0, 255)
                ib.put((0xFFFFFF00.toInt()) or a)
            }
        }
        buildPix.pixels.position(0)
        t!!.bind()
        Gdx.gl.glTexSubImage2D(GL20.GL_TEXTURE_2D, 0, ux0, GH - 1 - uy1, rw, rh, GL20.GL_RGBA, GL20.GL_UNSIGNED_BYTE, buildPix.pixels)
        lastUx0[p.id] = ux0; lastUy0[p.id] = uy0; lastUx1[p.id] = ux1; lastUy1[p.id] = uy1
    }

    /** Zero the whole texture (used on create and on a region shrink — both rare). */
    private fun clearTexture(t: Texture) {
        java.util.Arrays.fill(pixArr, 0)
        val ib = buildPix.pixels.asIntBuffer(); ib.position(0); ib.put(pixArr)
        buildPix.pixels.position(0)
        t.draw(buildPix, 0, 0)
    }

    // Windowed separable box-blur. Output over cols [ox0..ox1] x rows [ry0..ry1]; reads clamped to the
    // coverage scratch window [cLo..cHi] (anything outside is genuinely 0 — no territory there).
    private fun blurHWin(src: FloatArray, dst: FloatArray, r: Int, ox0: Int, ox1: Int, ry0: Int, ry1: Int, cLo: Int, cHi: Int) {
        val inv = 1f / (2 * r + 1)
        for (y in ry0..ry1) {
            val b = y * GW
            var sum = 0f
            var k = ox0 - r
            while (k <= ox0 + r) { if (k in cLo..cHi) sum += src[b + k]; k++ }
            var x = ox0
            while (x <= ox1) {
                dst[b + x] = sum * inv
                val addI = x + r + 1; if (addI <= cHi) sum += src[b + addI]
                val remI = x - r; if (remI >= cLo) sum -= src[b + remI]
                x++
            }
        }
    }

    private fun blurVWin(src: FloatArray, dst: FloatArray, r: Int, ox0: Int, ox1: Int, ry0: Int, ry1: Int, cLo: Int, cHi: Int) {
        val inv = 1f / (2 * r + 1)
        for (x in ox0..ox1) {
            var sum = 0f
            var k = ry0 - r
            while (k <= ry0 + r) { if (k in cLo..cHi) sum += src[k * GW + x]; k++ }
            var y = ry0
            while (y <= ry1) {
                dst[y * GW + x] = sum * inv
                val addY = y + r + 1; if (addY <= cHi) sum += src[addY * GW + x]
                val remY = y - r; if (remY >= cLo) sum -= src[remY * GW + x]
                y++
            }
        }
    }

    // --- trail ribbon (miter strip), lighter than the turf -----------------

    private var offLX = FloatArray(64); private var offLY = FloatArray(64)
    private var offRX = FloatArray(64); private var offRY = FloatArray(64)

    private fun drawRibbon(p: Player, dying: Boolean) {
        val n = p.trailX.size
        if (n == 0) return
        // alive: render the whole polyline + the live head (n+1 verts). dying: render the un-retracted
        // front of the polyline (head end zips back), no live head.
        val m = if (dying) (p.dyingFrac * n).toInt().coerceIn(2, n) else n + 1
        if (m < 2) return
        ensureOff(m)
        val hw = TRAIL_R * 0.92f
        var i = 0
        while (i < m) {
            val cx = if (i < n) p.trailX[i] else p.x
            val cy = if (i < n) p.trailY[i] else p.y
            val pxv = if (i > 0) (if (i - 1 < n) p.trailX[i - 1] else p.x) else cx
            val pyv = if (i > 0) (if (i - 1 < n) p.trailY[i - 1] else p.y) else cy
            val nxv = if (i < m - 1) (if (i + 1 < n) p.trailX[i + 1] else p.x) else cx
            val nyv = if (i < m - 1) (if (i + 1 < n) p.trailY[i + 1] else p.y) else cy
            var inx = cx - pxv; var iny = cy - pyv
            var oux = nxv - cx; var ouy = nyv - cy
            val li = Math.sqrt((inx * inx + iny * iny).toDouble()).toFloat()
            if (li > 1e-4f) { inx /= li; iny /= li } else { inx = oux; iny = ouy }
            val lo = Math.sqrt((oux * oux + ouy * ouy).toDouble()).toFloat()
            if (lo > 1e-4f) { oux /= lo; ouy /= lo } else { oux = inx; ouy = iny }
            var ax = inx + oux; var ay = iny + ouy
            var la = Math.sqrt((ax * ax + ay * ay).toDouble()).toFloat()
            if (la < 1e-4f) { ax = -iny; ay = inx; la = 1f }
            ax /= la; ay /= la
            val mnx = -ay; val mny = ax
            var d = mnx * (-ouy) + mny * oux
            if (d < 0.35f) d = 0.35f
            val len = (hw / d).coerceAtMost(hw * 2.5f)
            offLX[i] = cx + mnx * len; offLY[i] = cy + mny * len
            offRX[i] = cx - mnx * len; offRY[i] = cy - mny * len
            i++
        }
        // lighter than the solid turf so the trail reads as a distinct ribbon, not territory
        tmp.set(Palette.team(p.colorIdx)).lerp(Color.WHITE, 0.42f); tmp.a = 0.85f; sr.color = tmp
        i = 0
        while (i < m - 1) {
            sr.triangle(offLX[i], offLY[i], offRX[i], offRY[i], offRX[i + 1], offRY[i + 1])
            sr.triangle(offLX[i], offLY[i], offRX[i + 1], offRY[i + 1], offLX[i + 1], offLY[i + 1])
            i++
        }
    }

    private fun ensureOff(n: Int) {
        if (offLX.size >= n) return
        var s = offLX.size; while (s < n) s *= 2
        offLX = FloatArray(s); offLY = FloatArray(s); offRX = FloatArray(s); offRY = FloatArray(s)
    }

    // --- head + death burst -----------------------------------------------

    private fun drawHead(p: Player) {
        val base = Palette.team(p.colorIdx)
        val s = HEAD_R
        val cx = p.x; val cy = p.y
        val ca = cos(p.heading); val sa = sin(p.heading)
        fun rx(dx: Float, dy: Float) = cx + dx * ca - dy * sa
        fun ry(dx: Float, dy: Float) = cy + dx * sa + dy * ca
        // drop shadow: the same square shifted DOWN in world space (consistent at any heading)
        val sd = 0.2f
        tmp.set(base).mul(0.62f); tmp.a = 1f; sr.color = tmp
        quad(rx(-s, -s), ry(-s, -s) - sd, rx(s, -s), ry(s, -s) - sd, rx(s, s), ry(s, s) - sd, rx(-s, s), ry(-s, s) - sd)
        // bright top face
        tmp2.set(base).lerp(Color.WHITE, headBrighten); tmp2.a = 1f; sr.color = tmp2
        quad(rx(-s, -s), ry(-s, -s), rx(s, -s), ry(s, -s), rx(s, s), ry(s, s), rx(-s, s), ry(-s, s))
    }

    private fun quad(x1: Float, y1: Float, x2: Float, y2: Float, x3: Float, y3: Float, x4: Float, y4: Float) {
        sr.triangle(x1, y1, x2, y2, x3, y3)
        sr.triangle(x1, y1, x3, y3, x4, y4)
    }

    private fun buildShader(): ShaderProgram {
        val vert = """
            attribute vec4 a_position;
            attribute vec4 a_color;
            attribute vec2 a_texCoord0;
            uniform mat4 u_projTrans;
            varying vec4 v_color;
            varying vec2 v_texCoords;
            void main() { v_color = a_color; v_texCoords = a_texCoord0; gl_Position = u_projTrans * a_position; }
        """.trimIndent()
        val frag = """
            #ifdef GL_ES
            precision mediump float;
            #endif
            varying vec4 v_color;
            varying vec2 v_texCoords;
            uniform sampler2D u_texture;
            uniform float u_edge;
            void main() {
                float cov = texture2D(u_texture, v_texCoords).a;
                float a = smoothstep(0.5 - u_edge, 0.5 + u_edge, cov);
                gl_FragColor = vec4(v_color.rgb, a * v_color.a);
            }
        """.trimIndent()
        ShaderProgram.pedantic = false
        val sp = ShaderProgram(vert, frag)
        if (!sp.isCompiled) Gdx.app.log("WorldRenderer", "territory shader: ${sp.log}")
        return sp
    }

    fun dispose() {
        sr.dispose(); batch.dispose(); shader.dispose(); buildPix.dispose()
        texP.forEach { it?.dispose() }
    }
}
