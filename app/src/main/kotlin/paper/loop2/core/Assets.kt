package paper.loop2.core

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.Pixmap
import com.badlogic.gdx.graphics.Texture
import com.badlogic.gdx.graphics.g2d.BitmapFont
import com.badlogic.gdx.graphics.g2d.PixmapPacker
import com.badlogic.gdx.graphics.g2d.TextureRegion
import com.badlogic.gdx.graphics.g2d.freetype.FreeTypeFontGenerator
import com.badlogic.gdx.graphics.g2d.freetype.FreeTypeFontGenerator.FreeTypeBitmapFontData
import com.badlogic.gdx.graphics.g2d.freetype.FreeTypeFontGenerator.FreeTypeFontParameter
import com.badlogic.gdx.utils.Disposable
import com.badlogic.gdx.utils.Array as GdxArray

/** Shared fonts (rasterised from bundled TTFs via FreeType) and a white pixel texture. */
class Assets : Disposable {
    lateinit var huge: BitmapFont   // title / big percentage
    lateinit var big: BitmapFont    // headers, buttons
    lateinit var med: BitmapFont    // labels
    lateinit var small: BitmapFont  // leaderboard, hints
    lateinit var white: Texture

    /** True once the fonts are ready. The world + HUD shapes render before this; text waits for it. */
    @Volatile
    var loaded = false
        private set

    // The shared atlas the glyphs are packed into. Kept alive for the app's lifetime because the
    // fonts' textures live in its pages (disposed in [dispose]).
    private var packer: PixmapPacker? = null

    /**
     * Build the fonts **without blocking the render thread**. The white pixel is created immediately
     * (trivial GL), but FreeType glyph rasterisation — the dominant cold-start cost — is pure CPU
     * work, so it runs on a background thread that packs every glyph into CPU pixmaps; only the cheap
     * GL texture upload is posted back to the render thread. While [loaded] is false the world and the
     * HUD/menu *shapes* still draw and input is processed, so the player can tap to begin right away
     * instead of staring at a frozen screen. Falls back to a synchronous build if anything throws.
     */
    fun beginAsyncLoad() {
        white = makeWhite()
        val h = baseHeight()
        Thread({
            try {
                val pk = PixmapPacker(1024, 1024, Pixmap.Format.RGBA8888, 2, false)
                val data = rasterize(pk, h)                  // CPU only — packer supplied, no GL
                Gdx.app.postRunnable {                       // render thread: textures + fonts
                    runCatching { upload(pk, data) }.onFailure { runCatching { load() } }
                }
            } catch (t: Throwable) {
                Gdx.app.postRunnable { runCatching { load() } }
            }
        }, "font-init").apply { isDaemon = true; start() }
    }

    /** Synchronous build on the render thread. Used as the [beginAsyncLoad] fallback. */
    fun load() {
        if (!this::white.isInitialized) white = makeWhite()
        val pk = PixmapPacker(1024, 1024, Pixmap.Format.RGBA8888, 2, false)
        upload(pk, rasterize(pk, baseHeight()))
    }

    private fun baseHeight(): Float = Gdx.graphics.height.coerceAtLeast(480).toFloat()

    private fun makeWhite(): Texture {
        val pm = Pixmap(4, 4, Pixmap.Format.RGBA8888)
        pm.setColor(Color.WHITE); pm.fill()
        val tex = Texture(pm)
        tex.setFilter(Texture.TextureFilter.Linear, Texture.TextureFilter.Linear)
        pm.dispose()
        return tex
    }

    /** Rasterise all four fonts into [pk]'s CPU pages (safe off the GL thread). */
    private fun rasterize(pk: PixmapPacker, h: Float): List<FreeTypeBitmapFontData> {
        val bold = FreeTypeFontGenerator(Gdx.files.internal("font-bold.ttf"))
        // huge/big only ever draw the title, the % readout and the all-caps buttons → uppercase set.
        val dHuge = data(bold, pk, (h * 0.066f).coerceIn(36f, 96f), chars = CHARS_UPPER)
        val dBig = data(bold, pk, (h * 0.046f).coerceIn(26f, 62f), chars = CHARS_UPPER)
        val dMed = data(bold, pk, (h * 0.030f).coerceIn(20f, 44f), chars = CHARS_FULL)
        bold.dispose()
        val regular = FreeTypeFontGenerator(Gdx.files.internal("font-regular.ttf"))
        val dSmall = data(regular, pk, (h * 0.022f).coerceIn(15f, 32f), chars = CHARS_FULL)
        regular.dispose()
        return listOf(dHuge, dBig, dMed, dSmall)
    }

    /** Upload the packed pages to GL and wire up the BitmapFonts. Must run on the render thread. */
    private fun upload(pk: PixmapPacker, data: List<FreeTypeBitmapFontData>) {
        val regions = GdxArray<TextureRegion>()
        pk.updateTextureRegions(regions, Texture.TextureFilter.Linear, Texture.TextureFilter.Linear, false)
        huge = BitmapFont(data[0], regions, true)
        big = BitmapFont(data[1], regions, true)
        med = BitmapFont(data[2], regions, true)
        small = BitmapFont(data[3], regions, true)
        packer = pk
        loaded = true
    }

    private fun data(gen: FreeTypeFontGenerator, pk: PixmapPacker, size: Float, chars: String): FreeTypeBitmapFontData {
        val p = FreeTypeFontParameter().apply {
            this.size = size.toInt()
            color = Color.WHITE
            // Rasterise only the glyphs the game actually draws — the FreeType default set (~200
            // glyphs incl. accents) is the dominant cold-start cost. Keep these in sync with the UI.
            characters = chars
            // No mipmaps: text is drawn at ~1:1, so the mipmap pyramid is wasted work. Linear keeps
            // glyphs crisp. The glyphs pack into the shared CPU atlas (no GL on this thread).
            // (No FreeType drop-shadow: it is pathologically slow at large sizes — ~1.3s for the 96px
            // face alone — and the flat text reads cleanly on the scrim/cards anyway.)
            genMipMaps = false
            minFilter = Texture.TextureFilter.Linear
            magFilter = Texture.TextureFilter.Linear
            this.packer = pk
        }
        return gen.generateData(p)
    }

    override fun dispose() {
        runCatching { huge.dispose() }
        runCatching { big.dispose() }
        runCatching { med.dispose() }
        runCatching { small.dispose() }
        runCatching { white.dispose() }
        runCatching { packer?.dispose() }   // owns the font page textures
    }

    private companion object {
        // The exact glyphs the UI draws. huge/big are all-caps (title, %, buttons); med/small add
        // lower-case (menu hints, bot names) and the few punctuation marks (incl. the · separator).
        const val CHARS_UPPER = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.%#/"
        const val CHARS_FULL =
            " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:!?'%#/+-·()"
    }
}
