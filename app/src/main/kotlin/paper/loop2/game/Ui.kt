package paper.loop2.game

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.OrthographicCamera
import com.badlogic.gdx.graphics.g2d.BitmapFont
import com.badlogic.gdx.graphics.g2d.GlyphLayout
import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.badlogic.gdx.graphics.glutils.ShapeRenderer
import paper.loop2.core.Assets

/** Shared UI rendering context (pixel-space camera, batch, shapes, fonts) used by every screen. */
class Ui(val assets: Assets) {
    val batch = SpriteBatch()
    val shapes = ShapeRenderer()
    val cam = OrthographicCamera()
    private val layout = GlyphLayout()
    var w = 0f; private set
    var h = 0f; private set

    fun resize(width: Int, height: Int) {
        w = width.toFloat(); h = height.toFloat()
        cam.setToOrtho(false, w, h)
        batch.projectionMatrix = cam.combined
        shapes.projectionMatrix = cam.combined
    }

    fun textWidth(font: BitmapFont, s: String): Float { layout.setText(font, s); return layout.width }

    /** Draw left-aligned text; (x, yTop) is the top-left of the first line. Batch must be active. */
    fun text(font: BitmapFont, s: String, x: Float, yTop: Float, color: Color) {
        font.color = color
        font.draw(batch, s, x, yTop)
    }

    fun textCentered(font: BitmapFont, s: String, cx: Float, yTop: Float, color: Color) {
        layout.setText(font, s)
        font.color = color
        font.draw(batch, s, cx - layout.width / 2f, yTop)
    }

    /**
     * Draw text centred both horizontally on [cx] and vertically on [cyCenter], using the laid-out
     * glyph height so it sits truly centred inside a button/box (not eyeballed with a magic offset).
     */
    fun textCenteredIn(font: BitmapFont, s: String, cx: Float, cyCenter: Float, color: Color) {
        layout.setText(font, s)
        font.color = color
        font.draw(batch, s, cx - layout.width / 2f, cyCenter + layout.height / 2f)
    }

    /** Draw text centred horizontally on [cx] with its BOTTOM edge at [yBottom] (y-up) — for sitting
     *  a label just above a point regardless of font/screen size. */
    fun textCenteredAbove(font: BitmapFont, s: String, cx: Float, yBottom: Float, color: Color) {
        layout.setText(font, s)
        font.color = color
        font.draw(batch, s, cx - layout.width / 2f, yBottom + layout.height)
    }

    fun textRight(font: BitmapFont, s: String, xRight: Float, yTop: Float, color: Color) {
        layout.setText(font, s)
        font.color = color
        font.draw(batch, s, xRight - layout.width, yTop)
    }

    /** Filled rounded rect via the shared shapes renderer (which must be in Filled mode). */
    fun roundRect(x: Float, y: Float, w: Float, h: Float, r: Float, color: Color) {
        shapes.color = color
        Draw.roundRect(shapes, x, y, w, h, r)
    }

    fun dispose() { batch.dispose(); shapes.dispose() }
}
