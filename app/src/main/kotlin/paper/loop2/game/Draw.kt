package paper.loop2.game

import com.badlogic.gdx.graphics.glutils.ShapeRenderer

/** Small ShapeRenderer drawing helpers (used in both world and UI space). */
object Draw {
    /** Filled rounded rectangle. [sr] must already be in Filled mode. */
    fun roundRect(sr: ShapeRenderer, x: Float, y: Float, w: Float, h: Float, radius: Float) {
        val r = minOf(radius, w / 2f, h / 2f)
        if (r <= 0f) { sr.rect(x, y, w, h); return }
        sr.rect(x + r, y, w - 2 * r, h)
        sr.rect(x, y + r, r, h - 2 * r)
        sr.rect(x + w - r, y + r, r, h - 2 * r)
        sr.arc(x + r, y + r, r, 180f, 90f)
        sr.arc(x + w - r, y + r, r, 270f, 90f)
        sr.arc(x + w - r, y + h - r, r, 0f, 90f)
        sr.arc(x + r, y + h - r, r, 90f, 90f)
    }
}
