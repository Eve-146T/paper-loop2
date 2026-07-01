package paper.loop2.core

import com.badlogic.gdx.graphics.Color

/**
 * Paper-game palette with two themes. **Light** is the warm cream "paper" look; **dark**
 * recolours only the chrome (field / wall / text) to dark gray — the vivid team colours stay
 * the same so territories read identically in both modes.
 *
 * The chrome colours are mutable [Color] instances reused across the app: [setDark] mutates them
 * in place, so every `Palette.ARENA` reference picks up a theme switch on the next frame without
 * anyone re-fetching. Don't mutate these from elsewhere — copy via `tmp.set(Palette.X)` first.
 */
object Palette {
    // Arena / chrome — swapped by setDark(). Start in light so design-time reads are sane.
    val VOID = Color()      // outside the arena (window background)
    val ARENA = Color()     // the playfield
    val WALL = Color()      // arena border wall
    val INK = Color()       // primary text / outlines
    val MUTED = Color()     // secondary text
    val PANEL = Color()     // translucent panel over the field
    val WHITE: Color = Color.WHITE

    var dark = false
        private set

    /** Vivid team colours. Index 0 is the human (neon green). Theme-independent. */
    val TEAM: Array<Color> = arrayOf(
        rgb(0x35C44B), // 0 green   (human)
        rgb(0x2BA9FF), // 1 sky
        rgb(0xFF3B6B), // 2 raspberry
        rgb(0xFFB400), // 3 amber
        rgb(0xC44BFF), // 4 purple
        rgb(0x14D38A), // 5 mint
        rgb(0xFF7A1A), // 6 orange
        rgb(0x14C9C2), // 7 teal
        rgb(0xFF4BD8), // 8 magenta
        rgb(0x6A6BFF), // 9 violet
        rgb(0x8FD12A), // 10 lime
        rgb(0xFF5C5C), // 11 coral
    )

    fun team(i: Int): Color = TEAM[((i % TEAM.size) + TEAM.size) % TEAM.size]

    /** Switch the chrome palette in place. Call once at startup and whenever the toggle flips. */
    fun setDark(d: Boolean) {
        dark = d
        if (d) {
            set(VOID, 0x121317)   // near-black surround
            set(ARENA, 0x26282E)  // dark gray playfield
            set(WALL, 0x6B7280)   // border: clearly lighter than the field so the edge reads
            set(INK, 0xECEEF3)    // light text
            MUTED.set(0.60f, 0.62f, 0.70f, 1f)
            PANEL.set(1f, 1f, 1f, 0.10f)
        } else {
            set(VOID, 0xD7D1BB)
            set(ARENA, 0xF1EDDC)
            set(WALL, 0x6E6450)   // border: a clear dark frame against the cream field
            set(INK, 0x33333B)
            MUTED.set(0.40f, 0.40f, 0.45f, 1f)
            PANEL.set(0f, 0f, 0f, 0.10f)
        }
    }

    private fun set(c: Color, v: Int) {
        c.set(((v ushr 16) and 0xFF) / 255f, ((v ushr 8) and 0xFF) / 255f, (v and 0xFF) / 255f, 1f)
    }

    private fun rgb(v: Int): Color {
        val r = ((v ushr 16) and 0xFF) / 255f
        val g = ((v ushr 8) and 0xFF) / 255f
        val b = (v and 0xFF) / 255f
        return Color(r, g, b, 1f)
    }

    fun darker(c: Color, f: Float = 0.72f): Color = Color(c.r * f, c.g * f, c.b * f, c.a)

    init { setDark(false) }
}
