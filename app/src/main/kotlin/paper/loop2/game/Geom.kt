package paper.loop2.game

import kotlin.math.PI

// Angle constants (kept as Float; the sim is all single-precision).
const val PI_F = PI.toFloat()
const val TWO_PI = (PI * 2.0).toFloat()
const val HALF_PI = (PI / 2.0).toFloat()

/** Wrap an angle into (-PI, PI]. */
fun wrapAngle(a: Float): Float {
    var x = a % TWO_PI
    if (x <= -PI_F) x += TWO_PI
    else if (x > PI_F) x -= TWO_PI
    return x
}

/**
 * Growable primitive-float list. The trail polyline is appended many times per second across all
 * players; a boxed `ArrayList<Float>` would churn the GC, so the hot path uses this instead.
 */
class FloatArrayList(initial: Int = 32) {
    var items = FloatArray(initial.coerceAtLeast(4)); private set
    var size = 0; private set

    fun add(v: Float) {
        if (size == items.size) items = items.copyOf(items.size * 2)
        items[size++] = v
    }
    operator fun get(i: Int): Float = items[i]
    operator fun set(i: Int, v: Float) { items[i] = v }
    val last: Float get() = items[size - 1]
    fun clear() { size = 0 }
    fun isEmpty(): Boolean = size == 0
    fun isNotEmpty(): Boolean = size != 0
}

/** Squared distance from point P to segment AB — used for trail self-crossing tests. */
fun distSqPointSeg(px: Float, py: Float, ax: Float, ay: Float, bx: Float, by: Float): Float {
    val abx = bx - ax; val aby = by - ay
    val apx = px - ax; val apy = py - ay
    val len2 = abx * abx + aby * aby
    val t = if (len2 <= 1e-6f) 0f else ((apx * abx + apy * aby) / len2).coerceIn(0f, 1f)
    val cx = ax + abx * t; val cy = ay + aby * t
    val dx = px - cx; val dy = py - cy
    return dx * dx + dy * dy
}
