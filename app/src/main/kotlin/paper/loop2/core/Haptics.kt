package paper.loop2.core

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

/** Haptic feedback, gated by [Settings.hapticsOn]. */
object Haptics {
    private var vib: Vibrator? = null

    fun init(ctx: Context) {
        vib = if (Build.VERSION.SDK_INT >= 31) {
            (ctx.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            ctx.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
    }

    fun tick() = predefined(VibrationEffect.EFFECT_TICK)
    fun click() = predefined(VibrationEffect.EFFECT_CLICK)
    fun heavy() = predefined(VibrationEffect.EFFECT_HEAVY_CLICK)

    fun success() = waveform(longArrayOf(0, 26, 70, 40), intArrayOf(0, 170, 0, 255))
    fun fail() = waveform(longArrayOf(0, 70, 60, 140), intArrayOf(0, 120, 0, 230))

    /** A punchy double hit for landing a kill. */
    fun kill() = waveform(longArrayOf(0, 16, 36, 50), intArrayOf(0, 220, 0, 130))

    private fun predefined(effect: Int) {
        if (!Settings.hapticsOn) return
        runCatching { vib?.vibrate(VibrationEffect.createPredefined(effect)) }
    }

    private fun waveform(times: LongArray, amps: IntArray) {
        if (!Settings.hapticsOn) return
        runCatching { vib?.vibrate(VibrationEffect.createWaveform(times, amps, -1)) }
    }
}
