package paper.loop2.core

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.audio.Sound
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin
import kotlin.random.Random

/**
 * Tiny procedural sound bank. All effects are synthesized as 16-bit PCM at startup,
 * written to the app cache as .wav, and played via libGDX [Sound]. Fully defensive:
 * any failure leaves it silent rather than crashing. Must be init()'d on the GL thread.
 */
object SoundFx {
    private const val RATE = 22050
    private val sounds = HashMap<String, Sound>()
    // init() runs on a background thread (off the startup path); play()/the GL thread reads this.
    @Volatile private var ready = false

    fun init() {
        if (ready) return
        runCatching {
            register("capture", arp(floatArrayOf(523f, 659f, 784f, 1047f), 0.055f, 0.32f))
            register("place", blip(880f, 0.05f, 0.24f))
            register("die", fall(0.55f))
            register("eliminate", pop(0.15f))
            register("tap", blip(680f, 0.04f, 0.22f))
            register("start", arp(floatArrayOf(392f, 523f, 659f, 880f), 0.06f, 0.30f))
            register("ripple", ripple(0.42f))
            ready = true
        }
    }

    fun play(name: String, volume: Float = 1f) {
        if (!ready || !Settings.soundOn) return
        runCatching { sounds[name]?.play(volume.coerceIn(0f, 1f)) }
    }

    fun dispose() {
        sounds.values.forEach { runCatching { it.dispose() } }
        sounds.clear()
        ready = false
    }

    // --- synthesis ---------------------------------------------------------

    private fun register(name: String, samples: FloatArray) {
        val fh = Gdx.files.local("sfx/$name.wav")
        fh.writeBytes(wav(samples), false)
        sounds[name] = Gdx.audio.newSound(fh)
    }

    private fun blip(freq: Float, dur: Float, amp: Float): FloatArray {
        val n = (RATE * dur).toInt().coerceAtLeast(1)
        val out = FloatArray(n)
        val w = (2.0 * PI * freq).toFloat()
        for (i in 0 until n) {
            val t = i / RATE.toFloat()
            out[i] = amp * env(i, n) * sin(w * t)
        }
        return out
    }

    private fun arp(freqs: FloatArray, noteDur: Float, amp: Float): FloatArray {
        val per = (RATE * noteDur).toInt().coerceAtLeast(1)
        val out = FloatArray(per * freqs.size)
        for ((k, f) in freqs.withIndex()) {
            val w = (2.0 * PI * f).toFloat()
            val base = k * per
            for (i in 0 until per) {
                val t = i / RATE.toFloat()
                out[base + i] = amp * env(i, per) * sin(w * t)
            }
        }
        return out
    }

    private fun fall(dur: Float): FloatArray {
        val n = (RATE * dur).toInt().coerceAtLeast(1)
        val out = FloatArray(n)
        var phase = 0.0
        for (i in 0 until n) {
            val p = i / n.toFloat()
            val freq = 520f - 430f * p
            phase += 2.0 * PI * freq / RATE
            val noise = (Random.nextFloat() * 2f - 1f) * 0.12f * (1f - p)
            out[i] = 0.32f * (1f - p) * (sin(phase).toFloat() + noise)
        }
        return out
    }

    /** Soft, watery low "wub" — the dissolve ripple washing over you. */
    private fun ripple(dur: Float): FloatArray {
        val n = (RATE * dur).toInt().coerceAtLeast(1)
        val out = FloatArray(n)
        for (i in 0 until n) {
            val t = i / RATE.toFloat()
            val p = i / n.toFloat()
            val e = sin(PI * p).toFloat()                          // gentle swell in then out
            val freq = 132f + 34f * sin(2.0 * PI * 5.0 * t).toFloat()  // low tone, slow vibrato
            val w = 2.0 * PI * freq * t
            out[i] = 0.3f * e * (sin(w).toFloat() + 0.22f * sin(2.0 * w).toFloat())
        }
        return out
    }

    private fun pop(dur: Float): FloatArray {
        val n = (RATE * dur).toInt().coerceAtLeast(1)
        val out = FloatArray(n)
        val w = (2.0 * PI * 180f).toFloat()
        for (i in 0 until n) {
            val t = i / RATE.toFloat()
            val e = exp(-18f * t)
            val noise = Random.nextFloat() * 2f - 1f
            out[i] = 0.28f * e * noise + 0.25f * e * sin(w * t)
        }
        return out
    }

    /** Fast attack, exponential decay envelope. */
    private fun env(i: Int, n: Int): Float {
        val attack = (n * 0.06f).coerceAtLeast(1f)
        return if (i < attack) i / attack
        else exp(-3.2f * ((i - attack) / (n - attack).coerceAtLeast(1f)))
    }

    private fun wav(samples: FloatArray): ByteArray {
        val dataSize = samples.size * 2
        val buf = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("RIFF".toByteArray(Charsets.US_ASCII)); buf.putInt(36 + dataSize)
        buf.put("WAVE".toByteArray(Charsets.US_ASCII))
        buf.put("fmt ".toByteArray(Charsets.US_ASCII)); buf.putInt(16)
        buf.putShort(1.toShort())            // PCM
        buf.putShort(1.toShort())            // mono
        buf.putInt(RATE)
        buf.putInt(RATE * 2)                 // byte rate
        buf.putShort(2.toShort())            // block align
        buf.putShort(16.toShort())           // bits/sample
        buf.put("data".toByteArray(Charsets.US_ASCII)); buf.putInt(dataSize)
        for (s in samples) buf.putShort((s.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        return buf.array()
    }
}
