package paper.loop2.core

import android.content.Context
import android.content.SharedPreferences

/** Persistent player options. Read from the GL thread, so fields are @Volatile. */
object Settings {
    private lateinit var prefs: SharedPreferences

    @Volatile var soundOn: Boolean = true
        private set
    @Volatile var hapticsOn: Boolean = true
        private set
    @Volatile var darkMode: Boolean = false
        private set
    /** Opponent type: 0 = AI HARD (neural net), 1 = EASY SCRIPTED (the scripted Bot), 2 = NO BOTS. */
    @Volatile var botMode: Int = 0
        private set
    /** Whether eliminated bots re-spawn (applies to bots only; the human always ends the run). */
    @Volatile var respawnBots: Boolean = true
        private set

    fun init(ctx: Context) {
        prefs = ctx.applicationContext.getSharedPreferences("settings", Context.MODE_PRIVATE)
        soundOn = prefs.getBoolean("sound", true)
        hapticsOn = prefs.getBoolean("haptics", true)
        darkMode = prefs.getBoolean("dark", false)
        botMode = prefs.getInt("botMode", 0).coerceIn(0, 2)
        // "respawnMode" is a leftover from the build that briefly offered old/new siting (2 = off)
        respawnBots = if (prefs.contains("respawnMode")) prefs.getInt("respawnMode", 0) != 2
        else prefs.getBoolean("respawn", true)
    }

    fun setBotMode(v: Int) {
        botMode = v.coerceIn(0, 2)
        prefs.edit().putInt("botMode", botMode).apply()
    }

    fun setRespawnBots(v: Boolean) {
        respawnBots = v
        prefs.edit().putBoolean("respawn", v).remove("respawnMode").apply()
    }

    fun setDarkMode(v: Boolean) {
        darkMode = v
        prefs.edit().putBoolean("dark", v).apply()
    }

    fun setSound(v: Boolean) {
        soundOn = v
        prefs.edit().putBoolean("sound", v).apply()
    }

    fun setHaptics(v: Boolean) {
        hapticsOn = v
        prefs.edit().putBoolean("haptics", v).apply()
    }
}
