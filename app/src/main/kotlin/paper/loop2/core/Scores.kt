package paper.loop2.core

import android.content.Context
import android.content.SharedPreferences

/** Persistent best score, one entry per id. Score is stored as tenths-of-a-percent. */
object Scores {
    private lateinit var prefs: SharedPreferences

    fun init(ctx: Context) {
        prefs = ctx.applicationContext.getSharedPreferences("scores", Context.MODE_PRIVATE)
    }

    fun best(id: String): Int = prefs.getInt("best_$id", 0)

    /** Records [v] if it beats the stored best. Returns true when it is a new best. */
    fun submit(id: String, v: Int): Boolean {
        if (v <= best(id)) return false
        prefs.edit().putInt("best_$id", v).apply()
        return true
    }
}
