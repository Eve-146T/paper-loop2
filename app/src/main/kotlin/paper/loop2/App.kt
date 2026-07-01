package paper.loop2

import android.app.Application
import paper.loop2.core.Haptics
import paper.loop2.core.Scores
import paper.loop2.core.Settings

/** Initializes the Android-context singletons. SoundFx is initialized later, on the
 *  GL thread, because it needs Gdx.audio (only available once libGDX is running). */
class App : Application() {
    override fun onCreate() {
        super.onCreate()
        Scores.init(this)
        Settings.init(this)
        Haptics.init(this)
    }
}
