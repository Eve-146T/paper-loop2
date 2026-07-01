package paper.loop2.game

import com.badlogic.gdx.Game
import paper.loop2.core.Assets
import paper.loop2.core.Palette
import paper.loop2.core.Settings
import paper.loop2.core.SoundFx

/** Root libGDX game: loads shared assets/UI, then runs straight into the single game screen. */
class PaperLoopGame : Game() {
    lateinit var assets: Assets
        private set
    lateinit var ui: Ui
        private set

    val scoreId = "paperloop2"
    val numBots = 7

    override fun create() {
        Palette.setDark(Settings.darkMode)
        // Fonts build on a background thread (FreeType rasterisation is the big cold-start cost);
        // the white pixel is ready immediately, so the world + HUD shapes draw and the game is
        // tappable from the first frame instead of freezing while glyphs generate.
        assets = Assets()
        assets.beginAsyncLoad()
        ui = Ui(assets)
        setScreen(GameScreen(this))
        // The sound bank (WAV synthesis + decode) is also off the startup path. It builds on a
        // background thread; play() no-ops until ready.
        Thread({ SoundFx.init() }, "sfx-init").apply { isDaemon = true; start() }
    }

    override fun dispose() {
        screen?.dispose()
        ui.dispose()
        assets.dispose()
        SoundFx.dispose()
    }
}
