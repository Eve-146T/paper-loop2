package paper.loop2

import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.view.WindowManager
import com.badlogic.gdx.backends.android.AndroidApplication
import com.badlogic.gdx.backends.android.AndroidApplicationConfiguration
import paper.loop2.core.Settings
import paper.loop2.game.PaperLoopGame

/**
 * Single-activity libGDX host. All UI (menu, HUD, game-over) is rendered inside libGDX.
 *
 * Startup is tuned in [PaperLoopGame]/[paper.loop2.core.Assets]: the fonts are rasterised on a
 * background thread (FreeType glyph rendering is pure CPU and was the big cost), the sound bank is
 * synthesised on another, and the white pixel is the only GL asset made up-front. So the first frame
 * shows the live world immediately and input works while the menu text fades in a moment later —
 * there is no frozen "loading" screen. The remaining cold-start cost is OS process creation (zygote
 * fork + ART init + class load) — a platform floor identical for any app, not something we can cut.
 */
class GameActivity : AndroidApplication() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        // Set the window background to the theme colour so dark mode doesn't flash the light
        // (cream) theme background before the first GL frame draws.
        window.setBackgroundDrawable(ColorDrawable(if (Settings.darkMode) 0xFF1F2128.toInt() else 0xFFEAE4D2.toInt()))

        val config = AndroidApplicationConfiguration().apply {
            useImmersiveMode = true
            useAccelerometer = false
            useCompass = false
            useGyroscope = false
            numSamples = 2          // light MSAA — smooths diagonal trail folds / circles (free at startup)
            r = 8; g = 8; b = 8; a = 0   // opaque surface: no framebuffer alpha needed
        }
        initialize(PaperLoopGame(), config)
    }
}
