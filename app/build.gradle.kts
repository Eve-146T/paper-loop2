import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val gdxVersion = "1.13.1"
val natives: Configuration by configurations.creating

// Release signing. Local builds read keystore.properties (gitignored); CI reads
// the equivalent environment variables. With neither present the release build
// is produced unsigned, so the project still builds for anyone without the key.
val keystorePropsFile = rootProject.file("keystore.properties")
val keystoreProps = Properties().apply {
    if (keystorePropsFile.exists()) keystorePropsFile.inputStream().use { load(it) }
}
fun signingValue(prop: String, env: String): String? =
    keystoreProps.getProperty(prop) ?: System.getenv(env)
val releaseStoreFile: String? = signingValue("storeFile", "KEYSTORE_FILE")

// F-Droid ABI split. Release builds emit one APK per ABI (smaller downloads) plus a
// universal convenience APK. Each per-ABI APK gets a distinct versionCode so the store
// can offer the right native build per device: base*10 + index, with the ordering the
// F-Droid docs recommend (armeabi-v7a < arm64-v8a < x86 < x86_64). Splitting is limited
// to release tasks, so `assembleDebug`/`installDebug` still produce the usual single
// app-debug.apk for local dev. F-Droid builds one ABI per metadata block via
// `-PabiFilter=<abi>`; a normal release build (CI, local) produces all of them.
val abiVersionCodes = linkedMapOf(
    "armeabi-v7a" to 1,
    "arm64-v8a" to 2,
    "x86" to 3,
    "x86_64" to 4,
)
val abiFilter = (project.findProperty("abiFilter") as String?)?.trim()?.takeIf { it.isNotEmpty() }
val splitApks = gradle.startParameter.taskNames.any { it.contains("release", ignoreCase = true) }
val includedAbis = abiFilter?.let { listOf(it) } ?: abiVersionCodes.keys.toList()

android {
    namespace = "paper.loop2"
    compileSdk = 35

    defaultConfig {
        applicationId = "paper.loop2"
        minSdk = 29
        targetSdk = 35
        versionCode = 3
        versionName = "1.0.2"
    }

    // Keep release APKs free of Google's dependency-metadata signing block,
    // which F-Droid rejects (and which breaks reproducible builds).
    dependenciesInfo {
        includeInApk = false
        includeInBundle = false
    }

    // Per-ABI release APKs (see abiVersionCodes above). Disabled for debug so local
    // dev keeps a single app-debug.apk.
    splits {
        abi {
            isEnable = splitApks
            reset()
            include(*includedAbis.toTypedArray())
            isUniversalApk = abiFilter == null
        }
    }

    signingConfigs {
        if (releaseStoreFile != null) {
            create("release") {
                storeFile = file(releaseStoreFile)
                storePassword = signingValue("storePassword", "KEYSTORE_PASSWORD")
                keyAlias = signingValue("keyAlias", "KEY_ALIAS")
                keyPassword = signingValue("keyPassword", "KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Reproducible builds: don't embed VCS info (version-control-info.textproto).
            // It records the git commit/checkout state, which varies between the CI release
            // build and F-Droid's detached-HEAD rebuild and would break byte-for-byte matching.
            vcsInfo {
                include = false
            }
            if (releaseStoreFile != null) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    sourceSets["main"].jniLibs.srcDirs("libs")

    // 16 KB page-size compatibility: keep the native libs uncompressed so they're mapped straight
    // from the APK (extractNativeLibs=false) and AGP 8.7 zip-aligns them to 16 KB. Paired with the
    // 16 KB-aligned onnxruntime 1.22.0 above, the app loads cleanly on Android 15+ 16 KB devices.
    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }

    lint {
        // False positive: the adaptive launcher icon must stay in mipmap-anydpi-v26.
        disable += "ObsoleteSdkInt"
    }
}

// Stamp each per-ABI release APK with base*10 + index (universal keeps base*10).
if (splitApks) {
    android.applicationVariants.all {
        outputs.all {
            val out = this as com.android.build.gradle.internal.api.ApkVariantOutputImpl
            val abi = out.getFilter(com.android.build.OutputFile.ABI)
            out.versionCodeOverride =
                (android.defaultConfig.versionCode ?: 0) * 10 + (abi?.let { abiVersionCodes[it] } ?: 0)
        }
    }
}

dependencies {
    implementation("com.badlogicgames.gdx:gdx:$gdxVersion")
    implementation("com.badlogicgames.gdx:gdx-backend-android:$gdxVersion")
    implementation("com.badlogicgames.gdx:gdx-freetype:$gdxVersion")
    // On-device neural bots: onnxruntime-android (fast NNAPI/XNNPACK backend), with a pure-Kotlin
    // forward as the F-Droid-clean fallback + cross-check (mirrors Paper Loop 1's dual backend).
    // 1.22.0 ships 16 KB-aligned native libs (Android 15+ 16 KB page-size devices load cleanly).
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
    // Headless unit tests for the pure-Kotlin simulation (capture/collision invariants, no device).
    testImplementation(kotlin("test"))
    natives("com.badlogicgames.gdx:gdx-platform:$gdxVersion:natives-armeabi-v7a")
    natives("com.badlogicgames.gdx:gdx-platform:$gdxVersion:natives-arm64-v8a")
    natives("com.badlogicgames.gdx:gdx-platform:$gdxVersion:natives-x86")
    natives("com.badlogicgames.gdx:gdx-platform:$gdxVersion:natives-x86_64")
    natives("com.badlogicgames.gdx:gdx-freetype-platform:$gdxVersion:natives-armeabi-v7a")
    natives("com.badlogicgames.gdx:gdx-freetype-platform:$gdxVersion:natives-arm64-v8a")
    natives("com.badlogicgames.gdx:gdx-freetype-platform:$gdxVersion:natives-x86")
    natives("com.badlogicgames.gdx:gdx-freetype-platform:$gdxVersion:natives-x86_64")
}

// Extract libGDX native .so files into jniLibs before they are merged.
tasks.register("copyAndroidNatives") {
    doFirst {
        natives.files.forEach { jar ->
            val abi = jar.nameWithoutExtension.substringAfterLast("natives-")
            val outputDir = file("libs/$abi")
            outputDir.mkdirs()
            copy {
                from(zipTree(jar))
                into(outputDir)
                include("*.so")
            }
        }
    }
}

tasks.matching { it.name.contains("merge") && it.name.contains("JniLibFolders") }.configureEach {
    dependsOn("copyAndroidNatives")
}
