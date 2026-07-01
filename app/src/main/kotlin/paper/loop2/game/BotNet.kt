package paper.loop2.game

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.badlogic.gdx.Gdx
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Neural bot brain: the small (~0.53M-param) self-play PPO net — the SAME architecture and parameter
 * count as Paper Loop 1's shipped on-device net (only the policy head differs: 5 turn buckets for free
 * movement vs 3 grid turns). Two interchangeable inference backends, exactly like Paper Loop 1, so we
 * keep the fast path AND a clean fallback / cross-check:
 *   - [Mode.ONNX]   onnxruntime-android (NNAPI/XNNPACK native runtime) — the fast primary backend
 *   - [Mode.KOTLIN] pure-Kotlin CNN forward (no native dep — F-Droid-clean fallback + correctness check)
 *
 * The observation encoder is a 1:1 port of the gym's `encode_all` (egocentric crop rotated forward=up,
 * for continuous heading), so the net sees exactly what it trained on. Weights load from
 * `assets/bot.onnx` (ONNX) and `assets/bot.plb` (PL21 float32 blob). Any failure degrades gracefully.
 */
class BotNet {
    enum class Mode { ONNX, KOTLIN }
    var mode = Mode.ONNX; private set
    var ready = false; private set            // at least one backend usable
    var onnxReady = false; private set

    private val w = HashMap<String, FloatArray>()      // Kotlin-backend weights by name
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    // reusable obs buffers (batch-sized for MAXP seats; ONNX batches, Kotlin reads per slot)
    private val gridBuf = FloatArray(MAXP * C * HW)
    private val scalBuf = FloatArray(MAXP * S)

    // ---- async inference snapshot (sim thread writes `work*`; worker reads them) ----------------
    // Inference runs on a dedicated worker thread, OFF the render/sim thread, so a slow inference can
    // never stall rendering nor multiply across the fixed sub-steps (no frame-time spiral). Each
    // decision window the sim thread takes a cheap snapshot of exactly what `encode` needs into the
    // work buffers; the worker encodes + runs the net from that snapshot and publishes target headings
    // back. Player index == id (the players list is id-ordered and never reordered), so a slot's id is
    // its index. Bots hold heading between decisions, so a ~1-frame result latency is invisible.
    private val workOwner = ByteArray(NCELLS)
    private val workTrail = ByteArray(NCELLS)
    private val workX = FloatArray(MAXP); private val workY = FloatArray(MAXP); private val workHeading = FloatArray(MAXP)
    private val workAlive = BooleanArray(MAXP); private val workHuman = BooleanArray(MAXP)
    private val workArea = IntArray(MAXP); private val workHasTrail = BooleanArray(MAXP)
    private val workExitX = FloatArray(MAXP); private val workExitY = FloatArray(MAXP); private val workTrailLen = IntArray(MAXP)
    private var workCount = 0; private var workStep = 0; private var workGen = 0
    private val dueIdx = IntArray(MAXP)

    private val pendingHeading = FloatArray(MAXP)
    private val pendingDue = BooleanArray(MAXP)
    private var pendingGen = -1

    private val lock = Object()
    private var requestPending = false       // a snapshot is queued for the worker
    private var inFlight = false             // a request is outstanding (worker busy or queued)
    @Volatile private var resultReady = false
    @Volatile private var running = true
    private var worker: Thread? = null
    private var lastSubmitStep = -1000      // small sentinel (avoids Int overflow in the stepCount delta)

    init {
        var plb = false
        try { loadWeights(); plb = true } catch (e: Throwable) {
            Gdx.app.error("BotNet", "PLB load failed: $e") }
        initOnnx()
        mode = if (onnxReady) Mode.ONNX else Mode.KOTLIN
        ready = onnxReady || plb
        if (ready) startWorker()
        Gdx.app.log("BotNet", "ready=$ready mode=$mode onnx=$onnxReady plb=$plb threads=$ORT_THREADS async=${worker != null}")
    }

    private fun loadWeights() {
        val bb = ByteBuffer.wrap(Gdx.files.internal("bot.plb").readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4); bb.get(magic)
        require(String(magic) == "PL21") { "bad magic" }
        val n = bb.int
        repeat(n) {
            val nm = ByteArray(bb.int); bb.get(nm)
            val nd = bb.int
            var cnt = 1
            repeat(nd) { cnt *= bb.int }
            w[String(nm)] = FloatArray(cnt) { bb.float }
        }
    }

    private fun initOnnx() {
        try {
            val bytes = Gdx.files.internal("bot.onnx").readBytes()
            ortEnv = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions()
            opts.setIntraOpNumThreads(ORT_THREADS)
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (ORT_XNNPACK) {
                try { opts.addXnnpack(mapOf("intra_op_num_threads" to ORT_THREADS.toString())) }
                catch (e: Throwable) { Gdx.app.error("BotNet", "XNNPACK EP unavailable: $e") }
            }
            ortSession = ortEnv!!.createSession(bytes, opts)
            onnxReady = true
        } catch (e: Throwable) {
            Gdx.app.error("BotNet", "ONNX init failed (using Kotlin backend): $e")
        }
    }

    /** Sim-thread: copy exactly what [encodeAt] reads from the live world into the work buffers.
     *  Cheap (two 40 KB grid arraycopies + a handful of per-player scalars). Player index == id. */
    private fun snapshotInto(world: World) {
        System.arraycopy(world.owner, 0, workOwner, 0, NCELLS)
        System.arraycopy(world.trail, 0, workTrail, 0, NCELLS)
        val ps = world.players
        workCount = ps.size
        for (i in 0 until workCount) {
            val p = ps[i]
            workX[i] = p.x; workY[i] = p.y; workHeading[i] = p.heading
            workAlive[i] = p.alive; workHuman[i] = p.isHuman; workArea[i] = p.area
            workHasTrail[i] = p.hasTrail; workExitX[i] = p.exitX; workExitY[i] = p.exitY
            workTrailLen[i] = p.trailCells.size
        }
        workStep = world.stepCount; workGen = world.resetGen
    }

    // ---- observation encoder (1:1 port of env.encode_all, free movement) — reads the work snapshot ----
    private fun encodeAt(si: Int, gOff: Int, sOff: Int) {
        java.util.Arrays.fill(gridBuf, gOff, gOff + C * HW, 0f)
        val self = si + 1
        val hx = workX[si]; val hy = workY[si]; val h = workHeading[si]
        val fx = cos(h); val fy = sin(h); val rx = fy; val ry = -fx
        val wallLim2 = (ARENA_R - WALL_MARGIN) * (ARENA_R - WALL_MARGIN)
        for (iy in 0 until HH) {
            val k = R - iy
            for (ix in 0 until HH) {
                val r = ix - R
                val wx = hx + k * fx + r * rx; val wy = hy + k * fy + r * ry
                val base = gOff + iy * HH + ix
                val ddx = wx - ARENA_CX; val ddy = wy - ARENA_CY
                if (ddx * ddx + ddy * ddy >= wallLim2) { gridBuf[base] = 1f; continue }
                val cx = floor(wx).toInt(); val cy = floor(wy).toInt()
                if (cx < 0 || cx >= GW || cy < 0 || cy >= GH) { gridBuf[base] = 1f; continue }
                val cid = cy * GW + cx
                val o = workOwner[cid].toInt(); val t = workTrail[cid].toInt()
                if (o == self) gridBuf[base + HW] = 1f else if (o != 0) gridBuf[base + 3 * HW] = 1f
                if (t == self) gridBuf[base + 2 * HW] = 1f else if (t != 0) gridBuf[base + 4 * HW] = 1f
            }
        }
        var alive = 0
        for (qi in 0 until workCount) if (workAlive[qi]) alive++
        for (qi in 0 until workCount) {
            if (qi == si || !workAlive[qi]) continue
            val ex = workX[qi] - hx; val ey = workY[qi] - hy
            val iy = R - (ex * fx + ey * fy).roundToInt()
            val ix = R + (ex * rx + ey * ry).roundToInt()
            if (iy in 0 until HH && ix in 0 until HH) gridBuf[gOff + 5 * HW + iy * HH + ix] = 1f
        }
        java.util.Arrays.fill(scalBuf, sOff, sOff + S, 0f)
        val area = workArea[si]
        scalBuf[sOff] = if (workHasTrail[si]) 1f else 0f
        scalBuf[sOff + 1] = (area.toFloat() / NCELLS) * 5f
        scalBuf[sOff + 2] = minOf(workTrailLen[si] / 150f, 1.5f)
        scalBuf[sOff + 3] = alive.toFloat() / MAXP
        scalBuf[sOff + 4] = ((workStep / DECIDE_SUBSTEPS) % 2200).toFloat() / 2200f
        if (workHasTrail[si]) {
            val ex = workExitX[si] - hx; val ey = workExitY[si] - hy
            scalBuf[sOff + 5] = (abs(ex) + abs(ey)) / (GW + GH)
            scalBuf[sOff + 6] = (ex * fx + ey * fy) / GW
            scalBuf[sOff + 7] = (ex * rx + ey * ry) / GW
        }
        var best = Float.MAX_VALUE; var bf = 0f; var br = 0f
        for (qi in 0 until workCount) {
            if (qi == si || !workAlive[qi]) continue
            val ex = workX[qi] - hx; val ey = workY[qi] - hy; val md = abs(ex) + abs(ey)
            if (md < best) { best = md; bf = ex * fx + ey * fy; br = ex * rx + ey * ry }
        }
        if (best < Float.MAX_VALUE) {
            scalBuf[sOff + 8] = minOf(best / (GW + GH), 1f); scalBuf[sOff + 9] = bf / GW; scalBuf[sOff + 10] = br / GW
        } else scalBuf[sOff + 8] = 1f
        var more = 0
        for (qi in 0 until workCount) { if (qi == si || !workAlive[qi]) continue; if (workArea[qi] > area) more++ }
        scalBuf[sOff + 11] = more.toFloat() / MAXP
    }

    // ---- mode A: pure-Kotlin forward (returns N_ACTIONS logits) -------------
    private fun conv(inp: FloatArray, cin: Int, hin: Int, win: Int, wt: FloatArray, b: FloatArray,
                     cout: Int, s: Int): FloatArray {
        val hout = (hin + 2 - 3) / s + 1; val wout = (win + 2 - 3) / s + 1
        val out = FloatArray(cout * hout * wout)
        for (oc in 0 until cout) {
            val bias = b[oc]
            for (oy in 0 until hout) {
                val iy0 = oy * s - 1
                for (ox in 0 until wout) {
                    val ix0 = ox * s - 1
                    var acc = bias
                    for (ic in 0 until cin) {
                        val wb = (oc * cin + ic) * 9
                        val ib = ic * hin * win
                        for (ky in 0 until 3) {
                            val iy = iy0 + ky; if (iy < 0 || iy >= hin) continue
                            val row = ib + iy * win
                            val wr = wb + ky * 3
                            for (kx in 0 until 3) {
                                val ix = ix0 + kx; if (ix < 0 || ix >= win) continue
                                acc += inp[row + ix] * wt[wr + kx]
                            }
                        }
                    }
                    out[(oc * hout + oy) * wout + ox] = acc
                }
            }
        }
        return out
    }

    private fun groupNorm(x: FloatArray, ch: Int, hw: Int, g: FloatArray, b: FloatArray) {
        val groups = minOf(8, ch); val cpg = ch / groups; val gsz = cpg * hw
        for (grp in 0 until groups) {
            val c0 = grp * cpg
            var mean = 0f
            for (c in c0 until c0 + cpg) { val cb = c * hw; for (i in 0 until hw) mean += x[cb + i] }
            mean /= gsz
            var v = 0f
            for (c in c0 until c0 + cpg) { val cb = c * hw; for (i in 0 until hw) { val dd = x[cb + i] - mean; v += dd * dd } }
            v /= gsz
            val inv = 1f / sqrt(v + 1e-5f)
            for (c in c0 until c0 + cpg) { val cb = c * hw; val gg = g[c]; val bbv = b[c]; for (i in 0 until hw) x[cb + i] = (x[cb + i] - mean) * inv * gg + bbv }
        }
    }

    private fun relu(x: FloatArray) { for (i in x.indices) if (x[i] < 0f) x[i] = 0f }

    private fun linear(inp: FloatArray, nin: Int, wt: FloatArray, b: FloatArray, nout: Int): FloatArray {
        val out = FloatArray(nout)
        for (o in 0 until nout) {
            var acc = b[o]; val wb = o * nin
            for (i in 0 until nin) acc += inp[i] * wt[wb + i]
            out[o] = acc
        }
        return out
    }

    private fun resBlock(x: FloatArray, ch: Int, h: Int, wd: Int, pfx: String): FloatArray {
        val hw = h * wd
        val t = x.copyOf()
        groupNorm(t, ch, hw, w["$pfx.n1.weight"]!!, w["$pfx.n1.bias"]!!); relu(t)
        var y = conv(t, ch, h, wd, w["$pfx.c1.weight"]!!, w["$pfx.c1.bias"]!!, ch, 1)
        groupNorm(y, ch, hw, w["$pfx.n2.weight"]!!, w["$pfx.n2.bias"]!!); relu(y)
        y = conv(y, ch, h, wd, w["$pfx.c2.weight"]!!, w["$pfx.c2.bias"]!!, ch, 1)
        for (i in y.indices) y[i] += x[i]
        return y
    }

    private fun forwardKotlin(gOff: Int, sOff: Int): FloatArray {
        val g = gridBuf.copyOfRange(gOff, gOff + C * HW)
        var x = conv(g, C, HH, HH, w["stem.weight"]!!, w["stem.bias"]!!, 32, 2); relu(x)   // [32,11,11]
        x = resBlock(x, 32, 11, 11, "b1.0")
        x = resBlock(x, 32, 11, 11, "b1.1")
        x = conv(x, 32, 11, 11, w["d1.weight"]!!, w["d1.bias"]!!, 64, 2); relu(x)            // [64,6,6]
        x = resBlock(x, 64, 6, 6, "b2.0")
        x = resBlock(x, 64, 6, 6, "b2.1")                                                    // flat 2304
        val sc0 = scalBuf.copyOfRange(sOff, sOff + S)
        var sc = linear(sc0, S, w["scalar.0.weight"]!!, w["scalar.0.bias"]!!, 64); relu(sc)
        sc = linear(sc, 64, w["scalar.2.weight"]!!, w["scalar.2.bias"]!!, 64); relu(sc)
        val cat = FloatArray(x.size + sc.size)
        System.arraycopy(x, 0, cat, 0, x.size); System.arraycopy(sc, 0, cat, x.size, sc.size)
        var h = linear(cat, cat.size, w["trunk.0.weight"]!!, w["trunk.0.bias"]!!, 128); relu(h)
        h = linear(h, 128, w["trunk.2.weight"]!!, w["trunk.2.bias"]!!, 128); relu(h)
        return linear(h, 128, w["pi.weight"]!!, w["pi.bias"]!!, N_ACTIONS)
    }

    // ---- mode B: onnxruntime batched forward, returns [n][N_ACTIONS] -------
    private fun forwardOnnx(n: Int): Array<FloatArray> {
        val env = ortEnv!!; val sess = ortSession!!
        val gT = OnnxTensor.createTensor(env, FloatBuffer.wrap(gridBuf, 0, n * C * HW),
            longArrayOf(n.toLong(), C.toLong(), HH.toLong(), HH.toLong()))
        val sT = OnnxTensor.createTensor(env, FloatBuffer.wrap(scalBuf, 0, n * S),
            longArrayOf(n.toLong(), S.toLong()))
        val res = sess.run(mapOf("grid" to gT, "scalar" to sT))
        @Suppress("UNCHECKED_CAST")
        val out = res.get(0).value as Array<FloatArray>
        res.close(); gT.close(); sT.close()
        return out
    }

    private fun argmax(a: FloatArray, off: Int = 0): Int {
        var bi = 0; var bv = a[off]
        for (i in 1 until N_ACTIONS) if (a[off + i] > bv) { bv = a[off + i]; bi = i }
        return bi
    }

    private fun headFor(si: Int, action: Int): Float = wrapAngle(workHeading[si] + (action - HALF_SPAN) * TURN_STEP)

    /**
     * Sim-thread entry (wired to `World.neuralDecide`, called every sub-step). Cheap: it (a) applies
     * whatever decision the worker has finished, then (b) once per [DECIDE_SUBSTEPS] window snapshots
     * the world and kicks the worker. The net forward runs on [worker] OFF the render/sim thread, so
     * this never blocks rendering and never multiplies across the fixed sub-steps — no frame-time
     * spiral. Bots decide once per window (gym cadence) and hold heading between; the ~1-frame result
     * latency is invisible. The whole net batches into ONE forward per window (no per-call overhead ×4).
     */
    fun actAll(world: World) {
        if (!ready) return
        apply(world)
        // submit once per window; `stepCount < lastSubmitStep` detects a reset (stepCount rewinds to 0)
        if (world.stepCount - lastSubmitStep >= DECIDE_SUBSTEPS || world.stepCount < lastSubmitStep) {
            lastSubmitStep = world.stepCount
            submit(world)
        }
    }

    /** Sim-thread: claim the worker (if idle), snapshot the world, and signal the worker to run. */
    private fun submit(world: World) {
        synchronized(lock) { if (inFlight) return; inFlight = true }   // claim: now safe to write work*
        snapshotInto(world)                                            // worker idle -> no lock needed
        synchronized(lock) { requestPending = true; lock.notifyAll() }
    }

    /** Sim-thread: apply the worker's latest finished decision to the live players. */
    private fun apply(world: World) {
        if (!resultReady) return
        synchronized(lock) {
            if (!resultReady) return
            resultReady = false
            if (pendingGen != world.resetGen) return       // snapshot predates a reset -> discard
            val ps = world.players
            val cnt = minOf(ps.size, MAXP)
            for (i in 0 until cnt) if (pendingDue[i]) {
                val p = ps[i]; if (p.alive && !p.isHuman) p.targetHeading = pendingHeading[i]
            }
        }
    }

    private fun startWorker() {
        val t = Thread({
            // Warm up off the GL thread: JIT encode/forward and prime the ONNX session so the first
            // in-game decision isn't a cold-start spike. Uses a trivial 1-player snapshot.
            try {
                workCount = 1; workAlive[0] = true; workHuman[0] = false; workX[0] = ARENA_CX; workY[0] = ARENA_CY
                repeat(4) {
                    encodeAt(0, 0, 0)
                    if (mode == Mode.ONNX && onnxReady) forwardOnnx(1) else forwardKotlin(0, 0)
                }
                workCount = 0
            } catch (_: Throwable) {}
            while (running) {
                synchronized(lock) {
                    while (running && !requestPending) lock.wait()
                    if (!running) return@Thread
                    requestPending = false      // snapshot already in work* (submit wrote it before signalling)
                }
                try { runInferenceOnce() }
                catch (e: Throwable) { Gdx.app.error("BotNet", "async inference failed: $e") }
                finally { synchronized(lock) { inFlight = false } }
            }
        }, "BotNet-infer")
        t.isDaemon = true
        // Below-normal priority: inference has a generous (~66 ms) window budget, so the render thread
        // should always win the big cores. Keeps frame pacing smooth while bots still decide on cadence.
        t.priority = (Thread.NORM_PRIORITY - 2).coerceAtLeast(Thread.MIN_PRIORITY)
        t.start()
        worker = t
    }

    /** Worker thread: encode the due bots from the snapshot, run ONE batched forward, publish headings. */
    private fun runInferenceOnce() {
        var n = 0
        for (i in 0 until workCount) if (workAlive[i] && !workHuman[i]) dueIdx[n++] = i
        if (n == 0) return
        for (i in 0 until n) encodeAt(dueIdx[i], i * C * HW, i * S)
        val gen = workGen
        val acts = IntArray(n)
        if (mode == Mode.ONNX && onnxReady) {
            val logits = forwardOnnx(n)
            for (i in 0 until n) acts[i] = argmax(logits[i])
        } else {
            for (i in 0 until n) acts[i] = argmax(forwardKotlin(i * C * HW, i * S))
        }
        synchronized(lock) {
            java.util.Arrays.fill(pendingDue, false)
            for (i in 0 until n) { val si = dueIdx[i]; pendingHeading[si] = headFor(si, acts[i]); pendingDue[si] = true }
            pendingGen = gen; resultReady = true
        }
    }

    /** On-device latency + cross-check probe; result -> logcat (BotNetBench) AND files/botnet_bench.txt. */
    fun benchmark(iters: Int = 50) {
        val out = StringBuilder()
        try {
            if (!ready) { out.append("not ready\n") } else {
                val tw = World(); tw.reset(numBots = 8, numColors = 12)
                val bots = ArrayList<Player>()
                for (p in tw.players) if (p.alive && !p.isHuman) bots.add(p)
                val n = bots.size
                snapshotInto(tw)                                        // bot i is players[i+1] -> index i+1
                for (i in 0 until n) encodeAt(i + 1, i * C * HW, i * S)
                // correctness cross-check (only if both backends present)
                if (onnxReady) {
                    val lo = forwardOnnx(n)
                    var maxDiff = 0f; var agree = 0
                    for (i in 0 until n) {
                        val lk = forwardKotlin(i * C * HW, i * S)
                        for (j in 0 until N_ACTIONS) maxDiff = maxOf(maxDiff, abs(lk[j] - lo[i][j]))
                        if (argmax(lk) == argmax(lo[i])) agree++
                    }
                    out.append("cross-check n=$n argmax_agree=$agree/$n max_logit_diff=$maxDiff\n")
                }
                if (onnxReady) {
                    // encode-only cost (per due bot)
                    repeat(6) { encodeAt(1, 0, 0) }
                    var te = System.nanoTime()
                    repeat(iters) { encodeAt(1, 0, 0) }
                    out.append("ENCODE 1x ${"%.3f".format((System.nanoTime() - te) / 1e6 / iters)} ms\n")
                    // forwardOnnx fixed-overhead probe: time at several batch sizes (no encode)
                    for (bn in intArrayOf(1, 2, 4, 7, 8)) {
                        if (bn > n) continue
                        repeat(6) { forwardOnnx(bn) }
                        val t = System.nanoTime()
                        repeat(iters) { forwardOnnx(bn) }
                        val ms = (System.nanoTime() - t) / 1e6 / iters
                        out.append("ONNXfwd n=$bn ${"%.2f".format(ms)} ms (${"%.3f".format(ms / bn)} ms/bot)\n")
                    }
                    repeat(6) { forwardOnnx(n) }
                    val t0 = System.nanoTime()
                    repeat(iters) { for (i in 0 until n) encodeAt(i + 1, i * C * HW, i * S); forwardOnnx(n) }
                    val ms = (System.nanoTime() - t0) / 1e6 / iters
                    out.append("ONNX   n=$n ${"%.2f".format(ms)} ms/batch (${"%.3f".format(ms / n)} ms/bot)\n")
                }
                // pure-Kotlin timing intentionally skipped (known ~108 ms/bot; the loop runs ~50 s and
                // would contend with gameplay). The cross-check above already exercises that backend.
            }
        } catch (e: Throwable) { out.append("bench error: $e\n") }
        Gdx.app.error("BotNetBench", out.toString())
        try { Gdx.files.local("botnet_bench.txt").writeString(out.toString(), false) } catch (_: Throwable) {}
    }

    fun dispose() {
        running = false
        synchronized(lock) { lock.notifyAll() }
        try { worker?.join(500) } catch (_: Throwable) {}
        try { ortSession?.close(); ortEnv?.close() } catch (_: Throwable) {}
    }

    companion object {
        // ONNX session tuning (perf experiment knobs; see on-device probe)
        var ORT_THREADS = 3
        var ORT_XNNPACK = false
        const val C = 6; const val HH = 21; const val HW = HH * HH; const val S = 12; const val R = 10
        const val MAXP = 12; const val NCELLS = GW * GH
        const val HALF_SPAN = (N_ACTIONS - 1) * 0.5f
        val TURN_STEP = MAX_TURN_PER_DECISION / HALF_SPAN
    }
}
