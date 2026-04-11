// =============================================================================
// File:    sketch.js
// Project: MediaPipe Demos — MDes Prototyping, CCA
// Demo:    1 — Hand Tracking
//
// Authors: Copilot
//          Thomas J McLeish
// License: MIT — see LICENSE in the root of this repository
// =============================================================================
//
// PURPOSE
// -------
// This sketch demonstrates how to use MediaPipe Hands to detect and track
// hand landmarks in real time using your webcam. Up to two hands are tracked
// simultaneously. For each hand, 21 landmark points are returned — one per
// knuckle, fingertip, and palm joint.
//
// Landmarks are drawn as green dots directly over the webcam feed. The hand
// skeleton (connections between adjacent joints) is drawn in a lighter green.
//
// WHAT IS MediaPipe HAND LANDMARKER?
// -----------------------------------
// MediaPipe Hand Landmarker (part of the Tasks Vision API) is a
// machine-learning model from Google that analyses each video frame and
// returns the 3-D positions of 21 key points on each hand it detects. The
// model runs entirely in the browser — no data ever leaves your device.
//
// The 21 landmarks are numbered 0–20:
//   0  = WRIST
//   1–4  = THUMB (base to tip)
//   5–8  = INDEX FINGER (base to tip)
//   9–12 = MIDDLE FINGER
//   13–16= RING FINGER
//   17–20= PINKY FINGER
//
// Each landmark has x, y, and z properties. x and y are normalised to 0–1
// relative to the video frame width and height. z is depth (smaller = closer
// to the camera). To convert to pixel coordinates, multiply x by the canvas
// width and y by the canvas height.
//
// HOW THE TASKS API DIFFERS FROM THE LEGACY SOLUTIONS API
// --------------------------------------------------------
// This demo uses @mediapipe/tasks-vision (the current API) instead of the
// deprecated @mediapipe/hands package. Key differences:
//
//   Legacy pattern                Tasks API pattern
//   ─────────────────────────     ──────────────────────────────────────
//   new Hands({ locateFile })     HandLandmarker.createFromOptions(...)
//   hands.setOptions(...)         options passed to createFromOptions
//   hands.onResults(callback)     result returned by detectForVideo(...)
//   await hands.send({ image })   result = landmarker.detectForVideo(video, ts)
//   results.multiHandLandmarks    result.landmarks
//
// DEBUGGING
// ---------
// Set debugMode = true to log landmark counts and frame events to the
// browser console. Open DevTools (F12) → Console tab to see the output.
// Set it back to false for smoother performance during normal use.

// =============================================================================
// CONSTANTS
// =============================================================================

// Toggle console logging. Add ?debug=1 to the URL to surface live diagnostics
// on devices where DevTools is unavailable.
const queryParams = new URLSearchParams(window.location.search);
const debugMode = queryParams.get("debug") === "1";

// Phones and tablets are more likely to have flaky GPU delegate behaviour in
// browser-based MediaPipe tasks, so default them to CPU unless overridden.
const isMobileDevice = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
const requestedDelegate = (queryParams.get("delegate") || "").toLowerCase();
const preferredDelegate = requestedDelegate === "gpu" || requestedDelegate === "cpu"
  ? requestedDelegate.toUpperCase()
  : (isMobileDevice ? "CPU" : "GPU");

// Slightly lower thresholds make the demo more tolerant of soft focus and
// noisy mobile front cameras. They can be overridden from the URL.
const defaultConfidence = isMobileDevice ? 0.35 : 0.5;
const minHandDetectionConfidence = Number.parseFloat(
  queryParams.get("minDetect") || String(defaultConfidence)
);
const minHandPresenceConfidence = Number.parseFloat(
  queryParams.get("minPresence") || String(defaultConfidence)
);
const minTrackingConfidence = Number.parseFloat(
  queryParams.get("minTrack") || String(defaultConfidence)
);

// Shared CDN paths for the MediaPipe Tasks runtime and hand model.
const TASKS_VISION_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21";
const HAND_MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker" +
  "/hand_landmarker/float16/latest/hand_landmarker.task";

// Colour for landmark dots (green).
const LANDMARK_COLOR = "#4ade80";

// Colour for the skeleton lines connecting adjacent landmarks.
const SKELETON_COLOR = "rgba(74, 222, 128, 0.5)";

// Radius of each landmark dot in pixels.
const DOT_RADIUS = 6;

// =============================================================================
// HAND CONNECTIONS
// =============================================================================
// MediaPipe defines which landmark indices to connect when drawing the hand
// skeleton. Each pair [a, b] means "draw a line from landmark a to landmark b."
//
// The connections trace each finger from the wrist outward:
//   Thumb:  0→1→2→3→4
//   Index:  0→5→6→7→8
//   Middle: 0→9→10→11→12
//   Ring:   0→13→14→15→16
//   Pinky:  0→17→18→19→20
//   Palm:   5→9→13→17 (cross-palm connections)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],           // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],           // Index finger
  [0, 9], [9, 10], [10, 11], [11, 12],      // Middle finger
  [0, 13], [13, 14], [14, 15], [15, 16],    // Ring finger
  [0, 17], [17, 18], [18, 19], [19, 20],    // Pinky finger
  [5, 9], [9, 13], [13, 17]                 // Palm cross-connections
];

// =============================================================================
// ENTRY POINT
// =============================================================================

window.onload = async function () {

  if (debugMode) {
    console.log("Page loaded, initializing MediaPipe Hand Landmarker demo...");
  }

  // ── DOM elements ────────────────────────────────────────────────────────

  const video  = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx    = canvas.getContext("2d", { alpha: false });

  if (!video || !canvas) {
    console.error("Error: Could not find #video or #canvas in the DOM.");
    return;
  }

  // Lightweight status banner for mobile where DevTools is unavailable.
  const statusBanner = document.createElement("div");
  const statusHost = canvas.parentElement || document.body;
  if (statusHost !== document.body && getComputedStyle(statusHost).position === "static") {
    statusHost.style.position = "relative";
  }
  statusBanner.style.position = statusHost === document.body ? "fixed" : "absolute";
  statusBanner.style.left = "12px";
  statusBanner.style.top = "12px";
  statusBanner.style.zIndex = "9999";
  statusBanner.style.padding = "8px 10px";
  statusBanner.style.borderRadius = "8px";
  statusBanner.style.background = "rgba(0, 0, 0, 0.72)";
  statusBanner.style.color = "#f3f4f6";
  statusBanner.style.font = "12px/1.3 monospace";
  statusBanner.style.maxWidth = "min(80vw, 420px)";
  statusBanner.style.pointerEvents = "none";
  statusBanner.style.display = "none";
  statusHost.appendChild(statusBanner);

  const debugPanel = document.createElement("pre");
  debugPanel.style.position = statusHost === document.body ? "fixed" : "absolute";
  debugPanel.style.right = "12px";
  debugPanel.style.top = "12px";
  debugPanel.style.zIndex = "9998";
  debugPanel.style.margin = "0";
  debugPanel.style.padding = "10px 12px";
  debugPanel.style.borderRadius = "8px";
  debugPanel.style.background = "rgba(17, 24, 39, 0.84)";
  debugPanel.style.border = "1px solid rgba(74, 222, 128, 0.55)";
  debugPanel.style.color = "#d1fae5";
  debugPanel.style.font = "11px/1.4 monospace";
  debugPanel.style.maxWidth = "min(86vw, 320px)";
  debugPanel.style.whiteSpace = "pre-wrap";
  debugPanel.style.pointerEvents = "none";
  debugPanel.style.display = debugMode ? "block" : "none";
  statusHost.appendChild(debugPanel);

  let lastErrorMessage = "none";
  let currentDelegate = "GPU";
  let framesProcessed = 0;
  let lastDetectionCount = 0;
  let inferenceRecoveryAttempted = false;
  let inferenceRecoveryInProgress = false;
  let handResults = [];
  let lastTimestamp = -1;
  let currentStream = null;
  let frameLoopActive = false;
  let handLandmarker;
  let HandLandmarker;
  let FilesetResolver;
  let vision;

  function clampConfidence(value) {
    if (Number.isNaN(value)) return defaultConfidence;
    return Math.min(1, Math.max(0, value));
  }

  function formatError(err) {
    if (!err) return "Unknown error";
    if (typeof err === "string") return err;
    if (err instanceof Error) return err.message;
    return JSON.stringify(err);
  }

  function showStatus(message, isError = false) {
    statusBanner.textContent = message;
    statusBanner.style.display = "block";
    statusBanner.style.border = isError
      ? "1px solid rgba(248, 113, 113, 0.9)"
      : "1px solid rgba(156, 163, 175, 0.7)";
  }

  function clearStatus() {
    statusBanner.style.display = "none";
  }

  function updateDebugPanel(extraLines = []) {
    if (!debugMode) return;

    const streamTrack = currentStream ? currentStream.getVideoTracks()[0] : null;
    const settings = streamTrack ? streamTrack.getSettings() : null;
    const resolution = settings
      ? `${settings.width || "?"}x${settings.height || "?"}`
      : "no stream";

    debugPanel.textContent = [
      `mobile device: ${isMobileDevice}`,
      `preferred delegate: ${preferredDelegate}`,
      `delegate: ${currentDelegate}`,
      `secure context: ${window.isSecureContext}`,
      `min detect: ${clampConfidence(minHandDetectionConfidence).toFixed(2)}`,
      `min presence: ${clampConfidence(minHandPresenceConfidence).toFixed(2)}`,
      `min track: ${clampConfidence(minTrackingConfidence).toFixed(2)}`,
      `video readyState: ${video.readyState}`,
      `video size: ${video.videoWidth}x${video.videoHeight}`,
      `stream size: ${resolution}`,
      `frames processed: ${framesProcessed}`,
      `hands in last frame: ${lastDetectionCount}`,
      `last error: ${lastErrorMessage}`,
      ...extraLines
    ].join("\n");
  }

  async function buildHandLandmarker(delegate) {
    showStatus(`Loading hand model (${delegate})...`);
    return HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_MODEL_ASSET_PATH,
        delegate
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: clampConfidence(minHandDetectionConfidence),
      minHandPresenceConfidence: clampConfidence(minHandPresenceConfidence),
      minTrackingConfidence: clampConfidence(minTrackingConfidence)
    });
  }

  async function createHandLandmarkerWithFallback() {
    const fallbackDelegate = preferredDelegate === "GPU" ? "CPU" : "GPU";

    try {
      handLandmarker = await buildHandLandmarker(preferredDelegate);
      currentDelegate = preferredDelegate;
    } catch (delegateErr) {
      console.warn(`${preferredDelegate} delegate unavailable, retrying with ${fallbackDelegate}:`, delegateErr);
      lastErrorMessage = `${preferredDelegate} init failed: ${formatError(delegateErr)}`;
      updateDebugPanel();
      handLandmarker = await buildHandLandmarker(fallbackDelegate);
      currentDelegate = fallbackDelegate;
    }
  }

  async function recoverFromInferenceError(err) {
    const message = formatError(err);
    lastErrorMessage = message;
    updateDebugPanel(["recovery: attempting CPU fallback"]);

    if (currentDelegate !== "GPU" || inferenceRecoveryAttempted) {
      showStatus(`Tracking stopped: ${message}`, true);
      return;
    }

    inferenceRecoveryAttempted = true;
    frameLoopActive = false;
    showStatus("GPU inference failed. Retrying with CPU...", true);

    try {
      handLandmarker = await buildHandLandmarker("CPU");
      currentDelegate = "CPU";
      lastTimestamp = -1;
      handResults = [];
      lastDetectionCount = 0;
      lastErrorMessage = `Recovered from GPU failure: ${message}`;
      updateDebugPanel(["recovery: switched to CPU"]);

      frameLoopActive = true;
      requestAnimationFrame(frameLoop);
      requestAnimationFrame(renderLoop);
      showStatus("CPU tracking active after GPU failure.");
    } catch (cpuErr) {
      const cpuMessage = formatError(cpuErr);
      lastErrorMessage = `CPU recovery failed: ${cpuMessage}`;
      updateDebugPanel(["recovery: CPU fallback failed"]);
      showStatus(`Tracking stopped: ${cpuMessage}`, true);
    }
  }

  // Stop loops and release camera tracks when leaving the page.
  window.addEventListener("beforeunload", () => {
    frameLoopActive = false;
    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
    }
  });

  try {
    showStatus("Loading MediaPipe runtime...");

    // ── MediaPipe Tasks Vision ─────────────────────────────────────────────
    // The tasks-vision library is loaded via dynamic import — no extra <script>
    // tag is needed in index.html. The .mjs bundle is an ES module that exposes
    // HandLandmarker, FilesetResolver, and other Tasks API classes.

    ({ HandLandmarker, FilesetResolver } = await import(
      `${TASKS_VISION_CDN}/vision_bundle.mjs`
    ));

    // FilesetResolver downloads the WebAssembly runtime for Tasks Vision.
    vision = await FilesetResolver.forVisionTasks(`${TASKS_VISION_CDN}/wasm`);

    // ── MediaPipe HandLandmarker setup ─────────────────────────────────────
    // createFromOptions replaces new Hands() + setOptions() from the legacy API.
    // The .task file is the model binary — fetched from Google's CDN on first
    // load, then cached by the browser. delegate: "GPU" uses WebGL acceleration;
    // the catch block retries with CPU if GPU is unavailable.

    await createHandLandmarkerWithFallback();
    updateDebugPanel();

    if (debugMode) {
      console.log("HandLandmarker ready.");
    }

  /**
   * startCamera — opens the webcam with an optional specific device and
   * starts the per-frame loop that feeds images to the Hands model.
   *
   * @param {string} [deviceId] — exact device ID to open, or omit / pass ""
   *                              to let the browser choose the default camera.
   */
  async function startCamera(deviceId) {
    // Halt the previous frame loop before replacing the stream.
    frameLoopActive = false;
    showStatus("Requesting camera access...");

    if (!navigator.mediaDevices?.getUserMedia) {
      lastErrorMessage = "getUserMedia unavailable";
      updateDebugPanel(["camera: API unavailable"]);
      showStatus("Camera API unavailable. On phones this usually means the page is not HTTPS.", true);
      return;
    }

    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
      currentStream = null;
    }

    const videoConstraints = {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: { ideal: "user" }
    };
    if (deviceId) videoConstraints.deviceId = { exact: deviceId };

    try {
      currentStream = await navigator.mediaDevices.getUserMedia(
        { video: videoConstraints }
      );
    } catch (err) {
      console.error("Could not open camera:", err);
      const message = formatError(err);
      lastErrorMessage = message;
      updateDebugPanel([`camera: ${message}`]);
      if (!window.isSecureContext) {
        showStatus("Camera blocked: mobile browsers require HTTPS for webcam access.", true);
      } else {
        showStatus(`Camera access failed: ${message}`, true);
      }
      return;
    }

    video.srcObject = currentStream;
    video.muted = true;
    video.autoplay = true;
    video.setAttribute("playsinline", "");

    // Keep canvas dimensions in sync with the new stream's resolution.
    video.onloadedmetadata = () => {
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
      updateDebugPanel();
      if (debugMode) {
        console.log(`Canvas set to ${canvas.width}×${canvas.height}`);
      }
    };

    showStatus("Starting video playback...");
    try {
      await video.play();
    } catch (err) {
      console.error("Video playback failed:", err);
      showStatus("Video playback was blocked. Tap the page and reload.", true);
      return;
    }

    if (debugMode) {
      console.log("Webcam started.");
    }

    showStatus("Camera running. Starting tracking...");
    updateDebugPanel();
    frameLoopActive = true;
    requestAnimationFrame(frameLoop);
    requestAnimationFrame(renderLoop);
  }

  /**
   * renderLoop — redraws the canvas at the display's refresh rate (~60 fps),
   * independent of the inference rate. The video feed stays smooth even when
   * the ML model runs slower than 60 fps.
   */
  function renderLoop() {
    if (!frameLoopActive) return;
    drawFrame();
    requestAnimationFrame(renderLoop);
  }

  /**
   * frameLoop — runs the HandLandmarker on the current video frame on every
   * animation tick. detectForVideo is synchronous — it returns results
   * immediately without a callback. The timestamp guard ensures the value
   * always increases, which the Tasks API requires.
   */
  function frameLoop() {
    if (!frameLoopActive) return;
    if (video.readyState >= 2) {
      clearStatus();
      const ts = performance.now();
      if (ts > lastTimestamp) {
        lastTimestamp = ts;
        if (debugMode) {
          console.log("Running HandLandmarker on current frame...");
        }
        try {
          const result = handLandmarker.detectForVideo(video, ts);
          handResults = result.landmarks ?? [];
          framesProcessed += 1;
          lastDetectionCount = handResults.length;
          updateDebugPanel();
          if (debugMode) {
            console.log(`Detected ${handResults.length} hand(s).`);
          }
        } catch (err) {
          console.error("HandLandmarker.detectForVideo failed:", err);
          frameLoopActive = false;
          if (!inferenceRecoveryInProgress) {
            inferenceRecoveryInProgress = true;
            void recoverFromInferenceError(err).finally(() => {
              inferenceRecoveryInProgress = false;
            });
          }
          return;
        }
      }
    }
    requestAnimationFrame(frameLoop);
  }

  /**
   * populateCameraSelect — enumerates video-input devices and fills the
   * on-page <select>. The wrapper is revealed only when more than one
   * camera is available. Requires camera permission to have been granted
   * so that device labels are populated.
   *
   * @param {string} activeDeviceId — the deviceId currently in use, used
   *                                  to pre-select the matching option.
   */
  async function populateCameraSelect(activeDeviceId) {
    const select  = document.getElementById("cameraSelect");
    const wrapper = document.getElementById("cameraSelectWrapper");
    if (!select || !wrapper) return;

    const devices     = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === "videoinput");

    select.innerHTML = "";
    videoInputs.forEach((device, i) => {
      const opt    = document.createElement("option");
      opt.value    = device.deviceId;
      opt.text     = device.label || `Camera ${i + 1}`;
      opt.selected = device.deviceId === activeDeviceId;
      select.appendChild(opt);
    });

    // Show the selector only when there are multiple cameras to choose from.
    wrapper.style.display = videoInputs.length > 1 ? "flex" : "none";

    select.onchange = () => startCamera(select.value);
  }

  // Initialise the camera. Enumerate devices afterwards so that the camera
  // selector shows real labels (requires permission to have been granted).
  await startCamera();
  const track    = currentStream ? currentStream.getVideoTracks()[0] : null;
  const activeId = track ? track.getSettings().deviceId : "";
  populateCameraSelect(activeId);
  updateDebugPanel();

  // ── Drawing ──────────────────────────────────────────────────────────────

  /**
   * drawFrame — clears the canvas and redraws the webcam feed plus any
   * detected hand landmarks and skeleton connections.
   *
   * Uses video.videoWidth/Height directly so drawing is correct even if
   * the loadedmetadata event fires before the first detectForVideo call.
   */
  function drawFrame() {
    const w = video.videoWidth;
    const h = video.videoHeight;

    // Keep canvas dimensions in sync with the live video size.
    if (canvas.width !== w)  canvas.width  = w;
    if (canvas.height !== h) canvas.height = h;

    // Clear and redraw the webcam image as the background.
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(video, 0, 0, w, h);

    // Draw each detected hand.
    handResults.forEach((landmarks, handIndex) => {
      drawSkeleton(landmarks, w, h);
      drawLandmarks(landmarks, w, h);

      if (debugMode) {
        console.log(`Hand ${handIndex + 1}: ${landmarks.length} landmarks.`);
      }
    });

    // Show the count of hands detected in the top-left corner.
    drawHandCount(handResults.length, h);
  }

  /**
   * drawSkeleton — draws lines between connected landmark pairs to form
   * the hand skeleton (finger bones and palm structure).
   *
   * @param {Array}  landmarks - Array of 21 { x, y, z } landmark objects.
   * @param {number} w         - Canvas/video width in pixels.
   * @param {number} h         - Canvas/video height in pixels.
   */
  function drawSkeleton(landmarks, w, h) {
    ctx.strokeStyle = SKELETON_COLOR;
    ctx.lineWidth   = 2;

    ctx.beginPath();
    HAND_CONNECTIONS.forEach(([a, b]) => {
      const ptA = landmarks[a];
      const ptB = landmarks[b];
      ctx.moveTo(ptA.x * w, ptA.y * h);
      ctx.lineTo(ptB.x * w, ptB.y * h);
    });
    ctx.stroke();
  }

  /**
   * drawLandmarks — draws a filled circle at each of the 21 hand landmarks.
   *
   * @param {Array}  landmarks - Array of 21 { x, y, z } landmark objects.
   * @param {number} w         - Canvas/video width in pixels.
   * @param {number} h         - Canvas/video height in pixels.
   */
  function drawLandmarks(landmarks, w, h) {
    ctx.fillStyle = LANDMARK_COLOR;

    ctx.beginPath();
    landmarks.forEach((point) => {
      const cx = point.x * w;
      const cy = point.y * h;
      ctx.moveTo(cx + DOT_RADIUS, cy);
      ctx.arc(cx, cy, DOT_RADIUS, 0, Math.PI * 2);
    });
    ctx.fill();
  }

  /**
   * drawHandCount — displays the number of hands detected in the corner.
   *
   * @param {number} count - Number of hands currently detected.
   * @param {number} h     - Canvas height in pixels.
   */
  function drawHandCount(count, h) {
    ctx.font      = "bold 16px monospace";
    ctx.fillStyle = count > 0 ? LANDMARK_COLOR : "#888";
    ctx.fillText(
      `Hands detected: ${count}`,
      10,
      h - 10
    );
  }

  } catch (err) {
    const message = formatError(err);
    lastErrorMessage = message;
    console.error("Fatal startup error:", err);
    updateDebugPanel(["startup: failed before tracking began"]);
    showStatus(`Startup failed: ${message}`, true);
  }
};
