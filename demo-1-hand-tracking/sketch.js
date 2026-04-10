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
// This sketch demonstrates how to use MediaPipe HandLandmarker (Tasks API) to
// detect and track hand landmarks in real time using your webcam. Up to two
// hands are tracked simultaneously. For each hand, 21 landmark points are
// returned — one per knuckle, fingertip, and palm joint.
//
// Landmarks are drawn as green dots directly over the webcam feed. The hand
// skeleton (connections between adjacent joints) is drawn in a lighter green.
//
// WHAT IS THE MEDIAPIPE TASKS API?
// ---------------------------------
// This demo uses @mediapipe/tasks-vision (HandLandmarker), the current
// generation of MediaPipe for the web. It uses a WebGL2 delegate by default
// (falling back to WASM), which runs reliably on Chrome for Android and
// Samsung Internet 14+.
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
// ANDROID / SAMSUNG NOTE
// ----------------------
// Three improvements have been applied for mobile compatibility:
//   1. Tasks API (WebGL2 delegate) instead of legacy WASM-only solution.
//   2. Mobile resolution cap (≤480×360 @ 20 fps) to prevent overloading the
//      GPU/CPU on mid-range Android devices.
//   3. Browser warning shown when Samsung Internet or a non-Chromium mobile
//      browser is detected, recommending Chrome for Android.
//
// DEBUGGING
// ---------
// Set debugMode = true to log landmark counts and frame events to the
// browser console. Open DevTools (F12) → Console tab to see the output.
// Set it back to false for smoother performance during normal use.

// =============================================================================
// CONSTANTS
// =============================================================================

// Toggle console logging. Set to true to see per-frame debug output.
const debugMode = false;

// Colour for landmark dots (green).
const LANDMARK_COLOR = "#4ade80";

// Colour for the skeleton lines connecting adjacent landmarks.
const SKELETON_COLOR = "rgba(74, 222, 128, 0.5)";

// Radius of each landmark dot in pixels.
const DOT_RADIUS = 6;

// True when the page is loaded on a touch-capable mobile device.
const isMobile = navigator.maxTouchPoints > 1;

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
    console.log("Page loaded, initializing MediaPipe HandLandmarker demo...");
  }

  // ── DOM elements ────────────────────────────────────────────────────────

  const video  = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx    = canvas.getContext("2d", { alpha: false });

  if (!video || !canvas) {
    console.error("Error: Could not find #video or #canvas in the DOM.");
    return;
  }

  // ── Browser compatibility warning ───────────────────────────────────────
  // Shown when Samsung Internet or a non-Chromium mobile browser is detected.
  // These browsers may block the WebAssembly/WebGL delegate used by the model.
  showBrowserWarning();

  // ── MediaPipe HandLandmarker setup (Tasks API) ───────────────────────────
  // HandLandmarker uses a WebGL2 delegate by default (falls back to WASM),
  // which runs reliably on Chrome for Android and Samsung Internet 14+.

  const { FilesetResolver, HandLandmarker } = mpVision;

  let handLandmarker;
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2
    });
  } catch (err) {
    console.error("Failed to load HandLandmarker:", err);
    showError(new Error(
      "Failed to load AI model. " +
      "On Android, open this page in Chrome for best compatibility. " +
      err.message
    ));
    return;
  }

  // Store the most recent hand results so drawFrame() can access them.
  let handResults = [];

  if (debugMode) {
    console.log("MediaPipe HandLandmarker initialised.");
  }

  // ── Camera management ────────────────────────────────────────────────────

  // Holds the active MediaStream so we can stop it when switching cameras.
  let currentStream = null;

  // Controls whether the frame loop is running.
  let frameLoopActive = false;

  /**
   * startCamera — opens the webcam with an optional specific device and
   * starts the per-frame loop that feeds images to the HandLandmarker model.
   *
   * On mobile devices the resolution is capped to ≤480×360 at ≤20 fps to
   * prevent overloading mid-range Android GPU/CPU.
   *
   * @param {string} [deviceId] — exact device ID to open, or omit / pass ""
   *                              to let the browser choose the default camera.
   */
  async function startCamera(deviceId) {
    // Halt the previous frame loop before replacing the stream.
    frameLoopActive = false;

    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
      currentStream = null;
    }

    // Cap resolution and frame rate on mobile to prevent the WASM/WebGL
    // backend from being overloaded by full-resolution camera streams.
    const videoConstraints = isMobile ? {
      width:      { ideal: 320, max: 480 },
      height:     { ideal: 240, max: 360 },
      frameRate:  { ideal: 15, max: 20 },
      facingMode: { ideal: "user" }
    } : {
      width:      { ideal: 640 },
      height:     { ideal: 480 },
      facingMode: { ideal: "user" }
    };
    if (deviceId) videoConstraints.deviceId = { exact: deviceId };

    try {
      currentStream = await navigator.mediaDevices.getUserMedia(
        { video: videoConstraints }
      );
    } catch (err) {
      console.error("Could not open camera:", err);
      showError(err);
      return;
    }

    video.srcObject = currentStream;

    // Keep canvas dimensions in sync with the new stream's resolution.
    video.onloadedmetadata = () => {
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
      if (debugMode) {
        console.log(`Canvas set to ${canvas.width}×${canvas.height}`);
      }
    };

    try {
      await video.play();
    } catch (playErr) {
      console.error("video.play() rejected:", playErr);
      showError(playErr);
      return;
    }

    if (debugMode) {
      console.log("Webcam started.");
    }

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
   * animation tick and stores the results for renderLoop to draw.
   * detectForVideo is synchronous in the Tasks API, so no await is needed.
   * Stops automatically when frameLoopActive is false.
   */
  function frameLoop() {
    if (!frameLoopActive) return;
    if (video.readyState >= 2) {
      if (debugMode) {
        console.log("Running HandLandmarker on frame...");
      }
      try {
        const result = handLandmarker.detectForVideo(video, performance.now());
        // result.landmarks is an array of hands, each an array of 21 {x,y,z}
        handResults = result.landmarks;
      } catch (err) {
        console.error("Hand detection error:", err);
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

  // Start with the default camera, then enumerate devices for the selector.
  // Enumerate regardless of whether the initial stream succeeded so that
  // users can try a different camera if the default one failed.
  startCamera().then(() => {
    const track    = currentStream ? currentStream.getVideoTracks()[0] : null;
    const activeId = track ? track.getSettings().deviceId : "";
    populateCameraSelect(activeId);
  });

  // ── Drawing ──────────────────────────────────────────────────────────────

  /**
   * drawFrame — clears the canvas and redraws the webcam feed plus any
   * detected hand landmarks and skeleton connections.
   *
   * Uses video.videoWidth/Height directly so drawing is correct even if
   * the loadedmetadata event fires after the first detection result.
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

  /**
   * showBrowserWarning — reveals the #browserWarning banner when the page
   * is opened in Samsung Internet or a non-Chromium mobile browser.
   * These browsers may restrict the WebGL/WASM backend used by the model.
   */
  function showBrowserWarning() {
    const ua = navigator.userAgent;
    const isSamsung       = /SamsungBrowser/i.test(ua);
    const isMobileNonChrome = /Android|iPhone|iPad/i.test(ua) && !/Chrome\/[0-9]/i.test(ua);
    if (!isSamsung && !isMobileNonChrome) return;
    const el = document.getElementById("browserWarning");
    if (!el) return;
    el.textContent =
      "For best results on Android, open this page in Chrome. " +
      "Samsung Internet and other non-Chromium browsers may not support " +
      "the AI models used in these demos.";
    el.style.display = "block";
  }

  /**
   * showError — displays a human-readable error on the page so that
   * mobile users who cannot open DevTools can still see what went wrong.
   *
   * @param {Error} err - The error to display.
   */
  function showError(err) {
    const el = document.getElementById("errorMessage");
    if (!el) return;
    el.textContent = err.name === "NotAllowedError"
      ? "Camera access was denied. Please allow camera permission and reload."
      : `Error: ${err.message || err.name}. Try reloading or use HTTPS.`;
    el.style.display = "block";
  }
};
