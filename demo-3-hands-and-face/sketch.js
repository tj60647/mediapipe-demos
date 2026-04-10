// =============================================================================
// File:    sketch.js
// Project: MediaPipe Demos — MDes Prototyping, CCA
// Demo:    3 — Hands and Face Combined
//
// Authors: Copilot
//          Thomas J McLeish
// License: MIT — see LICENSE in the root of this repository
// =============================================================================
//
// PURPOSE
// -------
// This sketch combines MediaPipe HandLandmarker and FaceLandmarker (Tasks API)
// to track both hand landmarks and facial landmarks simultaneously from your
// webcam.
//
// The canvas is split into two halves side-by-side:
//   LEFT  — the raw webcam feed, showing what the camera sees unmodified
//   RIGHT — the landmark overlay, showing the same frame with coloured dots
//           drawn at each detected landmark position
//
// Hand landmarks are drawn in green; face landmarks are drawn in red.
// This split view makes it easy to compare the raw image with the model output.
//
// ARCHITECTURE
// ------------
// Both models process the same video frame independently using detectForVideo,
// which is synchronous in the Tasks API. A shared drawResults() function reads
// both stored results and renders a single combined frame.
//
// The pipeline runs two independent loops:
//   frameLoop  — runs both landmarkers on each camera frame (synchronously)
//                and stores the latest results.
//   renderLoop — calls drawResults() on every requestAnimationFrame tick
//                (~60 fps), keeping the canvas smooth regardless of inference
//                speed. It always uses the most recently stored results.
//
// WHAT IS THE DIFFERENCE BETWEEN THE TWO HALVES?
// -----------------------------------------------
// LEFT (raw feed): ctx.drawImage() copies the webcam image as-is.
// RIGHT (landmark overlay): the same image is drawn again, offset by
//   video.videoWidth pixels along the x-axis, and then the landmark dots are
//   drawn on top of it. The x coordinates of each landmark are scaled to
//   video.videoWidth and then shifted right by another video.videoWidth —
//   placing them in the right half of the canvas.
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
// Set debugMode = true to log model events and landmark counts to the
// browser console. Open DevTools (F12) → Console tab to view the output.
// Set it back to false for smoother performance during normal use.

// =============================================================================
// CONSTANTS
// =============================================================================

// Toggle console logging. Set to true to see per-frame debug output.
const debugMode = false;

// Dot radius for hand landmarks (px).
const HAND_DOT_RADIUS = 5;

// Dot radius for face landmarks (px).
const FACE_DOT_RADIUS = 3;

// Colour for hand landmarks.
const HAND_COLOR = "#4ade80";  // green

// Colour for hand skeleton connections.
const HAND_SKELETON_COLOR = "rgba(74, 222, 128, 0.5)";

// Colour for face landmarks.
const FACE_COLOR = "#f87171";  // red / coral

// Default hand-tracking mode for Demo 3. Can be switched live in the UI.
let desiredMaxHands = 2;

// True when the page is loaded on a touch-capable mobile device.
const isMobile = navigator.maxTouchPoints > 1;

// =============================================================================
// HAND CONNECTIONS
// =============================================================================
// Each pair [a, b] connects landmark index a to landmark index b to form
// the hand skeleton. See Demo 1 for a detailed description.
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

// =============================================================================
// ENTRY POINT
// =============================================================================

window.onload = async function () {

  if (debugMode) {
    console.log("Page loaded, initializing combined HandLandmarker + FaceLandmarker demo...");
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
  showBrowserWarning();

  // ── MediaPipe Tasks API setup ────────────────────────────────────────────
  // Both landmarkers share the same FilesetResolver (WASM loader), so we
  // only resolve the fileset once.

  const { FilesetResolver, HandLandmarker, FaceLandmarker } = mpVision;

  let handLandmarker;
  let faceLandmarker;
  // Cache the resolved vision fileset so setupHandCountToggle can reuse it
  // without re-initializing the WASM loader on every toggle.
  let sharedVision;
  try {
    sharedVision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    handLandmarker = await HandLandmarker.createFromOptions(sharedVision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: desiredMaxHands
    });

    faceLandmarker = await FaceLandmarker.createFromOptions(sharedVision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numFaces: 1
    });
  } catch (err) {
    console.error("Failed to load AI models:", err);
    showError(new Error(
      "Failed to load AI models. " +
      "On Android, open this page in Chrome for best compatibility. " +
      err.message
    ));
    return;
  }

  // Arrays that store the most recent results from each model.
  let handResults = [];
  let faceResults = [];

  if (debugMode) {
    console.log("MediaPipe HandLandmarker and FaceLandmarker initialised.");
  }

  /**
   * setupHandCountToggle — wires the 1/2-hands selector to the HandLandmarker
   * so users can switch tracking mode without reloading the page.
   * Recreates the HandLandmarker with the new numHands value, reusing the
   * cached sharedVision fileset to avoid redundant WASM loader initialization.
   */
  function setupHandCountToggle() {
    const select = document.getElementById("handCountSelect");
    if (!select) return;

    select.value = String(desiredMaxHands);

    select.onchange = async () => {
      const nextMaxHands = Number.parseInt(select.value, 10) === 2 ? 2 : 1;
      desiredMaxHands = nextMaxHands;

      try {
        const updated = await HandLandmarker.createFromOptions(sharedVision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: desiredMaxHands
        });
        handLandmarker.close();
        handLandmarker = updated;
        if (debugMode) {
          console.log(`Updated numHands to ${desiredMaxHands}.`);
        }
      } catch (err) {
        console.error("Failed to update hand tracking mode:", err);
      }
    };
  }

  setupHandCountToggle();

  // ── Camera management ────────────────────────────────────────────────────

  // Holds the active MediaStream so we can stop it when switching cameras.
  let currentStream = null;

  // Controls whether the frame loop is running.
  let frameLoopActive = false;

  /**
   * startCamera — opens the webcam with an optional specific device and
   * starts the per-frame loop that feeds images to both models.
   *
   * On mobile devices the resolution is capped to ≤480×360 at ≤20 fps to
   * prevent overloading mid-range Android GPU/CPU.
   *
   * @param {string} [deviceId] — exact device ID to open, or omit / pass ""
   *                              to let the browser choose the default camera.
   */
  async function startCamera(deviceId) {
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

    // The canvas is twice the video width for the side-by-side layout.
    video.onloadedmetadata = () => {
      canvas.width  = video.videoWidth * 2;
      canvas.height = video.videoHeight;
      if (debugMode) {
        console.log(
          `Canvas set to ${canvas.width}×${canvas.height} (side-by-side layout)`
        );
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
   * both ML models run slower than 60 fps.
   */
  function renderLoop() {
    if (!frameLoopActive) return;
    drawResults();
    requestAnimationFrame(renderLoop);
  }

  /**
   * frameLoop — runs both HandLandmarker and FaceLandmarker on the current
   * video frame on every animation tick. detectForVideo is synchronous in the
   * Tasks API. Stops automatically when frameLoopActive is false.
   */
  function frameLoop() {
    if (!frameLoopActive) return;
    if (video.readyState >= 2) {
      try {
        const nowMs = performance.now();
        if (debugMode) {
          console.log("Running HandLandmarker and FaceLandmarker on frame...");
        }
        const handResult = handLandmarker.detectForVideo(video, nowMs);
        handResults = handResult.landmarks;

        const faceResult = faceLandmarker.detectForVideo(video, nowMs);
        faceResults = faceResult.faceLandmarks;
      } catch (err) {
        console.error("Frame processing error:", err);
      }
    }
    requestAnimationFrame(frameLoop);
  }

  /**
   * populateCameraSelect — enumerates video-input devices and fills the
   * on-page <select>. Requires camera permission to have been granted so
   * that device labels are populated. The wrapper is revealed only when
   * more than one camera is available.
   *
   * @param {string} activeDeviceId — the deviceId currently in use.
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
   * drawResults — renders one complete frame on the canvas.
   *
   * LEFT HALF:  raw webcam feed (no overlay)
   * RIGHT HALF: webcam feed + hand landmarks (green) + face landmarks (red)
   *
   * Coordinate conversion for the right half:
   *   pixelX = video.videoWidth  + point.x * video.videoWidth
   *   pixelY =                     point.y * video.videoHeight
   * The extra video.videoWidth offset shifts every landmark into the right half.
   */
  function drawResults() {
    const w = video.videoWidth;
    const h = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // ── Left half: raw feed ─────────────────────────────────────────────
    ctx.drawImage(video, 0, 0, w, h);

    // ── Right half: background (same feed, shifted right) ───────────────
    ctx.drawImage(video, w, 0, w, h);

    // Label each half.
    ctx.font      = "bold 14px monospace";
    ctx.fillStyle = "#aaa";
    ctx.fillText("RAW", 10, 20);
    ctx.fillText("LANDMARKS", w + 10, 20);

    // ── Right half: hand landmarks ──────────────────────────────────────
    handResults.forEach((handLandmarks, index) => {
      drawHandSkeleton(handLandmarks, w, h);
      drawHandDots(handLandmarks, w, h);

      if (debugMode) {
        console.log(`Hand ${index + 1}: ${handLandmarks.length} landmarks.`);
      }
    });

    // ── Right half: face landmarks ──────────────────────────────────────
    faceResults.forEach((faceLandmarks, index) => {
      drawFaceDots(faceLandmarks, w, h);

      if (debugMode) {
        console.log(`Face ${index + 1}: ${faceLandmarks.length} landmarks.`);
      }
    });

    // ── Stats overlay ───────────────────────────────────────────────────
    drawStats(w, h);
  }

  /**
   * drawHandSkeleton — draws skeleton lines for one hand in the right half.
   *
   * @param {Array}  landmarks - 21 { x, y, z } landmark objects for the hand.
   * @param {number} w         - Video width (used to offset into the right half).
   * @param {number} h         - Video height.
   */
  function drawHandSkeleton(landmarks, w, h) {
    ctx.strokeStyle = HAND_SKELETON_COLOR;
    ctx.lineWidth   = 2;

    ctx.beginPath();
    HAND_CONNECTIONS.forEach(([a, b]) => {
      const ptA = landmarks[a];
      const ptB = landmarks[b];
      ctx.moveTo(w + ptA.x * w, ptA.y * h);
      ctx.lineTo(w + ptB.x * w, ptB.y * h);
    });
    ctx.stroke();
  }

  /**
   * drawHandDots — draws landmark dots for one hand in the right half.
   *
   * @param {Array}  landmarks - 21 { x, y, z } landmark objects.
   * @param {number} w         - Video width.
   * @param {number} h         - Video height.
   */
  function drawHandDots(landmarks, w, h) {
    ctx.fillStyle = HAND_COLOR;

    ctx.beginPath();
    landmarks.forEach((point) => {
      const cx = w + point.x * w;
      const cy = point.y * h;
      ctx.moveTo(cx + HAND_DOT_RADIUS, cy);
      ctx.arc(cx, cy, HAND_DOT_RADIUS, 0, Math.PI * 2);
    });
    ctx.fill();
  }

  /**
   * drawFaceDots — draws landmark dots for one face in the right half.
   *
   * @param {Array}  landmarks - 468 { x, y, z } landmark objects.
   * @param {number} w         - Video width.
   * @param {number} h         - Video height.
   */
  function drawFaceDots(landmarks, w, h) {
    ctx.fillStyle = FACE_COLOR;

    ctx.beginPath();
    landmarks.forEach((point) => {
      const cx = w + point.x * w;
      const cy = point.y * h;
      ctx.moveTo(cx + FACE_DOT_RADIUS, cy);
      ctx.arc(cx, cy, FACE_DOT_RADIUS, 0, Math.PI * 2);
    });
    ctx.fill();
  }

  /**
   * drawStats — shows a count of detected hands and faces in the bottom-left
   * corner of the right half.
   *
   * @param {number} w - Video width (used to offset into the right half).
   * @param {number} h - Video height.
   */
  function drawStats(w, h) {
    ctx.font = "bold 14px monospace";

    // Hands count
    ctx.fillStyle = handResults.length > 0 ? HAND_COLOR : "#888";
    ctx.fillText(`Hands: ${handResults.length}`, w + 10, h - 28);

    // Faces count
    ctx.fillStyle = faceResults.length > 0 ? FACE_COLOR : "#888";
    ctx.fillText(`Faces: ${faceResults.length}`, w + 10, h - 10);
  }

  /**
   * showBrowserWarning — reveals the #browserWarning banner when the page
   * is opened in Samsung Internet or a non-Chromium mobile browser.
   * These browsers may restrict the WebGL/WASM backend used by the model.
   */
  function showBrowserWarning() {
    const ua = navigator.userAgent;
    const isSamsung         = /SamsungBrowser/i.test(ua);
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
