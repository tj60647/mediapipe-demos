// =============================================================================
// File:    sketch.js
// Project: MediaPipe Demos — MDes Prototyping, CCA
// Demo:    2 — Face Mesh
//
// Authors: Copilot
//          Thomas J McLeish
// License: MIT — see LICENSE in the root of this repository
// =============================================================================
//
// PURPOSE
// -------
// This sketch demonstrates how to use MediaPipe FaceLandmarker (Tasks API) to
// detect and track facial landmarks in real time using your webcam. When a
// face is detected, 478 landmark points are returned covering the full face
// surface — eyes, eyebrows, nose, lips, cheeks, chin, and irises.
//
// The landmarks are drawn as coloured dots over the webcam feed, with
// different colours highlighting specific face regions.
//
// WHAT IS MediaPipe FACELANDMARKER?
// ----------------------------------
// MediaPipe FaceLandmarker (Tasks API) is the current generation of MediaPipe
// for the web. It uses a WebGL2 delegate by default (falls back to WASM),
// which runs reliably on Chrome for Android and Samsung Internet 14+.
//
// Each landmark has x, y, and z properties:
//   x, y — normalised position (0.0–1.0 relative to frame width/height)
//   z     — depth estimate (smaller = closer to camera)
//
// To convert to pixel coordinates: multiply x by canvas width, y by height.
//
// FACE REGIONS
// ------------
// This demo highlights key sub-regions using different colours:
//   Eyes        — light blue  (#93c5fd)
//   Eyebrows    — yellow      (#fde68a)
//   Lips        — coral       (#f87171)
//   Nose        — lilac       (#c084fc)
//   Irises      — cyan        (#67e8f9)  [indices 468–477]
//   General     — white       (rgba 200,200,200)
//
// The region index arrays below come from the MediaPipe FaceMesh
// canonical face model documentation.
//
// SHARED UTILITIES
// ----------------
// isMobile, showBrowserWarning(), showError(), and populateCameraSelect()
// come from shared/utils.js, loaded before this file.
//
// ANDROID / SAMSUNG NOTE
// ----------------------
// Three improvements have been applied for mobile compatibility:
//   1. Tasks API (WebGL2 delegate) instead of legacy WASM-only solution.
//   2. Mobile resolution cap (≤480×360 @ 20 fps) to prevent overloading the
//      GPU/CPU on mid-range Android devices.
//   3. Browser warning shown for incompatible mobile browsers (see utils.js).
//
// DEBUGGING
// ---------
// Set debugMode = true to log landmark counts and model events to the
// browser console. Open DevTools (F12) → Console tab to see the output.

// =============================================================================
// CONSTANTS
// =============================================================================

// Toggle console logging. Set to true to see per-frame debug output.
const debugMode = false;

// Radius of each landmark dot in pixels.
const DOT_RADIUS = 1.5;

// Radius for special-region dots (eyes, lips, etc.).
const FEATURE_DOT_RADIUS = 2.5;

// isMobile is defined in shared/utils.js.

// =============================================================================
// FACE REGION INDEX SETS
// =============================================================================
// Selected landmark indices for specific facial features, drawn in distinct
// colours so you can see which landmarks correspond to which regions.
// These indices come from the MediaPipe Face Mesh canonical face model.

const EYE_INDICES = new Set([
  // Right eye outline
  33, 7, 163, 144, 145, 153, 154, 155, 133,
  173, 157, 158, 159, 160, 161, 246,
  // Left eye outline
  362, 382, 381, 380, 374, 373, 390, 249,
  263, 466, 388, 387, 386, 385, 384, 398
]);

const EYEBROW_INDICES = new Set([
  // Right eyebrow
  70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
  // Left eyebrow
  336, 296, 334, 293, 300, 276, 283, 282, 295, 285
]);

const LIP_INDICES = new Set([
  // Outer lips
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
  291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
  // Inner lips
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
  308, 415, 310, 311, 312, 13, 82, 81, 80, 191
]);

const NOSE_INDICES = new Set([
  // Nose tip and bridge
  1, 2, 3, 4, 5, 6, 19, 94,
  // Right nostril contour
  48, 49, 51, 102, 115, 131, 134, 141, 220, 235, 236,
  // Left nostril contour
  279, 281, 330, 331, 344, 360, 420
]);

const IRIS_INDICES = new Set([
  // Right iris
  468, 469, 470, 471, 472,
  // Left iris
  473, 474, 475, 476, 477
]);

// =============================================================================
// COLOUR MAP
// =============================================================================
// Maps a feature category name to a CSS colour string.
const REGION_COLORS = {
  iris:     "#67e8f9",  // cyan
  eye:      "#93c5fd",  // light blue
  eyebrow:  "#fde68a",  // yellow
  lip:      "#f87171",  // coral / red
  nose:     "#c084fc",  // lilac / purple
  general:  "rgba(200,200,200,0.6)"  // subtle white
};

// =============================================================================
// ENTRY POINT
// =============================================================================

window.onload = async function () {

  if (debugMode) {
    console.log("Page loaded, initializing MediaPipe FaceLandmarker demo...");
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

  // ── MediaPipe FaceLandmarker setup (Tasks API) ───────────────────────────
  // FaceLandmarker uses a WebGL2 delegate by default (falls back to WASM),
  // which runs reliably on Chrome for Android and Samsung Internet 14+.
  // The standard face_landmarker.task model includes 478 landmarks:
  // 468 mesh points (indices 0–467) + 10 iris points (indices 468–477).

  const { FilesetResolver, FaceLandmarker } = mpVision;

  let faceLandmarker;
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numFaces: 1
    });
  } catch (err) {
    console.error("Failed to load FaceLandmarker:", err);
    showError(new Error(
      "Failed to load AI model. " +
      "On Android, open this page in Chrome for best compatibility. " +
      err.message
    ));
    return;
  }

  // Model ready — hide the loading indicator.
  const loadingEl = document.getElementById("loadingMessage");
  if (loadingEl) loadingEl.style.display = "none";

  // Store the most recent face results so drawFrame() can access them.
  let faceResults = [];

  if (debugMode) {
    console.log("MediaPipe FaceLandmarker initialised.");
  }

  // ── Camera management ────────────────────────────────────────────────────

  // Holds the active MediaStream so we can stop it when switching cameras.
  let currentStream = null;

  // Controls whether the frame loop is running.
  let frameLoopActive = false;

  // Frame counter used to throttle inference on mobile (see frameLoop).
  let inferenceFrameSkip = 0;

  /**
   * startCamera — opens the webcam with an optional specific device and
   * starts the per-frame loop that feeds images to the FaceLandmarker model.
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
   * frameLoop — runs FaceLandmarker on the current video frame and stores
   * the results for renderLoop to draw. detectForVideo is synchronous in
   * the Tasks API, so no await is needed.
   *
   * On mobile, inference runs every other frame (~30 fps) to reduce battery
   * drain. The renderLoop still draws at 60 fps using the last stored result.
   *
   * Stops automatically when frameLoopActive is false.
   */
  function frameLoop() {
    if (!frameLoopActive) return;
    if (video.readyState >= 2) {
      inferenceFrameSkip = (inferenceFrameSkip + 1) % 2;
      if (!isMobile || inferenceFrameSkip === 0) {
        if (debugMode) {
          console.log("Running FaceLandmarker on frame...");
        }
        try {
          const result = faceLandmarker.detectForVideo(video, performance.now());
          // result.faceLandmarks is an array of faces, each an array of {x,y,z}
          faceResults = result.faceLandmarks;
        } catch (err) {
          console.error("Face detection error:", err);
        }
      }
    }
    requestAnimationFrame(frameLoop);
  }

  // Start with the default camera, then populate the camera selector.
  // Enumerate regardless of whether the initial stream succeeded so that
  // users can try a different camera if the default one failed.
  await startCamera();
  const track    = currentStream ? currentStream.getVideoTracks()[0] : null;
  const activeId = track ? track.getSettings().deviceId : "";
  await populateCameraSelect(activeId, startCamera);

  // ── Drawing ──────────────────────────────────────────────────────────────

  /**
   * drawFrame — clears the canvas and redraws the webcam feed plus any
   * detected face landmarks coloured by facial region.
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

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(video, 0, 0, w, h);

    faceResults.forEach((landmarks, faceIndex) => {
      drawFaceLandmarks(landmarks, w, h);

      if (debugMode) {
        console.log(`Face ${faceIndex + 1}: ${landmarks.length} landmarks.`);
      }
    });

    drawFaceCount(faceResults.length, h);
    drawLegend(w, h);
  }

  /**
   * drawFaceLandmarks — draws a coloured dot at each landmark position.
   * The colour depends on which facial region the landmark belongs to.
   *
   * @param {Array}  landmarks - Array of { x, y, z } landmark objects.
   * @param {number} w         - Canvas/video width in pixels.
   * @param {number} h         - Canvas/video height in pixels.
   */
  function drawFaceLandmarks(landmarks, w, h) {
    // Classify each landmark into a color group to minimise canvas state changes.
    // All dots of the same color are drawn in a single beginPath/fill call.
    const groups = {
      iris:    { color: REGION_COLORS.iris,    radius: FEATURE_DOT_RADIUS, pts: [] },
      eye:     { color: REGION_COLORS.eye,     radius: FEATURE_DOT_RADIUS, pts: [] },
      eyebrow: { color: REGION_COLORS.eyebrow, radius: FEATURE_DOT_RADIUS, pts: [] },
      lip:     { color: REGION_COLORS.lip,     radius: FEATURE_DOT_RADIUS, pts: [] },
      nose:    { color: REGION_COLORS.nose,    radius: FEATURE_DOT_RADIUS, pts: [] },
      general: { color: REGION_COLORS.general, radius: DOT_RADIUS,         pts: [] }
    };

    landmarks.forEach((point, index) => {
      const entry = { x: point.x * w, y: point.y * h };
      if      (IRIS_INDICES.has(index))    groups.iris.pts.push(entry);
      else if (EYE_INDICES.has(index))     groups.eye.pts.push(entry);
      else if (EYEBROW_INDICES.has(index)) groups.eyebrow.pts.push(entry);
      else if (LIP_INDICES.has(index))     groups.lip.pts.push(entry);
      else if (NOSE_INDICES.has(index))    groups.nose.pts.push(entry);
      else                                 groups.general.pts.push(entry);
    });

    Object.values(groups).forEach(({ color, radius, pts }) => {
      if (pts.length === 0) return;
      ctx.fillStyle = color;
      ctx.beginPath();
      pts.forEach(({ x, y }) => {
        ctx.moveTo(x + radius, y);
        ctx.arc(x, y, radius, 0, Math.PI * 2);
      });
      ctx.fill();
    });
  }

  /**
   * drawFaceCount — displays the number of faces detected in the corner.
   *
   * @param {number} count - Number of faces currently detected.
   * @param {number} h     - Canvas height in pixels.
   */
  function drawFaceCount(count, h) {
    ctx.font      = "bold 16px monospace";
    ctx.fillStyle = count > 0 ? "#60a5fa" : "#888";
    ctx.fillText(
      `Faces detected: ${count}`,
      10,
      h - 10
    );
  }

  /**
   * drawLegend — draws a small colour key in the bottom-right corner
   * showing which colour corresponds to which facial region.
   *
   * @param {number} w - Canvas width in pixels.
   * @param {number} h - Canvas height in pixels.
   */
  function drawLegend(w, h) {
    const regions = [
      { label: "Eyes",     color: REGION_COLORS.eye },
      { label: "Eyebrows", color: REGION_COLORS.eyebrow },
      { label: "Lips",     color: REGION_COLORS.lip },
      { label: "Nose",     color: REGION_COLORS.nose },
      { label: "Irises",   color: REGION_COLORS.iris }
    ];

    const lineHeight = 18;
    const startX = w - 120;
    const startY = h - (regions.length * lineHeight) - 10;

    ctx.font = "12px monospace";

    regions.forEach((region, i) => {
      const y = startY + i * lineHeight;
      ctx.fillStyle = region.color;
      ctx.beginPath();
      ctx.arc(startX + 6, y + 4, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(region.label, startX + 16, y + 9);
    });
  }
};
