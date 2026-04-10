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
// This sketch combines MediaPipe Hands and MediaPipe FaceMesh to track both
// hand landmarks and facial landmarks simultaneously from your webcam.
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
// Both models process the same video frame independently. The results of each
// detectForVideo call are stored and read by a shared drawResults() function
// that renders a single combined frame.
//
// The pipeline runs two independent loops:
//   frameLoop  — calls detectForVideo on both landmarkers sequentially for
//                each camera frame at the model inference rate.
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
    console.log("Page loaded, initializing combined Hand + Face Landmarker demo...");
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
  statusBanner.style.position = "fixed";
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
  document.body.appendChild(statusBanner);

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

  showStatus("Loading MediaPipe runtime...");

  // ── MediaPipe Tasks Vision ───────────────────────────────────────────────

  const { HandLandmarker, FaceLandmarker, FilesetResolver } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs"
  );

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm"
  );

  // ── Helper: create a HandLandmarker with the given hand count ────────────
  // Used at startup and when the user changes the Hands selector live.

  async function createHandLandmarker(numHands) {
    try {
      showStatus("Loading hand model (GPU)...");
      return await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker" +
            "/hand_landmarker/float16/latest/hand_landmarker.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands
      });
    } catch (gpuErr) {
      console.warn("GPU delegate unavailable, retrying with CPU:", gpuErr);
      showStatus("Loading hand model (CPU fallback)...");
      return HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker" +
            "/hand_landmarker/float16/latest/hand_landmarker.task",
          delegate: "CPU"
        },
        runningMode: "VIDEO",
        numHands
      });
    }
  }

  // ── MediaPipe HandLandmarker setup ───────────────────────────────────────

  let handLandmarker = await createHandLandmarker(desiredMaxHands);

  if (debugMode) {
    console.log("HandLandmarker ready.");
  }

  // ── MediaPipe FaceLandmarker setup ───────────────────────────────────────

  let faceLandmarker;
  try {
    showStatus("Loading face model (GPU)...");
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker" +
          "/face_landmarker/float16/latest/face_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numFaces: 1
    });
  } catch (gpuErr) {
    console.warn("GPU delegate unavailable, retrying with CPU:", gpuErr);
    showStatus("Loading face model (CPU fallback)...");
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker" +
          "/face_landmarker/float16/latest/face_landmarker.task",
        delegate: "CPU"
      },
      runningMode: "VIDEO",
      numFaces: 1
    });
  }

  if (debugMode) {
    console.log("FaceLandmarker ready.");
  }

  // Arrays that store the most recent results from each model.
  let handResults = [];
  let faceResults = [];

  // Each model keeps its own monotonic timestamp for detectForVideo.
  let lastTimestampHand = -1;
  let lastTimestampFace = -1;

  /**
   * setupHandCountToggle — wires the 1/2-hands selector so users can switch
   * tracking mode without reloading the page. The Tasks API does not support
   * mutating options after creation, so the HandLandmarker is recreated with
   * the new numHands value.
   */
  function setupHandCountToggle() {
    const select = document.getElementById("handCountSelect");
    if (!select) return;

    select.value = String(desiredMaxHands);

    select.onchange = async () => {
      const parsed = Number.parseInt(select.value, 10);
      desiredMaxHands = parsed === 2 ? 2 : 1;

      // Pause inference while recreating the landmarker.
      const wasRunning = frameLoopActive;
      frameLoopActive = false;
      handResults = [];
      lastTimestampHand = -1;

      try {
        if (handLandmarker) handLandmarker.close();
        handLandmarker = await createHandLandmarker(desiredMaxHands);
        if (debugMode) {
          console.log(`Updated numHands to ${desiredMaxHands}.`);
        }
      } catch (err) {
        console.error("Failed to update hand tracking mode:", err);
      }

      frameLoopActive = wasRunning;
      if (frameLoopActive) {
        requestAnimationFrame(frameLoop);
      }
    };
  }

  setupHandCountToggle();

  // ── Camera management ────────────────────────────────────────────────────

  // Holds the active MediaStream so we can stop it when switching cameras.
  let currentStream = null;

  // Controls whether the frame loop is running.
  let frameLoopActive = false;

  // Stop loops and release camera tracks when leaving the page.
  window.addEventListener("beforeunload", () => {
    frameLoopActive = false;
    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
    }
  });

  /**
   * startCamera — opens the webcam with an optional specific device and
   * starts the per-frame loop that feeds images to both models.
   *
   * @param {string} [deviceId] — exact device ID to open, or omit / pass ""
   *                              to let the browser choose the default camera.
   */
  async function startCamera(deviceId) {
    frameLoopActive = false;
    showStatus("Requesting camera access...");

    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
      currentStream = null;
    }

    const videoConstraints = { width: 640, height: 480 };
    if (deviceId) videoConstraints.deviceId = { exact: deviceId };

    try {
      currentStream = await navigator.mediaDevices.getUserMedia(
        { video: videoConstraints }
      );
    } catch (err) {
      console.error("Could not open camera:", err);
      showStatus("Camera access failed. Check permissions and reload.", true);
      return;
    }

    video.srcObject = currentStream;
    video.muted = true;
    video.autoplay = true;
    video.setAttribute("playsinline", "");

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
   * frameLoop — runs both landmarkers on the current video frame on every
   * animation tick. detectForVideo is synchronous. Both models receive the
   * same frame and their results are stored independently for the render loop.
   */
  function frameLoop() {
    if (!frameLoopActive) return;
    if (video.readyState >= 2) {
      clearStatus();
      const tsHand = performance.now();
      if (tsHand > lastTimestampHand) {
        lastTimestampHand = tsHand;
        const handResult = handLandmarker.detectForVideo(video, tsHand);
        handResults = handResult.landmarks ?? [];
      }

      const tsFace = performance.now();
      if (tsFace > lastTimestampFace) {
        lastTimestampFace = tsFace;
        const faceResult = faceLandmarker.detectForVideo(video, tsFace);
        faceResults = faceResult.faceLandmarks ?? [];
      }

      if (debugMode) {
        console.log("Running HandLandmarker and FaceLandmarker on current frame...");
        console.log(`Hands: ${handResults.length}, Faces: ${faceResults.length}`);
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

  // Initialise the camera. Enumerate devices afterwards so that the camera
  // selector shows real labels (requires permission to have been granted).
  await startCamera();
  const track    = currentStream ? currentStream.getVideoTracks()[0] : null;
  const activeId = track ? track.getSettings().deviceId : "";
  populateCameraSelect(activeId);

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
};
