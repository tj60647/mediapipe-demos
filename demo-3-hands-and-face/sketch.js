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
// Both models process the same video frame independently. Each model has its
// own onResults callback that stores the latest results. A shared drawResults()
// function reads both stored results and renders a single combined frame.
//
// To avoid drawing a partial frame (hands updated but face not yet), the
// Camera's onFrame callback calls drawResults() once per frame, after sending
// the image to both models. The update flags (newHand, newFace) guard against
// drawing an entirely empty frame before either model has returned results.
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

window.onload = function () {

  if (debugMode) {
    console.log("Page loaded, initializing combined Hands + FaceMesh demo...");
  }

  // ── DOM elements ────────────────────────────────────────────────────────

  const video  = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx    = canvas.getContext("2d");

  if (!video || !canvas) {
    console.error("Error: Could not find #video or #canvas in the DOM.");
    return;
  }

  // ── MediaPipe Hands setup ────────────────────────────────────────────────

  const hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 2,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
    modelComplexity: 1
  });

  // Flags that tell drawResults() when fresh data is available from each model.
  let newHand = false;
  let newFace = false;

  // Arrays that store the most recent results from each model.
  let handResults = [];
  let faceResults = [];

  hands.onResults(function (results) {
    handResults = results.multiHandLandmarks || [];
    newHand = true;
    if (debugMode) {
      console.log(`Hands: ${handResults.length} detected.`);
    }
  });

  if (debugMode) {
    console.log("MediaPipe Hands initialised.");
  }

  // ── MediaPipe FaceMesh setup ─────────────────────────────────────────────

  const faceMesh = new FaceMesh({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });

  faceMesh.setOptions({
    maxNumFaces: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  faceMesh.onResults(function (results) {
    faceResults = results.multiFaceLandmarks || [];
    newFace = true;
    if (debugMode) {
      console.log(`Faces: ${faceResults.length} detected.`);
    }
  });

  if (debugMode) {
    console.log("MediaPipe FaceMesh initialised.");
  }

  // ── Camera management ────────────────────────────────────────────────────

  // Holds the active MediaStream so we can stop it when switching cameras.
  let currentStream = null;

  // Controls whether the frame loop is running.
  let frameLoopActive = false;

  /**
   * startCamera — opens the webcam with an optional specific device and
   * starts the per-frame loop that feeds images to both models.
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

    const videoConstraints = { width: 640, height: 480 };
    if (deviceId) videoConstraints.deviceId = { exact: deviceId };

    try {
      currentStream = await navigator.mediaDevices.getUserMedia(
        { video: videoConstraints }
      );
    } catch (err) {
      console.error("Could not open camera:", err);
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

    video.play();

    if (debugMode) {
      console.log("Webcam started.");
    }

    frameLoopActive = true;
    requestAnimationFrame(frameLoop);
  }

  /**
   * frameLoop — sends the current video frame to both MediaPipe models on
   * every animation tick. Stops automatically when frameLoopActive is false.
   */
  async function frameLoop() {
    if (!frameLoopActive) return;
    if (video.readyState >= 2) {
      if (debugMode) {
        console.log("Sending frame to both models...");
      }
      await hands.send({ image: video });
      await faceMesh.send({ image: video });

      if (newHand || newFace) {
        drawResults();
        newHand = false;
        newFace = false;
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
  startCamera().then(() => {
    if (!currentStream) return;
    const track    = currentStream.getVideoTracks()[0];
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

    HAND_CONNECTIONS.forEach(([a, b]) => {
      const ptA = landmarks[a];
      const ptB = landmarks[b];
      ctx.beginPath();
      ctx.moveTo(w + ptA.x * w, ptA.y * h);
      ctx.lineTo(w + ptB.x * w, ptB.y * h);
      ctx.stroke();
    });
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

    landmarks.forEach((point) => {
      ctx.beginPath();
      ctx.arc(w + point.x * w, point.y * h, HAND_DOT_RADIUS, 0, Math.PI * 2);
      ctx.fill();
    });
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

    landmarks.forEach((point) => {
      ctx.beginPath();
      ctx.arc(w + point.x * w, point.y * h, FACE_DOT_RADIUS, 0, Math.PI * 2);
      ctx.fill();
    });
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
