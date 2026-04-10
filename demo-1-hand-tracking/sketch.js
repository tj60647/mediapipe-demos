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
// WHAT IS MediaPipe HANDS?
// ------------------------
// MediaPipe Hands is a machine-learning model from Google that analyses each
// video frame and returns the 3-D positions of 21 key points on each hand it
// detects. The model runs entirely in the browser — no data ever leaves your
// device.
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
// WHAT IS THE CAMERA UTILITY?
// ---------------------------
// @mediapipe/camera_utils provides a Camera class that continuously captures
// frames from your webcam and calls an onFrame callback. In that callback we
// pass the current video frame to the Hands model for processing. This loop
// runs at the camera's frame rate (typically 30 fps).
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

window.onload = function () {

  if (debugMode) {
    console.log("Page loaded, initializing MediaPipe Hands demo...");
  }

  // ── DOM elements ────────────────────────────────────────────────────────

  const video  = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx    = canvas.getContext("2d", { alpha: false });

  if (!video || !canvas) {
    console.error("Error: Could not find #video or #canvas in the DOM.");
    return;
  }

  // ── MediaPipe Hands setup ────────────────────────────────────────────────

  // The Hands model is loaded from the jsDelivr CDN. locateFile tells
  // MediaPipe where to find its WebAssembly and model binary files.
  const hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
  });

  hands.setOptions({
    maxNumHands: 2,          // detect up to two hands at once
    minDetectionConfidence: 0.5,  // confidence threshold to start tracking a hand
    minTrackingConfidence: 0.5,   // confidence threshold to keep tracking once found
    modelComplexity: 0       // 0 = lite (faster), 1 = full (more accurate)
  });

  // Store the most recent hand results so draw() can access them.
  let handResults = [];

  // onResults is called by MediaPipe after every frame is processed.
  // results.multiHandLandmarks is an array of hands, each containing
  // 21 landmark objects with { x, y, z } properties.
  hands.onResults(function (results) {
    if (results.multiHandLandmarks) {
      handResults = results.multiHandLandmarks;
      if (debugMode) {
        console.log(`Detected ${handResults.length} hand(s).`);
      }
    } else {
      // No hands in this frame — clear stored results.
      handResults = [];
    }
  });

  if (debugMode) {
    console.log("MediaPipe Hands initialised.");
  }

  // ── Camera management ────────────────────────────────────────────────────

  // Holds the active MediaStream so we can stop it when switching cameras.
  let currentStream = null;

  // Controls whether the frame loop is running.
  let frameLoopActive = false;

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

    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
      currentStream = null;
    }

    const videoConstraints = {
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
   * frameLoop — sends the current video frame to the Hands model on every
   * animation tick. Stops automatically when frameLoopActive is false.
   */
  async function frameLoop() {
    if (!frameLoopActive) return;
    if (video.readyState >= 2) {
      if (debugMode) {
        console.log("Sending frame to Hands model...");
      }
      await hands.send({ image: video });
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
   * the loadedmetadata event fires after the first onResults callback.
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
   * showError — displays a human-readable camera error on the page so that
   * mobile users who cannot open DevTools can still see what went wrong.
   *
   * @param {Error} err - The error thrown by getUserMedia.
   */
  function showError(err) {
    const el = document.getElementById("errorMessage");
    if (!el) return;
    el.textContent = err.name === "NotAllowedError"
      ? "Camera access was denied. Please allow camera permission and reload."
      : `Camera error: ${err.message || err.name}. Try reloading or use HTTPS.`;
    el.style.display = "block";
  }
};
