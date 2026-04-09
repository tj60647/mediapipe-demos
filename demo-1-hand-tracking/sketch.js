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
  const ctx    = canvas.getContext("2d");

  if (!video || !canvas) {
    console.error("Error: Could not find #video or #canvas in the DOM.");
    return;
  }

  // Set canvas size to match the video feed once dimensions are known.
  // The canvas is the same size as the video (640×480) so landmark
  // coordinates (which are normalised 0–1) map directly.
  video.addEventListener("loadedmetadata", () => {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    if (debugMode) {
      console.log(`Canvas set to ${canvas.width}×${canvas.height}`);
    }
  });

  // ── MediaPipe Hands setup ────────────────────────────────────────────────

  // The Hands model is loaded from the jsDelivr CDN. locateFile tells
  // MediaPipe where to find its WebAssembly and model binary files.
  const hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 2,          // detect up to two hands at once
    minDetectionConfidence: 0.5,  // confidence threshold to start tracking a hand
    minTrackingConfidence: 0.5,   // confidence threshold to keep tracking once found
    modelComplexity: 1       // 0 = lite (faster), 1 = full (more accurate)
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
    // Redraw the canvas whenever new results arrive.
    drawFrame();
  });

  if (debugMode) {
    console.log("MediaPipe Hands initialised.");
  }

  // ── Camera (webcam) setup ────────────────────────────────────────────────

  // The Camera utility captures frames from the webcam and calls onFrame
  // for each one. We pass the video element to Hands for processing.
  const cam = new Camera(video, {
    onFrame: async () => {
      if (debugMode) {
        console.log("Sending frame to Hands model...");
      }
      await hands.send({ image: video });
    },
    width: 640,
    height: 480
  });

  cam.start();

  if (debugMode) {
    console.log("Webcam started.");
  }

  // ── Drawing ──────────────────────────────────────────────────────────────

  /**
   * drawFrame — clears the canvas and redraws the webcam feed plus any
   * detected hand landmarks and skeleton connections.
   */
  function drawFrame() {
    // Clear and redraw the webcam image as the background.
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Draw each detected hand.
    handResults.forEach((landmarks, handIndex) => {
      drawSkeleton(landmarks);
      drawLandmarks(landmarks);

      if (debugMode) {
        console.log(`Hand ${handIndex + 1}: ${landmarks.length} landmarks.`);
      }
    });

    // Show the count of hands detected in the top-left corner.
    drawHandCount(handResults.length);
  }

  /**
   * drawSkeleton — draws lines between connected landmark pairs to form
   * the hand skeleton (finger bones and palm structure).
   *
   * @param {Array} landmarks - Array of 21 { x, y, z } landmark objects.
   */
  function drawSkeleton(landmarks) {
    ctx.strokeStyle = SKELETON_COLOR;
    ctx.lineWidth   = 2;

    HAND_CONNECTIONS.forEach(([a, b]) => {
      const ptA = landmarks[a];
      const ptB = landmarks[b];
      ctx.beginPath();
      ctx.moveTo(ptA.x * canvas.width,  ptA.y * canvas.height);
      ctx.lineTo(ptB.x * canvas.width,  ptB.y * canvas.height);
      ctx.stroke();
    });
  }

  /**
   * drawLandmarks — draws a filled circle at each of the 21 hand landmarks.
   *
   * @param {Array} landmarks - Array of 21 { x, y, z } landmark objects.
   */
  function drawLandmarks(landmarks) {
    ctx.fillStyle = LANDMARK_COLOR;

    landmarks.forEach((point) => {
      ctx.beginPath();
      ctx.arc(
        point.x * canvas.width,
        point.y * canvas.height,
        DOT_RADIUS, 0, Math.PI * 2
      );
      ctx.fill();
    });
  }

  /**
   * drawHandCount — displays the number of hands detected in the corner.
   *
   * @param {number} count - Number of hands currently detected.
   */
  function drawHandCount(count) {
    ctx.font      = "bold 16px monospace";
    ctx.fillStyle = count > 0 ? LANDMARK_COLOR : "#888";
    ctx.fillText(
      `Hands detected: ${count}`,
      10,
      canvas.height - 10
    );
  }
};
