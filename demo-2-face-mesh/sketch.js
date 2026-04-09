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
// This sketch demonstrates how to use MediaPipe FaceMesh to detect and track
// facial landmarks in real time using your webcam. When a face is detected,
// 468 landmark points are returned covering the full face surface — eyes,
// eyebrows, nose, lips, cheeks, and chin.
//
// The landmarks are drawn as coloured dots over the webcam feed, with
// different colours highlighting specific face regions.
//
// WHAT IS MediaPipe FACEMESH?
// ----------------------------
// MediaPipe FaceMesh is a machine-learning model from Google that analyses
// each video frame and returns the 3-D positions of 468 key points on any
// detected face. The model runs entirely in the browser — no data leaves
// your device.
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
//   Irises      — cyan        (#67e8f9)
//   General     — white       (rgba 200,200,200)
//
// The region index arrays below come from the MediaPipe FaceMesh
// canonical face model documentation.
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

window.onload = function () {

  if (debugMode) {
    console.log("Page loaded, initializing MediaPipe FaceMesh demo...");
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
  video.addEventListener("loadedmetadata", () => {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    if (debugMode) {
      console.log(`Canvas set to ${canvas.width}×${canvas.height}`);
    }
  });

  // ── MediaPipe FaceMesh setup ─────────────────────────────────────────────

  // The FaceMesh model is loaded from the jsDelivr CDN. locateFile tells
  // MediaPipe where to find its WebAssembly and model binary files.
  const faceMesh = new FaceMesh({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });

  faceMesh.setOptions({
    maxNumFaces: 1,               // track one face (increase for multi-face)
    refineLandmarks: true,        // enable iris landmarks (indices 468–477)
    minDetectionConfidence: 0.5,  // confidence needed to detect a face
    minTrackingConfidence: 0.5    // confidence needed to keep tracking it
  });

  // Store the most recent face results so drawFrame() can access them.
  let faceResults = [];

  // onResults is called by MediaPipe after every frame is processed.
  // results.multiFaceLandmarks is an array of faces, each containing
  // 468+ landmark objects with { x, y, z } properties.
  faceMesh.onResults(function (results) {
    if (results.multiFaceLandmarks) {
      faceResults = results.multiFaceLandmarks;
      if (debugMode) {
        console.log(`Detected ${faceResults.length} face(s).`);
      }
    } else {
      faceResults = [];
    }
    drawFrame();
  });

  if (debugMode) {
    console.log("MediaPipe FaceMesh initialised.");
  }

  // ── Camera (webcam) setup ────────────────────────────────────────────────

  const cam = new Camera(video, {
    onFrame: async () => {
      if (debugMode) {
        console.log("Sending frame to FaceMesh model...");
      }
      await faceMesh.send({ image: video });
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
   * detected face landmarks coloured by facial region.
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
    landmarks.forEach((point, index) => {
      const px = point.x * w;
      const py = point.y * h;

      // Determine colour based on region membership.
      let color  = REGION_COLORS.general;
      let radius = DOT_RADIUS;

      if (IRIS_INDICES.has(index)) {
        color  = REGION_COLORS.iris;
        radius = FEATURE_DOT_RADIUS;
      } else if (EYE_INDICES.has(index)) {
        color  = REGION_COLORS.eye;
        radius = FEATURE_DOT_RADIUS;
      } else if (EYEBROW_INDICES.has(index)) {
        color  = REGION_COLORS.eyebrow;
        radius = FEATURE_DOT_RADIUS;
      } else if (LIP_INDICES.has(index)) {
        color  = REGION_COLORS.lip;
        radius = FEATURE_DOT_RADIUS;
      } else if (NOSE_INDICES.has(index)) {
        color  = REGION_COLORS.nose;
        radius = FEATURE_DOT_RADIUS;
      }

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
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
