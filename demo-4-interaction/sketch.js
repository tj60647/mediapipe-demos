// =============================================================================
// File:    sketch.js
// Project: MediaPipe Demos — MDes Prototyping, CCA
// Demo:    4 — Interaction Basics
//
// Authors: Copilot
//          Thomas J McLeish
// License: MIT — see LICENSE in the root of this repository
// =============================================================================
//
// PURPOSE
// -------
// This sketch shows how to turn MediaPipe landmark coordinates into
// interactive inputs rather than just visual markers. The same webcam data
// used to draw dots in Demos 1–3 is used here to drive four distinct
// interaction patterns:
//
//   1. DISTANCE — Euclidean pixel distance between the index fingertip
//                 (hand landmark 8) and the nose tip (face landmark 4).
//                 A dashed line connects the two points and changes colour
//                 from green (far) to red (close). A vertical bar on the
//                 right edge shows the distance as a level.
//
//   2. ZONES    — The frame is divided into three equal vertical columns:
//                 LEFT, CENTRE, and RIGHT. The column containing the index
//                 fingertip is highlighted. This converts a continuous
//                 coordinate into a discrete on/off state — the simplest
//                 form of interaction logic.
//
//   3. COUNT    — Each of the four non-thumb fingers is checked: if its
//                 tip y-coordinate is above its PIP (middle) joint,
//                 the finger is considered raised. The count changes the
//                 colour of all hand dots from red (0) through to green (4).
//
//   4. MAPPING  — The index fingertip's y-position (0 = top, 1 = bottom) is
//                 mapped to a hue in HSL colour space. A gradient strip on
//                 the left edge shows the full range; a dot marks the current
//                 position. This illustrates how any continuous landmark
//                 value can drive an arbitrary output range.
//
// WHICH LANDMARKS ARE USED?
// -------------------------
//   Hand landmark  8         — Index fingertip (zone, distance, mapping)
//   Hand landmarks 8/6, 12/10, 16/14, 20/18 — Tip/PIP pairs for finger count
//   Face landmark  4         — Nose tip (distance)
//
// DEBUGGING
// ---------
// Set debugMode = true to log the interaction state on every frame.

// =============================================================================
// CONSTANTS
// =============================================================================

const debugMode = false;

// Dot sizes (px).
const HAND_DOT_RADIUS  = 5;
const INDEX_TIP_RADIUS = 10;
const NOSE_TIP_RADIUS  = 8;

// Colour palette.
const COLOR_GREEN  = "#4ade80";
const COLOR_BLUE   = "#60a5fa";
const COLOR_PURPLE = "#c084fc";
const COLOR_RED    = "#f87171";
const COLOR_AMBER  = "#fbbf24";

// Maximum distance (px) that fills the distance bar to 100 %.
const MAX_DIST_PX = 400;

// Tip and PIP joint index pairs for each non-thumb finger [tip, pip].
const FINGER_PAIRS = [
  [8,  6],  // index
  [12, 10], // middle
  [16, 14], // ring
  [20, 18], // pinky
];

// Hand skeleton connections — same as Demos 1 and 3.
const HAND_CONNECTIONS = [
  [0, 1],   [1, 2],   [2, 3],   [3, 4],
  [0, 5],   [5, 6],   [6, 7],   [7, 8],
  [0, 9],   [9, 10],  [10, 11], [11, 12],
  [0, 13],  [13, 14], [14, 15], [15, 16],
  [0, 17],  [17, 18], [18, 19], [19, 20],
  [5, 9],   [9, 13],  [13, 17]
];

// =============================================================================
// PURE HELPERS (no canvas state)
// =============================================================================

/**
 * dist2D — Euclidean distance in pixels between two normalised landmarks.
 *
 * @param {{x:number, y:number}} a - First landmark (0–1 normalised).
 * @param {{x:number, y:number}} b - Second landmark (0–1 normalised).
 * @param {number} w - Canvas width in pixels.
 * @param {number} h - Canvas height in pixels.
 * @returns {number} Distance in pixels.
 */
function dist2D(a, b, w, h) {
  const dx = (a.x - b.x) * w;
  const dy = (a.y - b.y) * h;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * lerp — linear interpolation, clamped to [0, 1].
 *
 * @param {number} a - Start value.
 * @param {number} b - End value.
 * @param {number} t - Blend factor (0–1).
 * @returns {number}
 */
function lerp(a, b, t) {
  return a + (b - a) * Math.max(0, Math.min(1, t));
}

/**
 * countFingersUp — counts how many of the four non-thumb fingers are raised.
 * A finger is "up" when its tip y-coordinate is numerically less than its
 * PIP joint y-coordinate (y increases downward in image space).
 *
 * @param {Array} lm - Array of 21 hand landmark objects.
 * @returns {number} Number of raised fingers (0–4).
 */
function countFingersUp(lm) {
  return FINGER_PAIRS.reduce(
    (n, [tip, pip]) => n + (lm[tip].y < lm[pip].y ? 1 : 0),
    0
  );
}

/**
 * getZone — returns which horizontal third of the frame a normalised x
 * coordinate falls in.
 *
 * @param {number} normX - Normalised x coordinate (0–1).
 * @returns {"LEFT"|"CENTRE"|"RIGHT"}
 */
function getZone(normX) {
  if (normX < 1 / 3) return "LEFT";
  if (normX < 2 / 3) return "CENTRE";
  return "RIGHT";
}

/**
 * distanceColor — interpolates an rgb colour between red (close) and green
 * (far) based on a pixel distance value.
 *
 * @param {number} px  - Current distance in pixels.
 * @param {number} max - Distance that maps to pure green.
 * @returns {string} CSS rgb() colour string.
 */
function distanceColor(px, max) {
  const t = Math.max(0, Math.min(1, px / max));  // 0 = close, 1 = far
  const r = Math.round(lerp(248, 74,  t));
  const g = Math.round(lerp(113, 222, t));
  const b = Math.round(lerp(113, 128, t));
  return `rgb(${r},${g},${b})`;
}

/**
 * hexToRgba — converts a #rrggbb hex colour and an alpha value to an rgba()
 * CSS string. Accepts only the six-digit hex format.
 *
 * @param {string} hex   - CSS hex colour, e.g. "#4ade80".
 * @param {number} alpha - Opacity, 0–1.
 * @returns {string}
 */
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// =============================================================================
// ENTRY POINT
// =============================================================================

window.onload = async function () {

  if (debugMode) {
    console.log("Demo 4 — Interaction Basics loading...");
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

  // ── MediaPipe HandLandmarker setup ───────────────────────────────────────

  let handLandmarker;
  try {
    showStatus("Loading hand model (GPU)...");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker" +
          "/hand_landmarker/float16/latest/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 1
    });
  } catch (gpuErr) {
    console.warn("GPU delegate unavailable, retrying with CPU:", gpuErr);
    showStatus("Loading hand model (CPU fallback)...");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker" +
          "/hand_landmarker/float16/latest/hand_landmarker.task",
        delegate: "CPU"
      },
      runningMode: "VIDEO",
      numHands: 1
    });
  }

  if (debugMode) {
    console.log("MediaPipe HandLandmarker initialised.");
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
    console.log("MediaPipe FaceLandmarker initialised.");
  }

  let handResults = [];
  let faceResults = [];

  // Each model keeps its own monotonic timestamp for detectForVideo.
  let lastTimestampHand = -1;
  let lastTimestampFace = -1;

  // ── Camera management ────────────────────────────────────────────────────

  let currentStream   = null;
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
   * @param {string} [deviceId] — exact device ID, or omit for the default.
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

    video.onloadedmetadata = () => {
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
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

    showStatus("Camera running. Starting tracking...");
    frameLoopActive = true;
    requestAnimationFrame(frameLoop);
    requestAnimationFrame(renderLoop);
  }

  /**
   * renderLoop — redraws the canvas at ~60 fps, independent of inference rate.
   */
  function renderLoop() {
    if (!frameLoopActive) return;
    drawFrame();
    requestAnimationFrame(renderLoop);
  }

  /**
   * frameLoop — runs both landmarkers on the current video frame on every
   * animation tick. detectForVideo is synchronous. The timestamp guard
   * ensures the value always increases, which the Tasks API requires.
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

      if (debugMode) console.log("Running HandLandmarker and FaceLandmarker...");
    }
    requestAnimationFrame(frameLoop);
  }

  /**
   * populateCameraSelect — enumerates video-input devices and fills the
   * on-page <select>. The wrapper is shown only when more than one camera is
   * available.
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

  // ── Main draw ────────────────────────────────────────────────────────────

  /**
   * drawFrame — clears the canvas, draws the webcam feed, computes the four
   * interaction values, and renders all visual layers in order.
   */
  function drawFrame() {
    const w = video.videoWidth;
    const h = video.videoHeight;

    if (canvas.width  !== w) canvas.width  = w;
    if (canvas.height !== h) canvas.height = h;

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(video, 0, 0, w, h);

    // ── Extract first detected hand and face ─────────────────────────────

    const hand = handResults.length > 0 ? handResults[0] : null;
    const face = faceResults.length > 0 ? faceResults[0] : null;

    // ── Compute interaction values ───────────────────────────────────────

    // 1. Zone — which horizontal third is the index fingertip in?
    const zone = hand ? getZone(hand[8].x) : null;

    // 2. Distance — pixels between index tip and nose tip
    const distPx = (hand && face) ? dist2D(hand[8], face[4], w, h) : null;

    // 3. Count — how many non-thumb fingers are raised?
    const fingerCount = hand ? countFingersUp(hand) : 0;

    // 4. Mapping — index fingertip y-position (0 = top) → hue (200 = blue, 0 = red)
    const mappedHue = hand ? Math.round((1 - hand[8].y) * 200) : null;

    if (debugMode) {
      console.log({ zone, distPx, fingerCount, mappedHue });
    }

    // ── Layer 1: Zone overlay ────────────────────────────────────────────
    drawZoneOverlay(w, h, zone);

    // ── Layer 2: Hand skeleton and dots ──────────────────────────────────
    if (hand) {
      drawHandSkeleton(hand, w, h);
      drawHandDots(hand, fingerCount, w, h);
    }

    // ── Layer 3: Nose tip marker ─────────────────────────────────────────
    if (face) {
      drawNoseTip(face[4], w, h);
    }

    // ── Layer 4: Proximity line between index tip and nose ───────────────
    if (hand && face && distPx !== null) {
      drawProximityLine(hand[8], face[4], distPx, w, h);
    }

    // ── Layer 5: Distance bar (right edge) ───────────────────────────────
    drawDistanceBar(w, h, distPx);

    // ── Layer 6: Height-to-hue strip (left edge) ─────────────────────────
    drawHeightStrip(w, h, mappedHue, hand ? hand[8] : null);

    // ── Layer 7: HUD info panel ──────────────────────────────────────────
    drawHUD(w, h, fingerCount, zone, distPx, mappedHue, hand !== null, face !== null);
  }

  // ── Drawing sub-functions ────────────────────────────────────────────────

  /**
   * drawZoneOverlay — fills and outlines three vertical column zones. The
   * zone containing the index fingertip is highlighted.
   *
   * @param {number}                   w          - Canvas width.
   * @param {number}                   h          - Canvas height.
   * @param {"LEFT"|"CENTRE"|"RIGHT"|null} activeZone - Currently active zone.
   */
  function drawZoneOverlay(w, h, activeZone) {
    const zones = [
      { name: "LEFT",   x: 0,           color: COLOR_GREEN  },
      { name: "CENTRE", x: w / 3,       color: COLOR_BLUE   },
      { name: "RIGHT",  x: (2 * w) / 3, color: COLOR_PURPLE },
    ];
    const zw = w / 3;

    ctx.textAlign = "center";

    zones.forEach(({ name, x, color }) => {
      const active = name === activeZone;

      // Column tint
      ctx.fillStyle = hexToRgba(color, active ? 0.18 : 0.04);
      ctx.fillRect(x, 0, zw, h);

      // Column border
      ctx.strokeStyle = hexToRgba(color, active ? 0.5 : 0.1);
      ctx.lineWidth   = active ? 2 : 1;
      ctx.setLineDash([]);
      ctx.strokeRect(x + 0.5, 0.5, zw - 1, h - 1);

      // Zone label
      ctx.font      = `bold ${active ? 13 : 10}px monospace`;
      ctx.fillStyle = active ? color : "rgba(255,255,255,0.2)";
      ctx.fillText(name, x + zw / 2, 20);
    });

    ctx.textAlign = "left";
  }

  /**
   * drawHandSkeleton — draws the hand skeleton into the full (un-split) canvas.
   *
   * @param {Array}  lm - 21 hand landmark objects.
   * @param {number} w  - Canvas width.
   * @param {number} h  - Canvas height.
   */
  function drawHandSkeleton(lm, w, h) {
    ctx.strokeStyle = "rgba(74,222,128,0.4)";
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([]);

    ctx.beginPath();
    HAND_CONNECTIONS.forEach(([a, b]) => {
      ctx.moveTo(lm[a].x * w, lm[a].y * h);
      ctx.lineTo(lm[b].x * w, lm[b].y * h);
    });
    ctx.stroke();
  }

  /**
   * drawHandDots — draws all 21 hand landmark dots. Dot colour shifts from
   * red (0 fingers raised) to green (4 fingers raised) so the count is
   * immediately visible. The index fingertip is drawn larger.
   *
   * @param {Array}  lm          - 21 hand landmark objects.
   * @param {number} fingerCount - Number of raised fingers (0–4).
   * @param {number} w           - Canvas width.
   * @param {number} h           - Canvas height.
   */
  function drawHandDots(lm, fingerCount, w, h) {
    // Hue: 0 (red) at 0 fingers, 120 (green) at 4 fingers.
    const hue   = Math.round((fingerCount / 4) * 120);
    const color = `hsl(${hue},80%,65%)`;

    // Draw all dots except index tip (drawn separately below).
    ctx.fillStyle = color;
    ctx.beginPath();
    lm.forEach((pt, i) => {
      if (i === 8) return;
      const cx = pt.x * w;
      const cy = pt.y * h;
      ctx.moveTo(cx + HAND_DOT_RADIUS, cy);
      ctx.arc(cx, cy, HAND_DOT_RADIUS, 0, Math.PI * 2);
    });
    ctx.fill();

    // Index fingertip — larger white circle with a coloured ring.
    const ix = lm[8].x * w;
    const iy = lm[8].y * h;

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(ix, iy, INDEX_TIP_RADIUS, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.stroke();
  }

  /**
   * drawNoseTip — draws an amber ring at the nose tip landmark.
   *
   * @param {{x:number, y:number}} pt - Nose tip landmark (normalised).
   * @param {number} w                - Canvas width.
   * @param {number} h                - Canvas height.
   */
  function drawNoseTip(pt, w, h) {
    const x = pt.x * w;
    const y = pt.y * h;

    ctx.strokeStyle = COLOR_AMBER;
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(x, y, NOSE_TIP_RADIUS, 0, Math.PI * 2);
    ctx.stroke();
  }

  /**
   * drawProximityLine — draws a dashed line from the index tip to the nose
   * tip. The colour interpolates from green (far) to red (close). The
   * pixel distance is labelled at the midpoint.
   *
   * @param {{x:number, y:number}} indexTip - Index fingertip (normalised).
   * @param {{x:number, y:number}} noseTip  - Nose tip (normalised).
   * @param {number} px                     - Distance in pixels.
   * @param {number} w                      - Canvas width.
   * @param {number} h                      - Canvas height.
   */
  function drawProximityLine(indexTip, noseTip, px, w, h) {
    const color = distanceColor(px, MAX_DIST_PX);
    const ix    = indexTip.x * w;
    const iy    = indexTip.y * h;
    const nx    = noseTip.x  * w;
    const ny    = noseTip.y  * h;

    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(ix, iy);
    ctx.lineTo(nx, ny);
    ctx.stroke();
    ctx.setLineDash([]);

    // Distance label at the midpoint.
    const mx = (ix + nx) / 2;
    const my = (iy + ny) / 2;
    ctx.font      = "bold 13px monospace";
    ctx.fillStyle = color;
    ctx.textAlign = "center";
    ctx.fillText(`${Math.round(px)} px`, mx, my - 8);
    ctx.textAlign = "left";
  }

  /**
   * drawDistanceBar — vertical bar on the right edge. The fill level rises
   * with distance (empty = touching, full = 400 px or more away).
   *
   * @param {number}      w    - Canvas width.
   * @param {number}      h    - Canvas height.
   * @param {number|null} px   - Current distance in pixels, or null if unknown.
   */
  function drawDistanceBar(w, h, px) {
    const bw = 18;
    const bh = Math.round(h * 0.5);
    const bx = w - bw - 8;
    const by = Math.round((h - bh) / 2);

    // Track background.
    ctx.fillStyle = "rgba(255,255,255,0.06)";
    ctx.fillRect(bx, by, bw, bh);

    // Fill proportional to distance.
    if (px !== null) {
      const frac  = Math.min(1, px / MAX_DIST_PX);
      const fh    = Math.round(bh * frac);
      const color = distanceColor(px, MAX_DIST_PX);
      ctx.fillStyle = color;
      ctx.fillRect(bx, by + bh - fh, bw, fh);
    }

    // Border.
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth   = 1;
    ctx.setLineDash([]);
    ctx.strokeRect(bx, by, bw, bh);

    // Label.
    ctx.font      = "10px monospace";
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.textAlign = "center";
    ctx.fillText("DIST", bx + bw / 2, by - 5);
    ctx.textAlign = "left";
  }

  /**
   * drawHeightStrip — narrow gradient strip on the left edge showing the full
   * hue-mapping range. A dot marks the index fingertip's current position.
   *
   * @param {number}                        w        - Canvas width.
   * @param {number}                        h        - Canvas height.
   * @param {number|null}                   hue      - Current mapped hue (0–200).
   * @param {{x:number,y:number}|null}      indexTip - Index tip landmark, or null.
   */
  function drawHeightStrip(w, h, hue, indexTip) {
    const sw = 10;
    const sh = Math.round(h * 0.5);
    const sx = 8;
    const sy = Math.round((h - sh) / 2);

    // Gradient strip — always visible as a reference.
    for (let i = 0; i < sh; i++) {
      const refHue  = Math.round((1 - i / sh) * 200);
      ctx.fillStyle = `hsl(${refHue},70%,55%)`;
      ctx.fillRect(sx, sy + i, sw, 1);
    }

    // Dot showing the current position.
    if (indexTip !== null && hue !== null) {
      const dotY  = sy + sh * indexTip.y;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth   = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(sx + sw / 2, dotY, 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = `hsl(${hue},80%,65%)`;
      ctx.fill();
    }

    // Label.
    ctx.font      = "10px monospace";
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.textAlign = "center";
    ctx.fillText("HUE", sx + sw / 2, sy - 5);
    ctx.textAlign = "left";
  }

  /**
   * drawHUD — translucent info panel in the bottom-left corner listing the
   * four current interaction values.
   *
   * @param {number}                   w           - Canvas width.
   * @param {number}                   h           - Canvas height.
   * @param {number}                   fingerCount - Raised finger count (0–4).
   * @param {"LEFT"|"CENTRE"|"RIGHT"|null} zone    - Active zone, or null.
   * @param {number|null}              distPx      - Distance in px, or null.
   * @param {number|null}              mappedHue   - Mapped hue (0–200), or null.
   * @param {boolean}                  hasHand     - Whether a hand is detected.
   * @param {boolean}                  hasFace     - Whether a face is detected.
   */
  function drawHUD(w, h, fingerCount, zone, distPx, mappedHue, hasHand, hasFace) {
    const panelX = 8;
    const panelY = h - 108;
    const panelW = 220;
    const panelH = 100;

    ctx.fillStyle = "rgba(0,0,0,0.55)";
    roundRect(ctx, panelX, panelY, panelW, panelH, 6);
    ctx.fill();

    ctx.font = "12px monospace";

    const zoneColor = zone === "LEFT"
      ? COLOR_GREEN
      : zone === "CENTRE"
        ? COLOR_BLUE
        : zone === "RIGHT"
          ? COLOR_PURPLE
          : "#888";

    const lines = [
      {
        label: "1. Fingers up",
        value: hasHand ? `${fingerCount}` : "—",
        color: hasHand ? `hsl(${Math.round((fingerCount / 4) * 120)},80%,65%)` : "#888"
      },
      {
        label: "2. Zone",
        value: zone || "—",
        color: zoneColor
      },
      {
        label: "3. Distance",
        value: distPx !== null ? `${Math.round(distPx)} px` : "— (need face)",
        color: distPx !== null ? distanceColor(distPx, MAX_DIST_PX) : "#888"
      },
      {
        label: "4. Height hue",
        value: mappedHue !== null ? `hsl(${mappedHue})` : "—",
        color: mappedHue !== null ? `hsl(${mappedHue},80%,65%)` : "#888"
      },
    ];

    lines.forEach(({ label, value, color }, i) => {
      const ly = panelY + 18 + i * 21;
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.fillText(label, panelX + 10, ly);
      ctx.fillStyle = color;
      ctx.fillText(value, panelX + 130, ly);
    });
  }

  // ── Canvas utility ───────────────────────────────────────────────────────

  /**
   * roundRect — traces a rounded-rectangle path on the given context.
   *
   * @param {CanvasRenderingContext2D} context
   * @param {number} x
   * @param {number} y
   * @param {number} width
   * @param {number} height
   * @param {number} radius
   */
  function roundRect(context, x, y, width, height, radius) {
    context.beginPath();
    context.moveTo(x + radius, y);
    context.arcTo(x + width, y,          x + width, y + height, radius);
    context.arcTo(x + width, y + height, x,         y + height, radius);
    context.arcTo(x,         y + height, x,         y,          radius);
    context.arcTo(x,         y,          x + width, y,          radius);
    context.closePath();
  }

};
