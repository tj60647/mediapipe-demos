// =============================================================================
// File:    sketch.js
// Project: MediaPipe Demos — MDes Prototyping, CCA
// Demo:    5 — Face Instrument
//
// Authors: Copilot
//          Thomas J McLeish
// License: MIT — see LICENSE in the root of this repository
// =============================================================================
//
// PURPOSE
// -------
// This sketch turns the tracked face into a proximity-sensitive control
// surface. Nine named regions are defined on the face — each mapped to a
// single representative landmark. When the index fingertip (hand landmark 8)
// comes within PROXIMITY_THRESHOLD pixels of a region, that region is
// "activated."
//
// WHAT YOU SEE
// ------------
// - The webcam feed with subtle face landmark dots drawn underneath
// - Inactive regions: dim rings at each region's position
// - Active regions:   bright glowing rings + an expanding ripple effect
// - A ripple animation spawns each time a region first becomes active
// - The index fingertip is highlighted as a larger ring
// - A legend panel on the right lists all regions and their activation state
//
// FACE REGIONS
// ------------
// Each region is defined by a single face landmark index:
//
//   Name              Landmark  Colour
//   ──────────────    ────────  ──────
//   Forehead          10        amber  (#fbbf24)
//   Left Eyebrow      70        pink   (#f472b6)
//   Right Eyebrow     300       pink   (#f472b6)
//   Left Eye          33        blue   (#60a5fa)
//   Right Eye         263       blue   (#60a5fa)
//   Nose Tip          4         purple (#c084fc)
//   Lips              13        red    (#f87171)
//   Left Cheek        234       green  (#4ade80)
//   Right Cheek       454       green  (#4ade80)
//
// (Left/right are from the camera's perspective, mirrored from the user's
// perspective.)
//
// All nine indices (10, 70, 300, 33, 263, 4, 13, 234, 454) are within the
// 478-point Face Landmarker model (468 face points + 10 iris points).
//
// INTERACTION PATTERN: PROXIMITY TRIGGER
// ---------------------------------------
// At each frame:
//   1. Convert both the fingertip and each region landmark to pixel coords.
//   2. Compute Euclidean distance.
//   3. If distance < PROXIMITY_THRESHOLD, the region is "active" this frame.
//   4. If a region just became active (was inactive last frame), spawn a ripple.
//
// This is a binary proximity trigger — the simplest possible conversion of
// a continuous distance value into an on/off state.
//
// WHAT TO CHANGE
// --------------
// - PROXIMITY_THRESHOLD — increase to make regions easier to activate,
//   decrease to require the finger to be very close.
// - REGION_RING_RADIUS  — size of the ring drawn at each region.
// - Add or remove entries in FACE_REGIONS to change which areas are tracked.
// - Replace the visual effect with any output: sound, colour fills, text.
//
// DEBUGGING
// ---------
// Set debugMode = true to log the active regions to the console each frame.

// =============================================================================
// CONSTANTS
// =============================================================================

const debugMode = false;

// Distance in pixels below which a region is considered "active."
// Tune this to the desired feel: 60 px works well at 640×480.
const PROXIMITY_THRESHOLD = 65;

// Radius of the ring drawn at each face region (px).
const REGION_RING_RADIUS = 20;

// Radius of the index fingertip indicator (px).
const TIP_RING_RADIUS = 12;

// Maximum radius to which a ripple expands (px).
const RIPPLE_MAX_RADIUS = 80;

// Speed at which a ripple expands (px per frame at ~60 fps).
const RIPPLE_SPEED = 2.5;

// Hand skeleton connections — same as Demos 1–4.
const HAND_CONNECTIONS = [
  [0, 1],   [1, 2],   [2, 3],   [3, 4],
  [0, 5],   [5, 6],   [6, 7],   [7, 8],
  [0, 9],   [9, 10],  [10, 11], [11, 12],
  [0, 13],  [13, 14], [14, 15], [15, 16],
  [0, 17],  [17, 18], [18, 19], [19, 20],
  [5, 9],   [9, 13],  [13, 17]
];

// =============================================================================
// FACE REGION DEFINITIONS
// =============================================================================
// Each object defines one trackable region on the face:
//   name     — label shown in the legend
//   index    — MediaPipe Face Landmarker landmark index (0–467)
//   color    — CSS hex colour for this region

const FACE_REGIONS = [
  { name: "Forehead",       index: 10,  color: "#fbbf24" },  // amber
  { name: "Left Eyebrow",   index: 70,  color: "#f472b6" },  // pink
  { name: "Right Eyebrow",  index: 300, color: "#f472b6" },  // pink
  { name: "Left Eye",       index: 33,  color: "#60a5fa" },  // blue
  { name: "Right Eye",      index: 263, color: "#60a5fa" },  // blue
  { name: "Nose Tip",       index: 4,   color: "#c084fc" },  // purple
  { name: "Lips",           index: 13,  color: "#f87171" },  // red/coral
  { name: "Left Cheek",     index: 234, color: "#4ade80" },  // green
  { name: "Right Cheek",    index: 454, color: "#4ade80" },  // green
];

// =============================================================================
// ENTRY POINT
// =============================================================================

window.onload = async function () {

  if (debugMode) {
    console.log("Demo 5 — Face Instrument loading...");
  }

  // ── DOM elements ────────────────────────────────────────────────────────

  const video  = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx    = canvas.getContext("2d", { alpha: false });

  if (!video || !canvas) {
    console.error("Error: Could not find #video or #canvas in the DOM.");
    return;
  }

  function setStatus(msg, isError = false) {
    const el = document.getElementById("status");
    if (!el) return;
    el.textContent = msg;
    el.className = isError ? "error" : "";
  }

  // ── Ripple state ─────────────────────────────────────────────────────────
  // Each ripple: { x, y, color, radius, alpha }
  // Ripples expand and fade over time.
  let ripples = [];

  // Track which regions were active last frame to detect rising edges.
  let prevActive = new Set();

  // ── MediaPipe Tasks Vision ───────────────────────────────────────────────

  setStatus("Loading hand model…");

  const { HandLandmarker, FaceLandmarker, FilesetResolver } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs"
  );

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm"
  );

  // ── MediaPipe HandLandmarker setup ───────────────────────────────────────

  let handLandmarker;
  try {
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

  setStatus("Loading face model…");

  // ── MediaPipe FaceLandmarker setup ───────────────────────────────────────

  let faceLandmarker;
  try {
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

  setStatus("Camera starting…");

  let handResults = [];
  let faceResults = [];

  // Monotonically increasing timestamp required by detectForVideo.
  let lastTimestamp = -1;

  // ── Camera management ────────────────────────────────────────────────────

  let currentStream   = null;
  let frameLoopActive = false;

  /**
   * startCamera — opens the webcam with an optional specific device and
   * starts the per-frame inference and render loops.
   *
   * @param {string} [deviceId] — exact device ID, or omit for the default.
   */
  async function startCamera(deviceId) {
    frameLoopActive = false;

    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
      currentStream = null;
    }

    setStatus("Requesting camera…");

    const videoConstraints = { width: { ideal: 640 }, height: { ideal: 480 } };
    if (deviceId) {
      videoConstraints.deviceId = { exact: deviceId };
    } else {
      videoConstraints.facingMode = { ideal: "user" };
    }

    try {
      currentStream = await navigator.mediaDevices.getUserMedia(
        { video: videoConstraints }
      );
    } catch (err) {
      console.error("Could not open camera:", err);
      setStatus("Camera error: " + err.message, true);
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
    } catch (err) {
      console.error("video.play() failed:", err);
      setStatus("Video error: " + err.message, true);
      return;
    }

    setStatus("");

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
      const ts = performance.now();
      if (ts > lastTimestamp) {
        lastTimestamp = ts;
        const handResult = handLandmarker.detectForVideo(video, ts);
        handResults = handResult.landmarks ?? [];
        const faceResult = faceLandmarker.detectForVideo(video, ts);
        faceResults = faceResult.faceLandmarks ?? [];
      }
    }
    requestAnimationFrame(frameLoop);
  }

  /**
   * populateCameraSelect — enumerates video-input devices and fills the
   * on-page <select>. Shown only when more than one camera is available.
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
   * drawFrame — clears the canvas, draws the webcam feed, updates ripples,
   * computes which face regions are active, and renders all visual layers.
   */
  function drawFrame() {
    const w = video.videoWidth;
    const h = video.videoHeight;

    if (canvas.width  !== w) canvas.width  = w;
    if (canvas.height !== h) canvas.height = h;

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(video, 0, 0, w, h);

    const hand = handResults.length > 0 ? handResults[0] : null;
    const face = faceResults.length > 0 ? faceResults[0] : null;

    // ── Compute active regions ───────────────────────────────────────────

    // activeThisFrame maps region name → pixel position { x, y, color }
    const activeThisFrame = new Map();

    if (hand && face) {
      const tipX = hand[8].x * w;
      const tipY = hand[8].y * h;

      FACE_REGIONS.forEach(region => {
        const lm = face[region.index];
        const rx = lm.x * w;
        const ry = lm.y * h;
        const dx = tipX - rx;
        const dy = tipY - ry;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < PROXIMITY_THRESHOLD) {
          activeThisFrame.set(region.name, { x: rx, y: ry, color: region.color });

          // Spawn a ripple when this region first becomes active.
          if (!prevActive.has(region.name)) {
            ripples.push({
              x: rx, y: ry,
              color: region.color,
              radius: REGION_RING_RADIUS,
              alpha: 0.9
            });
          }
        }
      });

      if (debugMode && activeThisFrame.size > 0) {
        console.log("Active regions:", [...activeThisFrame.keys()].join(", "));
      }
    }

    prevActive = new Set(activeThisFrame.keys());

    // ── Layer 1: Face landmark dots (very subtle) ────────────────────────
    if (face) {
      drawFaceDots(face, w, h);
    }

    // ── Layer 2: Inactive region rings ───────────────────────────────────
    if (face) {
      drawInactiveRegions(face, activeThisFrame, w, h);
    }

    // ── Layer 3: Active region glows ─────────────────────────────────────
    activeThisFrame.forEach(({ x, y, color }) => {
      drawActiveGlow(x, y, color);
    });

    // ── Layer 4: Ripple animations ───────────────────────────────────────
    updateAndDrawRipples();

    // ── Layer 5: Hand skeleton and index tip ─────────────────────────────
    if (hand) {
      drawHandSkeleton(hand, w, h);
      drawIndexTip(hand[8], w, h);
    }

    // ── Layer 6: Legend panel ────────────────────────────────────────────
    drawLegend(w, h, activeThisFrame);
  }

  // ── Drawing sub-functions ────────────────────────────────────────────────

  /**
   * drawFaceDots — draws all face landmarks as very small, dim dots so the
   * face mesh is visible but doesn't compete with the interaction overlays.
   *
   * @param {Array}  lm - 468 face landmark objects.
   * @param {number} w  - Canvas width.
   * @param {number} h  - Canvas height.
   */
  function drawFaceDots(lm, w, h) {
    ctx.fillStyle = "rgba(200,200,200,0.25)";
    ctx.beginPath();
    lm.forEach(pt => {
      const cx = pt.x * w;
      const cy = pt.y * h;
      ctx.moveTo(cx + 1.5, cy);
      ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
    });
    ctx.fill();
  }

  /**
   * drawInactiveRegions — draws a dim ring at each face region that is not
   * currently active.
   *
   * @param {Array}  faceLm       - 468 face landmark objects.
   * @param {Map}    activeRegions - Map of currently active region names.
   * @param {number} w            - Canvas width.
   * @param {number} h            - Canvas height.
   */
  function drawInactiveRegions(faceLm, activeRegions, w, h) {
    FACE_REGIONS.forEach(region => {
      if (activeRegions.has(region.name)) return;  // skip active ones

      const lm = faceLm[region.index];
      const x  = lm.x * w;
      const y  = lm.y * h;

      ctx.strokeStyle = hexToRgba(region.color, 0.3);
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(x, y, REGION_RING_RADIUS, 0, Math.PI * 2);
      ctx.stroke();
    });
  }

  /**
   * drawActiveGlow — draws a bright glowing ring at the position of an
   * active region. Uses canvas shadow blur for the glow effect.
   *
   * @param {number} x     - Region centre x in pixels.
   * @param {number} y     - Region centre y in pixels.
   * @param {string} color - CSS hex colour for this region.
   */
  function drawActiveGlow(x, y, color) {
    // Outer glow via shadow.
    ctx.shadowColor = color;
    ctx.shadowBlur  = 24;

    ctx.strokeStyle = color;
    ctx.lineWidth   = 3;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(x, y, REGION_RING_RADIUS, 0, Math.PI * 2);
    ctx.stroke();

    // Inner filled dot.
    ctx.fillStyle = hexToRgba(color, 0.4);
    ctx.beginPath();
    ctx.arc(x, y, REGION_RING_RADIUS * 0.5, 0, Math.PI * 2);
    ctx.fill();

    // Reset shadow so it doesn't affect other drawings.
    ctx.shadowColor = "transparent";
    ctx.shadowBlur  = 0;
  }

  /**
   * updateAndDrawRipples — advances each ripple by one step and draws it.
   * Removes ripples that have fully faded or reached their maximum radius.
   */
  function updateAndDrawRipples() {
    ripples = ripples.filter(rip => rip.alpha > 0.01 && rip.radius < RIPPLE_MAX_RADIUS);

    ripples.forEach(rip => {
      rip.radius += RIPPLE_SPEED;
      rip.alpha  *= 0.94;  // fade out gradually

      ctx.strokeStyle = hexToRgba(rip.color, rip.alpha);
      ctx.lineWidth   = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(rip.x, rip.y, rip.radius, 0, Math.PI * 2);
      ctx.stroke();
    });
  }

  /**
   * drawHandSkeleton — draws the hand skeleton into the canvas.
   *
   * @param {Array}  lm - 21 hand landmark objects.
   * @param {number} w  - Canvas width.
   * @param {number} h  - Canvas height.
   */
  function drawHandSkeleton(lm, w, h) {
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
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
   * drawIndexTip — draws a highlighted ring at the index fingertip position
   * to show which point is used for proximity detection.
   *
   * @param {{x:number, y:number}} pt - Index tip landmark (normalised).
   * @param {number} w                - Canvas width.
   * @param {number} h                - Canvas height.
   */
  function drawIndexTip(pt, w, h) {
    const x = pt.x * w;
    const y = pt.y * h;

    // White filled dot.
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x, y, TIP_RING_RADIUS * 0.4, 0, Math.PI * 2);
    ctx.fill();

    // Outer ring.
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(x, y, TIP_RING_RADIUS, 0, Math.PI * 2);
    ctx.stroke();
  }

  /**
   * drawLegend — draws a panel on the right side listing all face regions
   * with their colours and activation state.
   *
   * @param {number} w             - Canvas width.
   * @param {number} h             - Canvas height.
   * @param {Map}    activeRegions - Map of active region names.
   */
  function drawLegend(w, h, activeRegions) {
    const lineH  = 20;
    const panelH = FACE_REGIONS.length * lineH + 20;
    const panelW = 160;
    const panelX = w - panelW - 8;
    const panelY = Math.round((h - panelH) / 2);

    // Panel background.
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    roundRect(ctx, panelX, panelY, panelW, panelH, 6);
    ctx.fill();

    ctx.font = "11px monospace";

    FACE_REGIONS.forEach((region, i) => {
      const active = activeRegions.has(region.name);
      const ly     = panelY + 14 + i * lineH;

      // Colour dot.
      ctx.fillStyle = active ? region.color : hexToRgba(region.color, 0.3);
      ctx.beginPath();
      ctx.arc(panelX + 12, ly, 4, 0, Math.PI * 2);
      ctx.fill();

      // Region name.
      ctx.fillStyle = active ? "#ffffff" : "rgba(255,255,255,0.3)";
      ctx.fillText(region.name, panelX + 22, ly + 4);

      // Active indicator.
      if (active) {
        ctx.fillStyle = region.color;
        ctx.fillText("●", panelX + panelW - 16, ly + 4);
      }
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

};
