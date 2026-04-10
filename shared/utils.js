// =============================================================================
// File:    shared/utils.js
// Project: MediaPipe Demos — MDes Prototyping, CCA
//
// Authors: Copilot
//          Thomas J McLeish
// License: MIT — see LICENSE in the root of this repository
// =============================================================================
//
// PURPOSE
// -------
// Shared utilities used by every demo sketch. Loaded before sketch.js in each
// demo's index.html via:
//
//   <script defer src="../shared/utils.js"></script>
//
// All symbols are declared at the top level and are therefore available as
// browser globals to the sketch.js that follows.
//
// EXPORTS
// -------
//   isMobile                              — true on touch-capable devices
//   HAND_CONNECTIONS                      — MediaPipe hand skeleton pairs
//   showBrowserWarning()                  — shows banner for limited browsers
//   showError(err)                        — shows on-page error message
//   populateCameraSelect(id, onChange)    — fills the camera <select>
//   hexToRgba(hex, alpha)                 — colour utility

// =============================================================================
// DEVICE DETECTION
// =============================================================================

/** True when the page is loaded on a touch-capable mobile device. */
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
// BROWSER COMPATIBILITY WARNING
// =============================================================================

/**
 * showBrowserWarning — reveals the #browserWarning banner when the page is
 * opened in a browser likely to have limited WebGL/WASM support.
 *
 * - Android + non-Chrome (incl. Samsung Internet): recommend Chrome.
 * - iOS + in-app browser (not Safari, not Chrome for iOS): recommend Safari.
 *
 * Chrome on iOS uses the same WebKit engine as Safari and performs equally
 * well, so no warning is shown for Chrome-for-iOS users. The original
 * catch-all "use Chrome" message was incorrect for iOS — Chrome for iOS
 * offers no WebGL/WASM advantage over Safari.
 */
function showBrowserWarning() {
  const ua        = navigator.userAgent;
  const isAndroid = /Android/i.test(ua);
  const isIOS     = /iPhone|iPad/i.test(ua);
  const isSamsung = /SamsungBrowser/i.test(ua);
  const isChrome  = /Chrome\/[0-9]/i.test(ua);
  // Safari's UA includes "Safari"; Chrome for iOS uses "CriOS" but also
  // includes "Safari", so hasSafariUA covers both real Safari and CriOS.
  const hasSafariUA = /Safari\/[0-9]/i.test(ua);

  let message = null;
  if (isAndroid && (isSamsung || !isChrome)) {
    message =
      "For best results on Android, open this page in Chrome. " +
      "Samsung Internet and other non-Chromium browsers may not support " +
      "the AI models used in these demos.";
  } else if (isIOS && !hasSafariUA) {
    // In-app browsers (e.g. Instagram, Facebook) have neither "Safari" nor
    // "CriOS" in their user-agent and often block WebAssembly.
    message =
      "For best results on iOS, open this page in Safari. " +
      "In-app browsers may not support the AI models used in these demos.";
  }

  if (!message) return;
  const el = document.getElementById("browserWarning");
  if (!el) return;
  el.textContent   = message;
  el.style.display = "block";
}

// =============================================================================
// ERROR DISPLAY
// =============================================================================

/**
 * showError — displays a human-readable error in the #errorMessage element so
 * that mobile users who cannot open DevTools can still see what went wrong.
 *
 * Also hides the loading indicator (#loadingMessage) since the error message
 * takes its place.
 *
 * @param {Error} err - The error to display.
 */
function showError(err) {
  const loading = document.getElementById("loadingMessage");
  if (loading) loading.style.display = "none";

  const el = document.getElementById("errorMessage");
  if (!el) return;
  el.textContent = err.name === "NotAllowedError"
    ? "Camera access was denied. Please allow camera permission and reload."
    : `Error: ${err.message || err.name}. Try reloading or use HTTPS.`;
  el.style.display = "block";
}

// =============================================================================
// CAMERA SELECTOR
// =============================================================================

/**
 * populateCameraSelect — enumerates video-input devices and fills the
 * on-page <select>. The wrapper is revealed only when more than one camera
 * is available. Requires camera permission to have been granted so that
 * device labels are populated.
 *
 * @param {string}   activeDeviceId - The deviceId currently in use, used to
 *                                    pre-select the matching option.
 * @param {Function} onCameraChange - Callback invoked with the new deviceId
 *                                    when the user selects a different camera.
 */
async function populateCameraSelect(activeDeviceId, onCameraChange) {
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
  select.onchange = () => onCameraChange(select.value);
}

// =============================================================================
// COLOUR UTILITIES
// =============================================================================

/**
 * hexToRgba — converts a #rrggbb hex colour and an alpha value to a CSS
 * rgba() string. Accepts only the six-digit hex format.
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
