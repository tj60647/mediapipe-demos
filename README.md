# MediaPipe Demos

**MediaPipe Demos** is a project in the MDes Prototyping course at CCA. It focuses on real-time computer vision in the browser using [MediaPipe](https://google.github.io/mediapipe/) — Google's framework for applying machine-learning models to live webcam feeds.

The hands-on thread running through the whole project is a progressive series of webcam demos:

1. **Hand tracking** — detect and visualise the 21 joints of one or two hands
2. **Face mesh** — map 468 facial landmarks with region-specific colouring
3. **Hands and face combined** — run both models together in a split-screen view

> **🖼️ Gallery:** Browse and launch every demo in the project from the [Code Example Gallery](https://tj60647.github.io/mediapipe-demos/gallery/). *(Also works locally — open `gallery/index.html` directly in your browser.)*

---

## The Tools We Use

This project links two main pieces:

1. **MediaPipe** — Google's real-time machine-learning pipeline that runs entirely in the browser. No data ever leaves your device. Models are loaded from a CDN.
2. **Plain JavaScript + Canvas** — each demo uses vanilla JS and the native HTML `<canvas>` API to draw landmark dots and skeleton lines over a live webcam feed.

### MediaPipe

[MediaPipe](https://google.github.io/mediapipe/) provides pre-trained models for common computer-vision tasks. In this project we use two:

| Model | What it detects | Landmarks |
|---|---|---|
| **Hands** | Up to two hands | 21 points per hand (wrist, knuckles, fingertips) |
| **FaceMesh** | Up to 1 face (configurable) | 468 points covering the full face surface |

Each landmark has normalised `x`, `y`, and `z` coordinates (0.0–1.0 relative to the frame). To draw on a canvas, multiply `x` by the canvas width and `y` by the canvas height.

### OpenProcessing

[OpenProcessing](https://openprocessing.org) is an online platform for running and sharing browser sketches. Every `sketch.js` in this project can be pasted directly into a new OpenProcessing sketch and run there without any local setup.

> **Note:** Paste the contents of any `sketch.js` into a new OpenProcessing sketch to run it there. If you prefer to work locally, open the matching `index.html` file directly in your browser.

---

## Where This Fits in the Course

| Project | Focus |
|---|---|
| ← **Smart Object Foundations** | WebSerial, p5.js, signal processing |
| → **MediaPipe Demos** _(you are here)_ | Real-time computer vision, webcam, landmark tracking |

---

## What You Will Build

A series of real-time browser demos that process your webcam feed with MediaPipe machine-learning models:

```
webcam → video element → MediaPipe model(s) → landmark results → canvas drawing
```

Each demo builds on the last:

```
Demo 1 — hand landmarks only
    ↓ add face mesh
Demo 2 — face landmarks only
    ↓ combine both
Demo 3 — hands + face, side-by-side split view
```

---

## Folder Structure

Each demo has a self-contained folder with an `index.html` page and a `sketch.js` file containing all the JavaScript logic.

```
gallery/
└── index.html                         ← launch pad — links to all demos

demo-1-hand-tracking/
├── index.html                         ← open locally in any modern browser
└── sketch.js                          ← paste into OpenProcessing

demo-2-face-mesh/
├── index.html
└── sketch.js

demo-3-hands-and-face/
├── index.html
└── sketch.js
```

---

## Demo 1 — Hand Tracking

Detect up to two hands in your webcam feed and draw the 21 landmark points per hand along with the full hand skeleton.

**What you see:**
- Green dots at each of the 21 hand joints (wrist, knuckles, fingertips)
- Green lines connecting adjacent joints to form the hand skeleton
- A count of hands currently detected in the corner

📂 **Sketch:** [`demo-1-hand-tracking/sketch.js`](demo-1-hand-tracking/sketch.js)

> **📐 Concept Sidebar: Normalised Coordinates**
>
> MediaPipe returns landmark positions as normalised values in the range 0.0–1.0, not as pixel coordinates. A landmark with `x = 0.5, y = 0.5` is exactly in the centre of the frame, regardless of the actual camera resolution.
>
> To draw on a canvas, convert to pixels:
> ```js
> const pixelX = landmark.x * canvas.width;
> const pixelY = landmark.y * canvas.height;
> ```
> This makes your code resolution-independent — the same sketch works whether your webcam is 640×480 or 1920×1080.

> **📐 Concept Sidebar: The 21 Hand Landmarks**
>
> MediaPipe Hands numbers each landmark 0–20:
>
> | Index | Location |
> |---|---|
> | 0 | Wrist |
> | 1–4 | Thumb (base → tip) |
> | 5–8 | Index finger (base → tip) |
> | 9–12 | Middle finger |
> | 13–16 | Ring finger |
> | 17–20 | Pinky finger |
>
> The `z` component is a depth estimate — smaller (more negative) values mean the point is closer to the camera. Depth can be used to detect when a finger is pressing forward or pulling back.

**Deliverable:** A live canvas showing your hand(s) with green skeleton dots and lines that follow your movements in real time.

---

## Demo 2 — Face Mesh

Map 468 facial landmarks onto a detected face, with distinct colours for the eyes, eyebrows, nose, lips, and irises.

**What you see:**
- 468–478 coloured dots spread across the full face surface (468 base landmarks + 10 iris points when `refineLandmarks: true`)
- Region-specific colouring (blue = eyes, yellow = eyebrows, red = lips, purple = nose, cyan = irises)
- A count of faces currently detected in the corner

📂 **Sketch:** [`demo-2-face-mesh/sketch.js`](demo-2-face-mesh/sketch.js)

> **📐 Concept Sidebar: Landmark Indices and Sub-regions**
>
> The 468 face landmarks are numbered 0–467 (plus indices 468–477 for iris points when `refineLandmarks: true` is set). MediaPipe's canonical face model diagram shows which index maps to which facial feature.
>
> You can isolate any region by checking whether an index belongs to a predefined `Set`:
> ```js
> const EYE_INDICES = new Set([33, 7, 163, ...]);
> if (EYE_INDICES.has(index)) { /* draw in blue */ }
> ```

**Deliverable:** A live canvas showing your face covered in coloured dots that track your movements and expressions.

---

## Demo 3 — Hands and Face Combined

Run both models simultaneously on the same webcam frame, displayed in a split-screen canvas.

**What you see:**
- **Left half** — the raw webcam feed with no overlay
- **Right half** — the same frame with green hand landmarks and red face landmarks drawn on top
- Counts for both hands and faces in the corner

📂 **Sketch:** [`demo-3-hands-and-face/sketch.js`](demo-3-hands-and-face/sketch.js)

> **📐 Concept Sidebar: Running Two Models Per Frame**
>
> Both models process the same video frame inside the Camera's `onFrame` callback using `await`:
> ```js
> await hands.send({ image: video });
> await faceMesh.send({ image: video });
> ```
> Each model calls its own `onResults` callback asynchronously when it finishes. The update flags `newHand` and `newFace` prevent `drawResults()` from being called before either model has returned its first result.

> **📐 Concept Sidebar: Side-by-Side Canvas Layout**
>
> The canvas is created at twice the video width (`video.videoWidth * 2`) so both halves fit side by side:
>
> ```
> ┌──────────────────┬──────────────────┐
> │   x: 0 → width  │  x: width → 2×w  │
> │   (raw feed)     │  (landmark view) │
> └──────────────────┴──────────────────┘
> ```
>
> Landmark coordinates are offset into the right half by adding `video.videoWidth` to every `x` value:
> ```js
> ctx.arc(video.videoWidth + point.x * video.videoWidth, point.y * video.videoHeight, ...)
> ```

**Deliverable:** A live split-screen canvas showing the raw feed next to the landmark overlay for both hands and face simultaneously.

---

## Getting Started Checklist

- [ ] Open the [Gallery](https://tj60647.github.io/mediapipe-demos/gallery/) in your browser (or `gallery/index.html` locally)
- [ ] Click **Demo 1 — Hand Tracking** and allow webcam access
- [ ] Hold your hand up in front of the camera — you should see green dots and lines follow your joints
- [ ] Open **Demo 2 — Face Mesh** — look at the camera and see 468 coloured dots map to your facial features
- [ ] Open **Demo 3 — Hands and Face Combined** — hold your hand near your face and see both models running at once
- [ ] Open `sketch.js` in any demo and read through the comments to understand how each part works
- [ ] Try changing `debugMode = true` to see per-frame log output in the browser DevTools Console

---

## Usage Notes

1. All models are loaded from the [jsDelivr CDN](https://www.jsdelivr.com/) — an internet connection is required.
2. Webcam access requires the browser to be served over HTTPS, or from `localhost`.
3. MediaPipe runs entirely client-side. No video data is sent to any server.
4. For best performance, use Chrome or Edge on a reasonably modern device.
5. To use a sketch in OpenProcessing, paste the `sketch.js` contents into a new sketch and add the MediaPipe CDN `<script>` tags in the OpenProcessing sketch settings.

---

## License

MIT — see [LICENSE](LICENSE) in the root of this repository.
