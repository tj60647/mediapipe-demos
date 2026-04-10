# MediaPipe Demos

**MediaPipe Demos** is a project in the MDes Prototyping course at CCA. It focuses on real-time computer vision in the browser using [MediaPipe](https://google.github.io/mediapipe/) — Google's framework for applying machine-learning models to live webcam feeds.

The hands-on thread running through the whole project is a progressive series of webcam demos:

1. **Hand tracking** — detect and visualise the 21 joints of one or two hands
2. **Face mesh** — map 468 facial landmarks with region-specific colouring
3. **Hands and face combined** — run both models together in a split-screen view
4. **Interaction basics** — turn landmark coordinates into interactive inputs: distance, zones, counts, and mapped values
5. **Face instrument** — use your face as a proximity-triggered control surface with nine named regions

> **🖼️ Gallery:** Browse and launch every demo in the project from the [Code Example Gallery](https://tj60647.github.io/mediapipe-demos/gallery/). *(Also works locally — serve this repo and open `http://localhost:<port>/gallery/`.)*

> **🌐 Beyond Face and Hands:** A comparative overview of five additional MediaPipe tasks — object detection, gesture recognition, pose tracking, image classification, and audio classification — with project prompts and Studio links. Open [`beyond/index.html`](https://tj60647.github.io/mediapipe-demos/beyond/) in your browser.

---

## Run Locally (2 Minutes)

Use a local server so webcam access works reliably.

1. Open a terminal in the repository root.
2. Start a local server:
    ```powershell
    python -m http.server 5500
    ```
3. Open one of these URLs:
    - `http://localhost:5500/gallery/` (start here)
    - `http://localhost:5500/demo-1-hand-tracking/`
    - `http://localhost:5500/demo-2-face-mesh/`
    - `http://localhost:5500/demo-3-hands-and-face/`
    - `http://localhost:5500/demo-4-interaction/`
    - `http://localhost:5500/demo-5-face-instrument/`

If `python` is unavailable, try:

```powershell
py -m http.server 5500
```

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

> **Note:** Paste the contents of any `sketch.js` into a new OpenProcessing sketch to run it there. If you prefer to work locally, serve this repo on `localhost` and open the matching demo URL in your browser.

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
    ↓ add interaction logic
Demo 4 — interaction basics: distance, zones, counts, mapping
    ↓ face as control surface
Demo 5 — face instrument: proximity triggers across face regions
```

---

## Folder Structure

Each demo has a self-contained folder with an `index.html` page and a `sketch.js` file containing all the JavaScript logic.

```
gallery/
└── index.html                         ← launch pad — links to all demos

demo-1-hand-tracking/
├── index.html                         ← open via localhost in any modern browser
└── sketch.js                          ← paste into OpenProcessing

demo-2-face-mesh/
├── index.html
└── sketch.js

demo-3-hands-and-face/
├── index.html
└── sketch.js

demo-4-interaction/
├── index.html
└── sketch.js

demo-5-face-instrument/
├── index.html
└── sketch.js

resources/
└── index.html                         ← exploration sources and links

beyond/
└── index.html                         ← beyond face and hands — five additional tasks
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

Run both models on the same webcam stream in a split-screen canvas (raw feed on the left, overlay on the right).

**What you see:**
- **Left half** — the raw webcam feed with no overlay
- **Right half** — the same frame with green hand landmarks and red face landmarks drawn on top
- Counts for both hands and faces in the corner

📂 **Sketch:** [`demo-3-hands-and-face/sketch.js`](demo-3-hands-and-face/sketch.js)

> **📐 Concept Sidebar: Running Two Models Per Frame**
>
> This demo uses two loops:
> - **Inference loop** (`frameLoop`) sends each frame to Hands first, then FaceMesh, using `await`.
> - **Render loop** (`renderLoop`) redraws at display refresh rate using the latest stored results.
>
> ```js
> await hands.send({ image: video });
> await faceMesh.send({ image: video });
> ```
>
> Sequential sends are intentional for stability in this combined legacy-solution setup.

> **🖐️ Hand Count Control (Demo 3)**
>
> Demo 3 includes a **Hands** selector (1 or 2) above the canvas. This updates `maxNumHands` live so you can trade off robustness vs. multi-hand interaction without reloading.

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

## Demo 4 — Interaction Basics

Turn landmark coordinates into interactive inputs. This demo shows four patterns, all driven by the same webcam data used for drawing in Demos 1–3.

**What you see:**
- Zone overlay: three highlighted columns (LEFT / CENTRE / RIGHT) showing which zone the index fingertip is in
- Proximity line: a dashed coloured line connecting the index fingertip to the nose tip, labelled with the pixel distance
- Colour-coded hand dots: hue shifts from red (0 fingers raised) to green (4 fingers raised)
- Distance bar: right-edge vertical bar showing nose-to-finger distance
- Height strip: left-edge gradient strip with a marker showing the current fingertip height mapped to a hue

📂 **Sketch:** [`demo-4-interaction/sketch.js`](demo-4-interaction/sketch.js)

> **📐 Concept Sidebar: Four Interaction Patterns**
>
> | Pattern | Input | Output |
> |---|---|---|
> | Distance | Two landmark positions | A pixel value (continuous) |
> | Zone | One normalised x coordinate | LEFT / CENTRE / RIGHT (discrete) |
> | Count | Four tip-vs-PIP comparisons | Integer 0–4 (discrete) |
> | Mapping | One normalised y coordinate | A hue value 0–200 (continuous) |
>
> All four patterns convert raw coordinate data into something that can drive a visual, audio, or behavioural response. The key insight: any landmark value can be an input, not just a drawing position.

> **📐 Concept Sidebar: Proximity (Distance)**
>
> Euclidean distance between two landmarks in pixels:
> ```js
> const dx = (a.x - b.x) * canvasWidth;
> const dy = (a.y - b.y) * canvasHeight;
> const dist = Math.sqrt(dx * dx + dy * dy);
> ```
> Use `dist` to trigger effects (is the hand close enough?), map to a range (how close is it?), or smooth over time (is it getting closer or further away?).

**Deliverable:** A live canvas with all four interaction patterns running simultaneously, with labelled HUD values showing their current state.

---

## Demo 5 — Face Instrument

Turn the face into a proximity-triggered control surface. Nine named regions — forehead, eyebrows, eyes, nose, lips, and cheeks — each activate when the index fingertip comes within range.

**What you see:**
- Subtle face mesh dots showing all 468 landmarks
- Dim rings at each of the nine face regions (inactive)
- Bright glowing rings + filled centre when a region is active
- Expanding ripple animation each time a region is newly activated
- A legend panel listing all regions with their current activation state
- The index fingertip highlighted as a larger ring

📂 **Sketch:** [`demo-5-face-instrument/sketch.js`](demo-5-face-instrument/sketch.js)

> **📐 Concept Sidebar: Proximity Trigger**
>
> A proximity trigger converts a continuous distance value into an on/off state:
> ```js
> const dist = Math.sqrt(dx * dx + dy * dy);
> const active = dist < PROXIMITY_THRESHOLD;
> ```
> This is the simplest possible interaction rule. The threshold is tuneable — increase it to make regions easier to activate, decrease it to require the finger to be very close. Noisy tracking becomes a design material rather than a problem: proximity thresholds let you absorb small jitter without false triggers.

> **📐 Concept Sidebar: Rising Edge Detection**
>
> To spawn a visual effect only when a region *first* becomes active (not on every frame while it stays active), compare the current active set to the previous frame's active set:
> ```js
> // prevActive is a Set from the last frame
> if (!prevActive.has(region.name)) {
>   // region just became active — spawn a ripple
>   ripples.push({ x, y, color, radius: 20, alpha: 0.9 });
> }
> prevActive = new Set(currentActive.keys());
> ```
> This is a *rising edge* — it fires once per activation event rather than continuously.

**Deliverable:** A live canvas where moving your index finger near different parts of your face triggers distinct coloured glows and ripples across nine face regions.

---

## Beyond Face and Hands

Once you are comfortable with the demos, there are many more MediaPipe tasks to explore. The [`beyond/index.html`](https://tj60647.github.io/mediapipe-demos/beyond/) page introduces five of them with a comparative frame — what each task detects, what it outputs, what interactions it makes possible, and where it is likely to fail.

| Task | What it detects | Project prompt |
|---|---|---|
| **Object Detection** | Up to 80 labelled object categories with bounding boxes | Domestic Object Orchestra |
| **Gesture Recognition** | Seven built-in hand gestures plus underlying landmarks | Secret Handshake Operating System |
| **Pose Tracking** | 33 full-body landmarks in 3D | Full-Body Puppet Weather System |
| **Image Classification** | Thousands of ImageNet categories from a single frame | Found Image Fortune Teller |
| **Audio Classification** | 521 sound categories from a live microphone | Noisy Room Translator |

Each task card on the Beyond page links directly to MediaPipe Studio so you can try it before writing any code.

---

## How to Explore Tasks Before Coding

Before you commit to implementing a new task, spend 10–15 minutes in **MediaPipe Studio** to test it live with your webcam. This helps you answer three key questions:

1. **Does this model detect what I need?**
2. **Is the output stable enough for my project?**
3. **What settings (confidence, hand count, etc.) make it work best?**

### MediaPipe Studio Session (Guided Workflow)

1. Open [MediaPipe Studio](https://mediapipe-studio.webapps.google.com/) in your browser.
2. Pick one task from the list (start with **Face Landmarker** if you're new):
   - Face Landmarker (good follow-up to Demo 2)
   - Hand Landmarker (good follow-up to Demo 1)
   - Object Detector
   - Gesture Recognizer
   - Pose Landmarker
   - Image Classifier
   - Audio Classifier
3. Allow webcam access and adjust these settings one at a time:
   - **Confidence threshold** — lower it until the model becomes noisy, then raise it until it's stable
   - **Hand count** (if available) — test both 1 and 2 to understand the tradeoff
   - **Running mode** — try both live camera and single image modes
4. Try these things:
   - Move around the room (does it follow?).
   - Dim the lights (does it still detect?).
   - Change your pose or expression dramatically (does it jump or smooth?).
   - Hold an object in front of your face (does it fail gracefully?).
5. **Write down one thing you discovered** — does the model behave the way you expected? What surprised you?

> **📐 Concept Sidebar: Evaluation Before Implementation**
>
> Studio saves you hours of debugging. If a model doesn't behave well in Studio with your actual webcam, implementing it in code won't magically fix it. Use Studio to decide whether a task is right for your project *before* you build.

---

## Learning from Code Examples

Once you know which task you want to use, look at working code to understand the pattern. **CodePen** has near-official MediaPipe examples right in the browser; **OpenProcessing** has community sketches.

### CodePen: MediaPipe Task Examples

The [MediaPipe CodePen profile](https://codepen.io/mediapipe-preview/) has working examples for every task. Each pen shows the current JavaScript API in a clean, runnable format.

**How to use CodePen examples:**

1. Browse [MediaPipe CodePen](https://codepen.io/mediapipe-preview/) to find your task.
2. Click **Fork** to make your own copy.
3. Read the JavaScript panel and look for:
   - How the model is imported and initialized
   - How results are parsed in the callback
   - How landmarks are drawn or used
4. Modify one thing at a time (change colours, thresholds, or drawing logic) and watch the result.
5. **Copy patterns, not the whole code.** You'll learn more by patching it into your own structure than by starting from someone else's sketch.

> **📐 Concept Sidebar: Pattern Over Copy-Paste**
>
> CodePen examples are reference implementations, not templates. The value is in understanding *how* the model is used, not in copying and pasting entire projects. If you're stuck on a specific task — like parsing hand landmarks or handling confidence scores — CodePen is the fastest way to see the answer.

### OpenProcessing: Community Sketches

The [OpenProcessing](https://openprocessing.org) community often remixes the sketches from this project. Use it to:

- See creative uses of the models beyond the demos
- Find variations (different drawing styles, thresholds, layouts)
- Understand what "stable enough" means for your own work

**Suggested search:**
- `mediapipe hand`
- `face landmark`
- `webcam interactive`

Every `sketch.js` in this project can be pasted directly into OpenProcessing, so you can also use your own demos as the baseline for exploration.

---

## AI Remix Workflow

Students can use AI coding assistants to remix patterns across demos while keeping a stable baseline.

### Remix Recipe

1. Start from one working demo in the [Gallery](https://tj60647.github.io/mediapipe-demos/gallery/).
2. Borrow one pattern from the core demos:
   - Demo 3: split view and combined model pipeline
   - Demo 4: distance / zone / count / mapping logic
   - Demo 5: proximity triggers and rising-edge effects
3. Add one prompt from [Beyond Face and Hands](https://tj60647.github.io/mediapipe-demos/beyond/).
4. Ask the assistant for one small change only (one feature per request).
5. Run and test after each change before asking for the next one.

> **📐 Concept Sidebar: Keep One Thing Constant**
>
> When remixing with AI, keep your base demo unchanged except for one clearly scoped addition. If something breaks, you can immediately identify which change caused it and roll back only that part.

### Suggested Prompt Template

```text
I am editing demo-X in mediapipe-demos.
Keep the existing structure and comments.
Add only one feature: <feature>.
Do not rewrite the whole file.
Explain which landmarks, thresholds, and mappings you used.
```

### Remix Deliverable

- [ ] One baseline demo still works exactly as before
- [ ] One new interaction feature works reliably
- [ ] You can explain the landmarks and threshold(s) used
- [ ] Your code still runs from `http://localhost:<port>/...`

---

## Getting Started Checklist

- [ ] Open the [Gallery](https://tj60647.github.io/mediapipe-demos/gallery/) in your browser (or via `http://localhost:<port>/gallery/` when running locally)
- [ ] Click **Demo 1 — Hand Tracking** and allow webcam access — you should see green dots and lines follow your joints
- [ ] Open **Demo 2 — Face Mesh** — look at the camera and see 468 coloured dots map to your facial features
- [ ] Open **Demo 3 — Hands and Face Combined** — hold your hand near your face and see both models running at once
- [ ] Open **Demo 4 — Interaction Basics** — move your index finger around and watch distance/zone/finger-count values change
- [ ] Open **Demo 5 — Face Instrument** — hover your index fingertip near your forehead, eyebrows, nose, lips, and cheeks to activate regions
- [ ] Open `sketch.js` in any demo and read through the comments
- [ ] Try changing `debugMode = true` to see per-frame logs in the browser DevTools Console
- [ ] Open the [Beyond Face and Hands](https://tj60647.github.io/mediapipe-demos/beyond/) page and read the task descriptions
- [ ] Follow the **MediaPipe Studio Session** workflow: pick one new task from the Beyond page and test it live in Studio with your webcam
- [ ] Follow the **Learning from Code Examples** section: find a CodePen example of the same task and read the code
- [ ] **Optional — Open [Exploration Sources](https://tj60647.github.io/mediapipe-demos/resources/)** for a curated list of reference links and tools

---

## Exploration Sources

Before writing code for a new project idea, use these resources to explore what MediaPipe can do and to find reference examples.

| Source | Best for |
|---|---|
| [MediaPipe Studio](https://mediapipe-studio.webapps.google.com/) | Testing any MediaPipe task in the browser with your webcam — no code required. Adjust thresholds, try different inputs, and decide whether a model fits your idea before building. |
| [MediaPipe CodePen](https://codepen.io/mediapipe-preview/) | Near-official working examples for each MediaPipe task using the JavaScript task API. Fork a pen to experiment without a local setup. |
| [OpenProcessing](https://openprocessing.org/) | Sketch culture, remixing, and sharing. Paste any `sketch.js` from this project directly into a new sketch. |
| [p5.js Examples](https://p5js.org/examples/) | Interaction, drawing, animation, and visual design patterns in p5.js. Useful reference when building on top of the demos. |

📂 **Resources page:** [`resources/index.html`](resources/index.html)

---

## Usage Notes

1. All models are loaded from the [jsDelivr CDN](https://www.jsdelivr.com/) — an internet connection is required.
2. Webcam access requires the browser to be served over HTTPS, or from `localhost`.
3. MediaPipe runs entirely client-side. No video data is sent to any server.
4. For best performance, use Chrome or Edge on a reasonably modern device.
5. To use a sketch in OpenProcessing, paste the `sketch.js` contents into a new sketch and add the MediaPipe CDN `<script>` tags in the OpenProcessing sketch settings.

---

## Troubleshooting

1. **No webcam prompt or blank video:** make sure you're opening the demo via `http://localhost:<port>/...` (not `file://...`).
2. **Camera blocked:** click the camera icon in the browser address bar and allow access, then reload.
3. **Old behavior after code changes:** hard refresh with `Ctrl+F5` to clear cached scripts.
4. **Console message about async listener / `installHook.js`:** this is usually from a browser extension, not the demo code.
5. **404 for `/.well-known/appspecific/com.chrome.devtools.json`:** harmless Chrome/DevTools probe, safe to ignore.

---

## License

MIT — see [LICENSE](LICENSE) in the root of this repository.
