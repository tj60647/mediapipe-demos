# Tutorial Extension Roadmap

This roadmap extends the existing MediaPipe tutorial from landmark visualisation into interactive prototyping for non-expert coders. The goal is to keep the current progression simple, browser-based, and playful while introducing a broader view of what MediaPipe can do.

## Extension Goals

1. Move from tracking to interaction.
2. Keep everything beginner-friendly and fast to prototype.
3. Show that MediaPipe is not only for face and hands.
4. Introduce MediaPipe Studio as a low-friction way to explore models before coding.
5. Expand students' reference sources beyond OpenProcessing and the p5.js gallery.

## Proposed Structure

### Phase 1: From Seeing to Responding

**Purpose:** bridge the current demos into interaction design.

**New teaching focus:**
- landmark positions as inputs, not just visuals
- simple interaction rules: distance, overlap, speed, count, direction
- turning webcam data into triggers, sliders, switches, and playful controls

**Suggested lesson additions:**
- detect when a fingertip enters a zone on screen
- detect distance between hand and face landmarks
- use openness or closeness as a control value
- map motion to colour, sound, brush size, or animation speed

**Outcome:** students understand that MediaPipe landmarks can drive behaviour, not just drawings.

### Phase 2: Face + Hands Mini Interaction Unit

**Purpose:** extend the current hands-and-face demo into small interactive systems.

**Suggested topics:**
- finger-to-face interactions such as touching nose, eyebrow, or cheek regions
- mirrored interfaces and how webcam interaction feels different from mouse input
- designing for noisy, imperfect tracking rather than fighting it
- threshold tuning for stable interactions

**Suggested exercises:**
- trigger a visual effect when the index finger approaches the lips
- create a "face instrument" where different facial regions trigger different sounds or colours
- use both hands as two independent controllers while the face acts as a third input

**Outcome:** students can combine multiple tracked regions into one expressive prototype.

### Phase 3: MediaPipe Studio and Face Landmark Detection

**Purpose:** introduce Studio as a no-code or low-code exploration tool before students commit to implementation.

**Why it matters:**
- students can try tasks in the browser with their own webcam or uploaded media
- students can inspect outputs before writing code
- students can experiment with confidence thresholds and other settings visually
- it lowers the barrier for students who are still uncertain about coding

**What to cover:**
- what MediaPipe Studio is: a browser-based environment for trying MediaPipe solutions and adjusting settings
- how to use it to test face landmark detection on live input
- what the Face Landmarker gives you beyond dots: landmarks, facial transformation data, and blendshape scores for expressions
- how Studio helps answer practical questions early:
  - does this model detect what I need?
  - is the result stable enough?
  - what settings make it usable?

**Recommended teaching flow:**
1. Open MediaPipe Studio and try Face Landmarker with a webcam.
2. Observe the landmark mesh and expression-related outputs.
3. Adjust settings such as face count and confidence thresholds.
4. Ask students to describe one interaction idea they discovered before writing any code.
5. Then return to JavaScript and build a simplified version.

**Outcome:** students learn to evaluate model behaviour first, then prototype intentionally.

### Phase 4: MediaPipe Studio for Interactive Applications More Broadly

**Purpose:** frame Studio as a sketchbook for interaction design, not just a technical demo page.

**Key message:** MediaPipe Studio is useful at the concept stage because it lets students audition sensing capabilities before deciding what to build.

**Points to emphasise:**
- use Studio to compare tasks quickly
- use Studio to test whether a webcam, image, audio clip, or object is detectable enough for a project idea
- use Studio to help students choose between tasks that sound similar but behave differently
- use Studio as a critique tool: students can show what the model sees, not just what they intended

**Suggested class activity:**
- each student explores two Studio tasks and reports back:
  - what input it expects
  - what output it returns
  - one possible interaction idea
  - one limitation or surprise

**Outcome:** students build model literacy, not just coding habits.

### Phase 5: Guided Exploration Sources

**Purpose:** make exploration part of the tutorial instead of an afterthought.

**Add CodePen as a new source for examples:**
- introduce the MediaPipe preview profile: https://codepen.io/mediapipe-preview/
- position it alongside OpenProcessing and the p5.js gallery as a place to inspect working examples
- explain why it is useful:
  - many examples are close to official task APIs
  - students can fork examples quickly
  - it is good for comparing multiple MediaPipe tasks side by side

**Suggested framing in the tutorial:**
- OpenProcessing: good for sketch culture and remixing
- p5.js gallery: good for interaction and visual inspiration
- MediaPipe CodePen: good for official or near-official task examples and API patterns

**Recommended exploration exercise:**
- ask students to find one example from each source and answer:
  - what is the input?
  - what is the output?
  - what interaction rule makes it interesting?
  - what could be changed for a course project?

**Outcome:** students learn how to research examples instead of waiting for a complete tutorial.

### Phase 6: Beyond Face and Hands

**Purpose:** broaden the tutorial into a menu of sensing options for final projects.

**Suggested topics to introduce briefly:**
- object detection
- hand gesture recognition
- pose or holistic tracking
- image classification
- audio classification

**Teaching strategy:** keep this section comparative rather than deep. The goal is not to fully teach every task, but to show students that the same prototyping mindset applies across MediaPipe.

**Useful comparison prompts:**
- what does this task detect?
- what kind of input does it need?
- what kind of output does it return?
- what kind of interaction becomes possible?
- what are the likely failure cases?

**Outcome:** students can choose a task based on a design intention, not just familiarity.

## Three Project Prompts Using Face + Hands

These are intended to feel playful, unexpected, and achievable within a prototyping course.

### 1. Invisible Makeup Synth

Use your fingers as tools to "paint" reactive effects onto your tracked face. Different facial regions produce different audiovisual responses: eyebrows trigger one texture, cheeks another, lips another. The project becomes a performance instrument rather than a beauty filter.

**Prompt for students:**
Build a face-and-hands prototype where touching or approaching different parts of your face causes strange visual or sonic transformations. Avoid realistic makeup or beauty logic. Make it feel uncanny, theatrical, funny, or impossible.

### 2. Mouth DJ / Hand Conducted Voice Machine

Use face landmarks and hand distance to control a playful voice or sound system. Opening the mouth could raise intensity, a finger near the lips could switch samples, and hand height could change pitch or reverb.

**Prompt for students:**
Design a musical or vocal instrument controlled by your face and hands together. The audience should understand that your body is the interface, but the result should be surprising rather than literal.

### 3. Self-Portrait Creature Generator

Turn the user's face and hands into a live creature that grows extra eyes, antennae, mirrored limbs, or animated features based on tracked motion and spatial relationships.

**Prompt for students:**
Create a live self-portrait system where your face and hands are remixed into a weird character. Use tracking data to control when features appear, stretch, split, or mutate.

## Project Prompts Using Other MediaPipe Tasks

### Object Detection

**Project idea:** Domestic Object Orchestra

Students place everyday objects in front of the camera and assign each detected object a sound, animation, or role in a scene.

**Prompt for students:**
Build a system where common objects become controllers for a playful composition, game, or performance. Make the system reveal something strange or funny about ordinary things.

### Gesture Recognition

**Project idea:** Secret Handshake Operating System

Recognised gestures trigger different interface states, hidden messages, or scene changes.

**Prompt for students:**
Create an interface that only responds to a set of deliberate gestures. Treat gestures as spells, rituals, commands, or social signals rather than simple buttons.

### Audio Classification

**Project idea:** Noisy Room Translator

Ambient sounds such as claps, taps, voices, or room noise change visuals or game logic.

**Prompt for students:**
Make an interactive piece that listens to the room and reacts to its sound character. Focus on atmosphere, rhythm, or misinterpretation rather than accuracy.

### Image Classification

**Project idea:** Found Image Fortune Teller

Students show printed or found images to the camera and the system misreads them into generated messages, story fragments, or animations.

**Prompt for students:**
Build a project where image recognition becomes a storytelling engine. The system should feel opinionated, poetic, or absurd rather than purely informative.

### Pose or Holistic Tracking

**Project idea:** Full-Body Puppet Weather System

Body posture changes the environment, such as wind, rain, particles, or lighting.

**Prompt for students:**
Use whole-body tracking to build an environment that reacts to posture, balance, or movement quality. Think of the body as a weather controller, stage effect controller, or scene generator.

## Recommended Delivery Format

To keep the extension manageable, structure it as a short second arc after the existing three demos.

### Suggested sequence

1. **Current demos**
   Hand tracking, face mesh, combined hands + face.
2. **Interaction basics**
   Distances, thresholds, zones, triggers, mapped values.
3. **Face + hands mini-projects**
   Small playful sketches using the existing codebase as a base.
4. **MediaPipe Studio session**
   Explore Face Landmarker, then compare one or two other tasks.
5. **Exploration sources session**
   OpenProcessing, p5.js gallery, and MediaPipe CodePen.
6. **Other-task prompts**
   Short presentations of object detection, gesture recognition, audio classification, and more.
7. **Independent prototype sprint**
   Students propose and build one small interactive experiment.

## Suggested Deliverables for Students

- one modified face-and-hands sketch based on the existing demos
- one short reflection on a MediaPipe Studio experiment
- one reference roundup using OpenProcessing, p5.js gallery, and MediaPipe CodePen
- one proposal for a final prototype using any MediaPipe task

## Teaching Notes

- Keep the emphasis on behaviour and interaction, not technical completeness.
- Encourage students to prototype with rough thresholds and obvious feedback first.
- Treat model errors as design material, not just bugs.
- Keep examples short and remixable.
- Use Studio and CodePen to reduce setup friction for hesitant coders.

## Summary

The extension should shift the tutorial from "here is how to track landmarks" to "here is how to turn machine perception into an interactive material." The cleanest path is:

1. start with the existing hand and face demos
2. add interaction logic using distance, zones, and triggers
3. introduce MediaPipe Studio for fast exploration and model literacy
4. add CodePen as a practical research source alongside OpenProcessing and the p5.js gallery
5. broaden the field with other MediaPipe tasks so students can develop original project directions