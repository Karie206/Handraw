## Handraw
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Project Structure

```
HANDRAW/
├── camera/
│   ├── captures/           
│   ├── drawing.py         
│   └── hand_landmarker.task 
└── README.md
```

## Demo
<img width="480" height="272" alt="demo_handraw" src="https://github.com/user-attachments/assets/e05a4ea2-47a0-4ff3-98c5-06e40ab77bcf" />

## Install

```bash
pip install opencv-python opencv-contrib-python mediapipe numpy
```

## How to run

```bash
python cam_draw.py
```

## Controls

### Hand Gestures

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Draw |
| ✌️ Index + middle finger up | Erase |
| ✊ Closed hand | Idle |


## Hand Landmarks

MediaPipe detects **21 landmarks** on each hand, numbered 0 to 20:

<img src="https://editor.analyticsvidhya.com/uploads/25204hand_landmarks.png">

| Group | Landmarks | Description |
|---|---|---|
| Wrist | 0 | WRIST |
| Thumb | 1-4 | CMC -> MCP -> IP -> TIP |
| Index finger | 5-8 | MCP -> PIP -> DIP -> TIP |
| Middle finger | 9-12 | MCP -> PIP -> DIP -> TIP |
| Ring finger | 13-16 | MCP -> PIP -> DIP -> TIP |
| Pinky | 17-20 | MCP -> PIP -> DIP -> TIP |

> Each finger has 4 points: base knuckle (MCP), middle knuckle (PIP), upper knuckle (DIP), and fingertip (TIP).


### Keyboard

| Key | Action |
|---|---|
| `1` – `9` | Switch color (green, blue, pink, orange, white, ...) |
| `+` / `-` | Decrease / increase brush size |
| `C` | Clear board |
| `S` | Save screenshot to `captures/` |
| `Tab` | Toggle HUD |
| `Q` / `Esc` | Quit |


## Credits
- [MediaPipe](https://mediapipe.dev) by Google
- [OpenCV](https://opencv.org) by OpenCV.org (open-source community)
