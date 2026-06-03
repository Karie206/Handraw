## Hanraw
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Project Structure

```
vboard/
├── cam_draw.py              # Main program
├── hand_landmarker.task     # AI model (auto-downloaded on first run)
├── README.md
└── captures/                # Screenshots saved here (auto-created)
```

## Install

```bash
pip install opencv-python opencv-contrib-python mediapipe numpy
```

## Run

```bash
python cam_draw.py
```

> The AI model ~9MB is downloaded automatically on first run

## Controls

### Hand Gestures

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Draw |
| ✌️ Index + middle finger up | Erase |
| ✊ Closed hand | Idle |


## Hand Landmarks

MediaPipe detects **21 landmarks** on each hand, numbered 0 to 20:

<img src="https://techvidvan.com/tutorials/wp-content/uploads/sites/2/2021/07/hand-landmarks.jpg" alt="Hand Landmarks Diagram">

| Group | Landmarks | Description |
|---|---|---|
| Wrist | 0 | WRIST |
| Thumb | 1–4 | CMC → MCP → IP → TIP |
| Index finger | 5–8 | MCP → PIP → DIP → TIP |
| Middle finger | 9–12 | MCP → PIP → DIP → TIP |
| Ring finger | 13–16 | MCP → PIP → DIP → TIP |
| Pinky | 17–20 | MCP → PIP → DIP → TIP |

> Each finger has 4 points: base knuckle (MCP), middle knuckle (PIP), upper knuckle (DIP), and fingertip (TIP).


### Keyboard

| Key | Action |
|---|---|
| `1` – `5` | Switch color (green, blue, pink, orange, white) |
| `[` / `]` | Decrease / increase brush size |
| `C` | Clear board |
| `S` | Save screenshot to `captures/` |
| `Tab` | Toggle HUD |
| `Q` / `Esc` | Quit |

## Troubleshooting

- **Webcam not opening** — change `open_camera(0, ...)` to `open_camera(1, ...)`
- **Hand not detected** — improve lighting, lower `min_hand_detection_confidence` to `0.4`
- **Low FPS** — reduce `WINDOW_W/H` or `DETECT_W` in the config section
- **Model download fails** — download manually [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and place it next to `cam_draw.py`

## Credits

- [MediaPipe](https://developers.google.com/mediapipe) by Google
- [OpenCV](https://opencv.org/)
