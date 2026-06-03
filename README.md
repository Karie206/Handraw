# 🖐️ Virtual Whiteboard

Draw on your screen using your hand and webcam — no mouse needed.  
Powered by **MediaPipe** + **OpenCV**.

## 📁 Project Structure

```
vboard/
├── cam_draw.py              # Main program
├── hand_landmarker.task     # AI model (auto-downloaded on first run)
├── README.md
└── captures/                # Screenshots saved here (auto-created)
```

## ⚙️ Install

```bash
pip install opencv-python opencv-contrib-python mediapipe numpy
```

## 🚀 Run

```bash
python cam_draw.py
```

> The AI model (~9MB) is downloaded automatically on first run.

## 🎮 Controls

### Hand Gestures

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Draw |
| ✌️ Index + middle finger up | Erase |
| ✊ Closed hand | Idle |

### Keyboard

| Key | Action |
|---|---|
| `1` – `5` | Switch color (green, blue, pink, orange, white) |
| `[` / `]` | Decrease / increase brush size |
| `C` | Clear board |
| `S` | Save screenshot to `captures/` |
| `Tab` | Toggle HUD |
| `Q` / `Esc` | Quit |

## 🐛 Troubleshooting

- **Webcam not opening** — change `open_camera(0, ...)` to `open_camera(1, ...)`
- **Hand not detected** — improve lighting, lower `min_hand_detection_confidence` to `0.4`
- **Low FPS** — reduce `WINDOW_W/H` or `DETECT_W` in the config section
- **Model download fails** — download manually [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and place it next to `cam_draw.py`

---

## 🙏 Credits

- [MediaPipe](https://developers.google.com/mediapipe) by Google
- [OpenCV](https://opencv.org/)
