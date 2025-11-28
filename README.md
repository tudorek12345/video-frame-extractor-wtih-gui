# Video Frame Extractor with GIF Export

Modern PySide6 desktop app for inspecting videos frame-by-frame, extracting frames, and building experimental animated GIFs — made by Tudor.

## Features
- Load common video formats (mp4, mov, avi, mkv, wmv, m4v).
- Frame-by-frame navigation with timeline slider, play/pause, ±1 and ±10 jumps.
- Dark-themed preview that preserves aspect ratio on resize.
- Metadata panel: path, resolution, FPS, total frames, duration.
- Frame extraction: current frame, all frames, or custom ranges with filename patterns and PNG/JPG output.
- Background workers for extraction/GIF export with progress bars and cancel buttons (UI stays responsive).
- GIF export (experimental): full video or range, custom FPS, scaling, frame skipping, loop count, size clamping for safety.

## Requirements
- Python 3.8+
- Windows 10/11 (cross-platform where supported)
- Dependencies: PySide6, opencv-python, imageio, numpy, Pillow

## Installation
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Usage
1. Launch the app and click **Open Video** to choose a file.
2. Navigate frames with **Prev/Next**, **Jump -10/+10**, play/pause, or drag the slider.
3. Set output directory, filename pattern (e.g., `frame_{index:05d}`), and format (png/jpg).
4. Extract:
   - **Extract Current Frame** for the visible frame.
   - **Extract All Frames** for the whole video (warns on huge jobs).
   - **Extract Frame Range** after setting start/end.
5. GIF Export (Experimental):
   - Pick full video or specify start/end.
   - Adjust GIF FPS, scale %, frame skip, and loop count.
   - Choose output `.gif` path and click **Export GIF**.

## Notes & Limitations
- GIF export clamps very large jobs (defaults to 800 frames) to avoid oversized files; long videos may need higher frame skip or lower scale.
- GIF color quantization and compression are handled by `imageio`; expect quality/size trade-offs.
- Output directories must be writable; the app can create missing folders on demand.

## Troubleshooting
- **Imports not found in IDE**: ensure your selected interpreter has the requirements installed (VS Code → Python: Select Interpreter, then pip install).
- **Video won’t open**: verify the codec/format is supported by your OpenCV build and the file path is accessible.
- **UI freeze**: long operations should stay responsive; if they don’t, cancel and retry (very large videos can still be heavy).

## License

