# Vehicle Smoke Detection App

Start the app by double-clicking:

```text
run_app.bat
```

Clean folder structure:

- `frontend` - Streamlit screen/UI
- `backend` - split backend files for YOLO, Florence-2, reports, drawing, and video logic
- `models` - trained model files such as `best.pt`, `bestmt.pt`, and local Florence-2 files
- `outputs` - saved video report frames created while using the app
- `support` - small runtime settings/cache used by libraries

Current pipeline: YOLO detects smoke boxes, then Florence-2 classifies smoke color.

Files in the main folder:

- `run_app.bat` - double-click app launcher
- `requirements.txt` - Python package list
- `README.md` - this guide

Do not move `.venv`; it is the local Python environment used by the launcher.

Unused old files were moved outside this folder into a Desktop backup folder.

Main files you usually care about:

- `frontend/app.py` - frontend
- `backend/app.py` - backend entry point
- `backend/engine.py` - main AI detection/classification engine
- `backend/video_pipeline.py` - video report and saved-frame logic
