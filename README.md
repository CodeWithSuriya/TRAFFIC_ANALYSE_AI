# SmartTraffic AI

AI Traffic Monitoring & Vehicle Analytics with:
- YOLO vehicle detection
- ByteTrack object tracking
- Line-crossing counts by vehicle category
- Approximate road-region occupancy
- Simple congestion status and occupancy trend warning
- Streamlit dashboard
- Hourly historical analytics saved to CSV

## Requirements
- Windows 10/11
- Python 3.10 or 3.11 recommended
- Internet for first-time model download

## Install
Open PowerShell in this folder:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will show a local URL, usually http://localhost:8501.

Upload an MP4/AVI/MOV/MKV traffic video, then click **Run traffic analysis**.

## Output
- `output/<video>_annotated.mp4`: processed video
- `output/traffic_history.csv`: sampled history used for hourly charts

## Important limitations
- Occupancy is calculated from the combined image areas of detected bounding boxes within a rectangular region. It is not true physical road occupancy and may be affected by perspective and overlapping boxes.
- Congestion status uses configurable threshold rules, not a trained congestion prediction model.
- Trend compares current sampled occupancy with the previous sample. It is an early warning heuristic, not a guaranteed forecast.
- Historical timestamps are the time the clip was analyzed. `video_time_seconds` records where the sample came from within the clip. If videos are analyzed at different times, historical charts group by analysis time.
- Counts can be affected by missed detections, occlusion, or tracking ID changes.
- Use authorized traffic footage and do not use this prototype to control real traffic signals.
