# Face Swap Backend (Modal)

## Setup

### 1. Install Modal CLI
```bash
pip install modal
modal token new
```

### 2. Create Modal Secret for Supabase
```bash
modal secret create supabase-secret \
  SUPABASE_URL=https://YOUR_PROJECT.supabase.co \
  SUPABASE_KEY=YOUR_SERVICE_ROLE_KEY \
  SUPABASE_BUCKET=face-swap-results
```

### 3. Create the weights volume & download models (run once)
```bash
cd backend
modal run main.py::download_weights
```

### 4. Deploy
```bash
modal deploy main.py
```

## Test locally (with Modal)
```bash
modal run main.py::swap_face \
  --source-image /path/to/face.jpg \
  --target-video /path/to/video.mp4
```

## File structure
```
backend/
├── main.py           # Modal app + entry point
├── storage.py        # Supabase upload
├── requirements.txt  # Reference only (installed inside Modal image)
└── pipeline/
    ├── extract.py    # FFmpeg frame extraction
    ├── detect.py     # InsightFace face detection
    ├── swap.py       # inswapper_128 face swap
    ├── restore.py    # CodeFormer + histogram match
    └── rebuild.py    # FFmpeg video reassembly
```

## Environment variables (via Modal Secret)
| Variable | Description |
|---|---|
| `SUPABASE_URL` | `https://xxxx.supabase.co` |
| `SUPABASE_KEY` | Service role key |
| `SUPABASE_BUCKET` | Storage bucket name (default: `face-swap-results`) |
