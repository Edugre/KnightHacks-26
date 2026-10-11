# Windows launcher for laptop work. Mirrors the stub and test modes of run.sh, which needs bash.
# GPU mode only exists in run.sh because the models run on the Linux R9700 server.
#
#   .\run.ps1 stub          stub detector, no models; API + web UI on http://localhost:8000
#   .\run.ps1 stub 8010     same, on another port (use this if 8000 is taken)
#   .\run.ps1 test          run the pytest suite
#   .\run.ps1 bench         benchmark with the stub detector; writes bench\report.md and bench\report.json
#
# One-time setup:
#   uv venv --python 3.12 .venv
#   uv pip install --python .venv\Scripts\python.exe -r requirements.txt
# If Windows blocks the script, run it as:  powershell -ExecutionPolicy Bypass -File .\run.ps1 stub
param([string]$Mode = "stub", [int]$Port = 8000)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Prefer the repo venv (Python 3.12, same as the server); otherwise use whatever python is on PATH.
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONPATH = $PSScriptRoot
New-Item -ItemType Directory -Force logs, data | Out-Null

# The renderer and make_test_clip.py read CENSOR_FFMPEG. Without a system ffmpeg, use the imageio-ffmpeg binary.
if (-not $env:CENSOR_FFMPEG -and -not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $env:CENSOR_FFMPEG = & $py -c "import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())"
}

switch ($Mode) {
    "test" {
        & $py -m pytest -q
        exit $LASTEXITCODE
    }
    "stub" {
        $env:CENSOR_STUB = "1"
        Remove-Item Env:CENSOR_MODEL_URL -ErrorAction SilentlyContinue  # no model server: VLM checks answer "unknown"
        if (-not (Test-Path "bench\clips\synthetic.mp4")) {
            & $py scripts\make_test_clip.py bench\clips\synthetic.mp4
        }
        Write-Host "stub mode: open http://localhost:$Port, then upload bench\clips\synthetic.mp4 and its .tracks.json"
        & $py -m uvicorn censor.server.app:app --host 127.0.0.1 --port $Port
    }
    "bench" {
        # Laptops have no detector models, so the stub replays the synthetic clip's ground truth.
        $env:CENSOR_STUB = "1"
        Remove-Item Env:CENSOR_MODEL_URL -ErrorAction SilentlyContinue
        & $py -m bench.run_bench
        exit $LASTEXITCODE
    }
    default { Write-Error "usage: .\run.ps1 [stub|test|bench] [port]" }
}
