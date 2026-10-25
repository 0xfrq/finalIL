# Start a local Ollama server for the CDA-CDM narrative layer and pull Qwen3-4B-Instruct-2507.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_slm.ps1            # serve + pull + smoke test
#   powershell -ExecutionPolicy Bypass -File scripts\run_slm.ps1 -NoServe   # Ollama tray app already running
#
# Ollama publishes Qwen3-4B-Instruct-2507 as the tag "qwen3:4b-instruct" (Q4_K_M, ~2.5 GB).
# The app only talks HTTP to http://localhost:11434/v1; no model is loaded inside Streamlit.
param(
    [string]$Model = "qwen3:4b-instruct",
    [int]$ContextLength = 8192,
    [switch]$NoServe
)
$ErrorActionPreference = "Stop"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama not found. Install it from https://ollama.com/download (Windows) and re-run." -ForegroundColor Red
    exit 1
}

# Ollama's OpenAI-compatible endpoint ignores per-request num_ctx; the context window is a server setting.
# The default (4096) is enough for the compact prompt, 8192 leaves headroom for repairs/long classes.
$env:OLLAMA_CONTEXT_LENGTH = "$ContextLength"
# Bind to localhost only (the app refuses non-local URLs unless allow_remote is configured).
$env:OLLAMA_HOST = "127.0.0.1:11434"

if (-not $NoServe) {
    $running = $false
    try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2 | Out-Null; $running = $true } catch {}
    if ($running) {
        Write-Host "An Ollama server is already running; OLLAMA_CONTEXT_LENGTH only applies to a server started here." -ForegroundColor Yellow
        Write-Host "To change its context window: quit the Ollama tray app, then re-run this script." -ForegroundColor Yellow
    } else {
        Write-Host "Starting 'ollama serve' (context $ContextLength) in a new window ..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Minimized
        for ($i = 0; $i -lt 30; $i++) {
            try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Seconds 1 }
        }
    }
}

Write-Host "Pulling $Model (skipped quickly if already present) ..."
ollama pull $Model

Write-Host "Smoke test via the OpenAI-compatible endpoint ..."
$body = @{ model = $Model; messages = @(@{ role = "user"; content = "Jawab satu kata: siap?" }); max_tokens = 10; temperature = 0.2 } | ConvertTo-Json -Depth 5
$r = Invoke-RestMethod -Uri "http://127.0.0.1:11434/v1/chat/completions" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 180
Write-Host ("Model replied: " + $r.choices[0].message.content)
Write-Host "OK. Run the app with: streamlit run app.py" -ForegroundColor Green
