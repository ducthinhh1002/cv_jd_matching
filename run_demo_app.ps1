param(
  [string]$HostName = "127.0.0.1",
  [int]$Port = 8000,
  [string]$Device = "cuda",
  [switch]$DisableLlmQuestions
)

$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $Python = "python"
}

$ArgsList = @(
  (Join-Path $PSScriptRoot "demo_server.py"),
  "--host", $HostName,
  "--port", $Port,
  "--embedding-device", $Device
)

if ($DisableLlmQuestions) {
  $ArgsList += "--no-llm-questions"
}

& $Python @ArgsList
