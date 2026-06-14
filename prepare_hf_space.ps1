param(
  [string]$OutputDir = ".\deploy_hf_space"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Target = Resolve-Path -LiteralPath $Root
$Target = Join-Path $Target.Path $OutputDir

if (Test-Path -LiteralPath $Target) {
  Remove-Item -LiteralPath $Target -Recurse -Force
}

New-Item -ItemType Directory -Path $Target | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Target "web") | Out-Null

Copy-Item -LiteralPath (Join-Path $Root "Dockerfile") -Destination $Target
Copy-Item -LiteralPath (Join-Path $Root "requirements-demo.txt") -Destination $Target
Copy-Item -LiteralPath (Join-Path $Root ".dockerignore") -Destination $Target
Copy-Item -LiteralPath (Join-Path $Root "demo_server.py") -Destination $Target
Copy-Item -LiteralPath (Join-Path $Root "score_candidates.py") -Destination $Target
Copy-Item -LiteralPath (Join-Path $Root "prepare_external_benchmark.py") -Destination $Target
Copy-Item -Path (Join-Path $Root "web\*") -Destination (Join-Path $Target "web") -Recurse

@"
---
title: CV-JD Matching Demo
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# CV-JD Matching Demo

This Space hosts the CV-JD matching web demonstration.
"@ | Set-Content -LiteralPath (Join-Path $Target "README.md") -Encoding UTF8

Write-Host "Prepared Hugging Face Space files at: $Target"
