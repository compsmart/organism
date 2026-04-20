# Launch an autonomous organism training run on vast.ai.
#
# 1. Finds a cheap GPU offer (or uses one you pass)
# 2. Creates an instance with scripts/onstart.sh embedded as the onstart script
# 3. Prints the instance id so you can track it
#
# Usage:
#   .\launch_training.ps1                    # auto-pick cheapest reliable offer
#   .\launch_training.ps1 -OfferId 12345678  # use a specific offer id
#   .\launch_training.ps1 -Episodes 1000 -RunName 'gpu-big'
#
# After launch, poll with scripts/fetch_checkpoint.ps1 <instance_id>

param(
    [int]$OfferId = 0,
    [int]$Episodes = 500,
    [string]$RunName = "gpu-v1",
    [int]$Seed = 42,
    [string]$Branch = "main"
)

$ApiKey = $env:VAST_API_KEY
if (-not $ApiKey) { $ApiKey = "400f479d9f47c57f187db63791cec2f197442efa351ea4fa5dd904cff0add8e2" }
$Headers = @{ Authorization = "Bearer $ApiKey" }
$Base = "https://console.vast.ai/api/v0"

# Load the onstart script content
$scriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$onstartPath = Join-Path $scriptsDir "onstart.sh"
if (-not (Test-Path $onstartPath)) {
    Write-Error "onstart.sh not found at $onstartPath"
    exit 1
}
$onstart = Get-Content $onstartPath -Raw
# Normalize line endings for bash
$onstart = $onstart -replace "`r`n", "`n"

# Auto-pick offer if not specified
if ($OfferId -eq 0) {
    Write-Host "Searching for cheap reliable GPU offers..."
    $q = '{"num_gpus":"1","gpu_ram":">=12","rentable":"true","order":[["dph_total","asc"]],"limit":20}'
    $searchUrl = "$Base/search/asks/?q=" + [uri]::EscapeDataString($q)
    $r = Invoke-RestMethod -Uri $searchUrl -Headers $Headers -UseBasicParsing
    $offer = $r.offers |
        Where-Object { $_.reliability -ge 0.98 -and $_.inet_down -ge 100 -and $_.cuda_max_good -ge 12 } |
        Sort-Object dph_total |
        Select-Object -First 1
    if (-not $offer) { Write-Error "No suitable offer found"; exit 1 }
    $OfferId = $offer.id
    Write-Host "Selected offer $OfferId: $($offer.gpu_name), $($offer.gpu_ram) MB, `$$($offer.dph_total)/hr"
}

# Build create request
$body = @{
    client_id = 'me'
    image = 'pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime'
    disk = 20
    runtype = 'ssh'
    onstart = $onstart
    env = @{
        ORGANISM_EPISODES = "$Episodes"
        ORGANISM_RUN_NAME = $RunName
        ORGANISM_SEED = "$Seed"
        ORGANISM_BRANCH = $Branch
    }
}

$json = $body | ConvertTo-Json -Depth 10 -Compress
$url = "$Base/asks/$OfferId/"
Write-Host "Creating instance from offer $OfferId..."
$resp = Invoke-RestMethod -Uri $url -Headers $Headers -Method PUT -Body $json -ContentType 'application/json' -UseBasicParsing
$resp | ConvertTo-Json -Depth 5
if ($resp.new_contract) {
    Write-Host ""
    Write-Host "Instance id: $($resp.new_contract)"
    Write-Host "Check status: .\vast.ps1 show $($resp.new_contract)"
    Write-Host "Fetch checkpoint once done: .\fetch_checkpoint.ps1 $($resp.new_contract)"
}
