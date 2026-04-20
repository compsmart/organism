# Poll a vast.ai training instance for upload URLs, then download the checkpoint.
#
# The onstart.sh script on the instance uploads the trained checkpoint to 0x0.st
# and echoes the URL to stdout with a sentinel line "ORGANISM_UPLOAD_COMPLETE".
# We retrieve the instance log via the vast.ai API and parse out the URLs.
#
# Usage:
#   .\fetch_checkpoint.ps1 -InstanceId 12345678
#   .\fetch_checkpoint.ps1 -InstanceId 12345678 -RunName gpu-v1
#   .\fetch_checkpoint.ps1 -InstanceId 12345678 -Destroy  # destroy after download

param(
    [Parameter(Position=0, Mandatory=$true)] [int]$InstanceId,
    [string]$RunName = "gpu-v1",
    [switch]$Destroy,
    [int]$PollIntervalSec = 60,
    [int]$MaxWaitMinutes = 120
)

$ApiKey = $env:VAST_API_KEY
if (-not $ApiKey) { $ApiKey = "400f479d9f47c57f187db63791cec2f197442efa351ea4fa5dd904cff0add8e2" }
$Headers = @{ Authorization = "Bearer $ApiKey" }
$Base = "https://console.vast.ai/api/v0"

function Get-InstanceLog($id) {
    # Request a fresh log snapshot
    $resp = Invoke-RestMethod -Uri "$Base/instances/request_logs/$id/" -Headers $Headers -Method PUT -UseBasicParsing
    if (-not $resp.success) { return $null }
    $url = $resp.result_url
    if (-not $url) { return $null }
    # Small delay for the log to be uploaded
    Start-Sleep -Seconds 5
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing
        $content = $resp.Content
        if ($content -is [byte[]]) {
            $content = [System.Text.Encoding]::UTF8.GetString($content)
        }
        return $content
    } catch {
        return $null
    }
}

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$outDir = Join-Path $projectRoot "outputs\$RunName"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$deadline = (Get-Date).AddMinutes($MaxWaitMinutes)
$modelUrl = $null
$metricsUrl = $null
$evalsUrl = $null

Write-Host "Polling instance $InstanceId for training completion (max ${MaxWaitMinutes}m)..."
while ((Get-Date) -lt $deadline) {
    $log = Get-InstanceLog $InstanceId
    if ($log) {
        if ($log -match 'MODEL_URL=(\S+)') { $modelUrl = $Matches[1] }
        if ($log -match 'METRICS_URL=(\S+)') { $metricsUrl = $Matches[1] }
        if ($log -match 'EVALS_URL=(\S+)') { $evalsUrl = $Matches[1] }
        if ($modelUrl) { break }
    }
    Write-Host "  $(Get-Date -Format 'HH:mm:ss') — still training..."
    Start-Sleep -Seconds $PollIntervalSec
}

if (-not $modelUrl) {
    Write-Error "Training did not complete within ${MaxWaitMinutes}m. Check logs manually."
    exit 1
}

Write-Host ""
Write-Host "Downloading checkpoint from $modelUrl..."
Invoke-WebRequest -Uri $modelUrl -OutFile (Join-Path $outDir "model.pt") -UseBasicParsing
Write-Host "Downloading metrics from $metricsUrl..."
Invoke-WebRequest -Uri $metricsUrl -OutFile (Join-Path $outDir "metrics.jsonl") -UseBasicParsing
Write-Host "Downloading evaluations from $evalsUrl..."
Invoke-WebRequest -Uri $evalsUrl -OutFile (Join-Path $outDir "evaluations.jsonl") -UseBasicParsing

Write-Host ""
Write-Host "Download complete. Files:"
Get-ChildItem $outDir | Format-Table Name, Length, LastWriteTime

if ($Destroy) {
    Write-Host "Destroying instance $InstanceId..."
    Invoke-RestMethod -Uri "$Base/instances/$InstanceId/" -Headers $Headers -Method DELETE -UseBasicParsing
    Write-Host "Instance destroyed."
}
