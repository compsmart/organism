# vast.ai REST API helper for environments where the Python CLI can't reach console.vast.ai
# but PowerShell Invoke-RestMethod can (corporate proxy uses Windows HTTP stack).
#
# Usage:
#   .\vast.ps1 search           # list cheap reliable GPU offers
#   .\vast.ps1 create <id>      # create instance from offer id
#   .\vast.ps1 list             # list your instances
#   .\vast.ps1 show <id>        # show instance details (ssh info, status)
#   .\vast.ps1 destroy <id>     # destroy instance

param(
    [Parameter(Position=0, Mandatory=$true)] [string]$Action,
    [Parameter(Position=1)] [string]$Id,
    [Parameter(Position=2)] [string]$Extra
)

$ApiKey = $env:VAST_API_KEY
if (-not $ApiKey) { $ApiKey = "400f479d9f47c57f187db63791cec2f197442efa351ea4fa5dd904cff0add8e2" }
$Headers = @{ Authorization = "Bearer $ApiKey" }
$Base = "https://console.vast.ai/api/v0"

function Invoke-Vast([string]$Method, [string]$Path, $Body = $null) {
    $url = "$Base$Path"
    if ($Body) {
        $json = $Body | ConvertTo-Json -Compress -Depth 10
        Invoke-RestMethod -Uri $url -Headers $Headers -Method $Method -Body $json -ContentType 'application/json' -UseBasicParsing
    } else {
        Invoke-RestMethod -Uri $url -Headers $Headers -Method $Method -UseBasicParsing
    }
}

switch ($Action) {
    'search' {
        # Cheap, reliable, fast-network 1-GPU offers
        $q = '{"num_gpus":"1","gpu_ram":">=12","rentable":"true","order":[["dph_total","asc"]],"limit":20}'
        $url = "$Base/search/asks/?q=" + [uri]::EscapeDataString($q)
        $r = Invoke-RestMethod -Uri $url -Headers $Headers -UseBasicParsing
        $r.offers | Where-Object { $_.reliability -ge 0.98 -and $_.inet_down -ge 100 } |
            Select-Object -First 10 id, gpu_name, gpu_ram, dph_total, cpu_cores, cpu_ram, inet_down, reliability, cuda_max_good |
            Format-Table -AutoSize
    }
    'create' {
        if (-not $Id) { Write-Error "Usage: vast.ps1 create <offer_id> [image]"; exit 1 }
        $image = if ($Extra) { $Extra } else { 'pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime' }
        $body = @{
            client_id = 'me'
            image = $image
            disk = 20
            runtype = 'ssh'
            onstart = ''
        }
        $r = Invoke-Vast 'PUT' "/asks/$Id/" $body
        $r | ConvertTo-Json -Depth 5
    }
    'list' {
        $r = Invoke-Vast 'GET' '/instances/'
        $r.instances | Select-Object id, actual_status, gpu_name, ssh_host, ssh_port, dph_total, image_uuid |
            Format-Table -AutoSize
    }
    'show' {
        if (-not $Id) { Write-Error "Usage: vast.ps1 show <instance_id>"; exit 1 }
        $r = Invoke-Vast 'GET' "/instances/"
        $inst = $r.instances | Where-Object { $_.id -eq [int]$Id }
        if (-not $inst) { Write-Error "Instance $Id not found"; exit 1 }
        $inst | Select-Object id, actual_status, status_msg, gpu_name, ssh_host, ssh_port, public_ipaddr, dph_total, image_uuid, cur_state |
            Format-List
    }
    'destroy' {
        if (-not $Id) { Write-Error "Usage: vast.ps1 destroy <instance_id>"; exit 1 }
        $r = Invoke-Vast 'DELETE' "/instances/$Id/"
        $r | ConvertTo-Json -Depth 5
    }
    default {
        Write-Host "Usage: vast.ps1 [search|create|list|show|destroy] [id] [extra]"
        exit 1
    }
}
