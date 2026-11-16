<#
.SYNOPSIS
    Removes what setup-azure.ps1 created.

.EXAMPLE
    .\teardown-azure.ps1 -ResourceGroup DL_ALK_RH -EndpointsOnly    # stop the 24/7 costs, keep everything else
    .\teardown-azure.ps1 -ResourceGroup DL_ALK_RH -Everything       # delete the whole resource group (asks first)
#>
param(
    [Parameter(Mandatory = $true)] [string] $ResourceGroup,
    [string] $Workspace = "dl-alk-ml",
    [switch] $EndpointsOnly,
    [switch] $Everything
)

$ErrorActionPreference = "Stop"
$az = (Get-Command az -ErrorAction SilentlyContinue).Source
if (-not $az) { $az = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" }
if (-not ($EndpointsOnly -or $Everything)) { throw "Choose -EndpointsOnly or -Everything." }

if ($EndpointsOnly) {
    $endpoints = & $az ml online-endpoint list --workspace-name $Workspace --resource-group $ResourceGroup --query "[].name" -o tsv --only-show-errors
    if (-not $endpoints) { Write-Host "No online endpoints - nothing to delete."; return }
    foreach ($e in $endpoints) {
        Write-Host "Deleting endpoint $e ..."
        & $az ml online-endpoint delete --name $e --workspace-name $Workspace --resource-group $ResourceGroup --yes --only-show-errors
    }
    Write-Host "Done. The cluster scales to 0 by itself; the workspace, data and models stay."
    return
}

Write-Host "This deletes EVERYTHING in resource group '$ResourceGroup': workspace, data, models, endpoints, Foundry." -ForegroundColor Yellow
$answer = Read-Host "Type the resource group name to confirm"
if ($answer -ne $ResourceGroup) { Write-Host "Cancelled."; return }

# AI services / Foundry accounts are soft-deleted for 48 h and block their name; purge them so the name can be reused.
$accounts = & $az cognitiveservices account list -g $ResourceGroup --query "[].{name:name, location:location}" -o json --only-show-errors | ConvertFrom-Json
& $az group delete --name $ResourceGroup --yes --only-show-errors
foreach ($a in $accounts) {
    & $az cognitiveservices account purge --name $a.name --resource-group $ResourceGroup --location $a.location --only-show-errors 2>$null
}
Write-Host "Resource group $ResourceGroup deleted."
