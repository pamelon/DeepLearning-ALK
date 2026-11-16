<#
.SYNOPSIS
    Recreates the whole Azure setup for Class 5 (and the language model for Class 6) from scratch.

.DESCRIPTION
    Creates, in this order:
      1. resource providers registration (needed once per subscription)
      2. resource group
      3. Azure ML workspace (+ storage, key vault, monitoring - created automatically)
      4. CPU compute cluster (scales to 0 when idle)
      5. environment (Keras + TensorFlow + MLflow)
      6. California housing data (local CSV/MLTable) + data assets
      7. Azure AI Foundry resource (AI services) + project + model deployments
         (gpt-4.1-mini, gpt-image-1.5, whisper, tts, text-embedding-3-small, Phi-4-mini) + roles for you
      8. experiments: MNIST job, hyperparameter sweep, AutoML
      9. the best AutoML model as a REST API (online endpoint) - after AutoML finishes

    Everything is idempotent enough to re-run: existing resources are reused.
    Run it from this folder (Classes/Class-5/code/azure-ml), after `az login`.

.EXAMPLE
    .\setup-azure.ps1 -ResourceGroup DL_ALK_RH
    .\setup-azure.ps1 -ResourceGroup DL_ALK_RH -SkipEndpoint            # no 24/7 costs
    .\setup-azure.ps1 -ResourceGroup DL_ALK_RH -DryRun                   # only print the commands
#>
param(
    [Parameter(Mandatory = $true)] [string] $ResourceGroup,
    [string] $Location = "northeurope",          # Azure ML + cluster + endpoint
    [string] $Workspace = "dl-alk-ml",
    [string] $ClusterSize = "Standard_E4s_v3",   # choose a VM family you have quota for
    [int]    $ClusterMaxNodes = 4,
    [string] $FoundryLocation = "swedencentral", # where gpt-4.1-mini is available as Data Zone EU
    [string] $FoundryName = "",                  # globally unique; generated if empty
    [string] $EndpointName = "",                 # unique in the region; generated if empty
    [string] $EndpointVmSize = "Standard_E2s_v3",
    [switch] $SkipFoundry,
    [switch] $SkipExperiments,
    [switch] $SkipEndpoint,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# The Azure CLI (az) must be installed and logged in. If your antivirus inspects HTTPS
# (e.g. Norton), point REQUESTS_CA_BUNDLE to a bundle that includes its root certificate.
$az = (Get-Command az -ErrorAction SilentlyContinue).Source
if (-not $az) { $az = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" }
if (-not (Test-Path $az)) { throw "Azure CLI not found. Install it: winget install Microsoft.AzureCLI" }

function Step($text) { Write-Host "`n=== $text" -ForegroundColor Cyan }

function Az {
    # Runs an az command, prints it, stops on errors. Returns the output (as text).
    Write-Host ("az " + ($args -join " ")) -ForegroundColor DarkGray
    if ($DryRun) { return "" }
    $out = & $az @args --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw "az command failed (exit code $LASTEXITCODE)" }
    return $out
}

function Suffix { -join ((97..122) | Get-Random -Count 4 | ForEach-Object { [char]$_ }) }
$ws = @("--workspace-name", $Workspace, "--resource-group", $ResourceGroup)

Step "0. Checking the Azure CLI login"
if (-not $DryRun) {
    $account = & $az account show --query "{name:name, id:id, tenant:tenantId, user:user.name}" -o json --only-show-errors | ConvertFrom-Json
    if (-not $account) { throw "Not logged in. Run: az login --tenant <your-tenant-id>" }
    Write-Host "Subscription: $($account.name) ($($account.id)) as $($account.user)"
}
Az extension add --name ml --upgrade | Out-Null

Step "1. Registering resource providers (once per subscription)"
foreach ($p in "Microsoft.MachineLearningServices", "Microsoft.CognitiveServices", "Microsoft.Storage",
               "Microsoft.KeyVault", "Microsoft.ContainerRegistry", "Microsoft.Insights",
               "Microsoft.OperationalInsights", "Microsoft.Network", "Microsoft.Cdn", "Microsoft.PolicyInsights") {
    Az provider register --namespace $p --wait | Out-Null
}

Step "2. Resource group $ResourceGroup ($Location)"
Az group create --name $ResourceGroup --location $Location --query "properties.provisioningState" -o tsv

Step "3. Azure ML workspace $Workspace"
Az ml workspace create --name $Workspace --resource-group $ResourceGroup --location $Location `
    --display-name "Deep Learning ALK" --query name -o tsv

Step "4. CPU cluster cpu-cluster ($ClusterSize, 0-$ClusterMaxNodes nodes)"
Az ml compute create --name cpu-cluster --type AmlCompute --size $ClusterSize --min-instances 0 `
    --max-instances $ClusterMaxNodes --idle-time-before-scale-down 120 @ws --query name -o tsv

Step "5. Environment keras-tf-cpu (the image is built on first use, ~10-15 min)"
Az ml environment create --file environments/keras-cpu.yml @ws --query "{name:name, version:version}" -o tsv

Step "6. California housing data + data assets"
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not (Test-Path "data/california/train/MLTable")) {
    if (-not $python) { throw "Python is needed to prepare the data: python prepare_automl_data.py" }
    Write-Host "python prepare_automl_data.py" -ForegroundColor DarkGray
    if (-not $DryRun) { & $python prepare_automl_data.py; if ($LASTEXITCODE -ne 0) { throw "data preparation failed" } }
}
foreach ($split in "train", "validation", "test") {
    $exists = if ($DryRun) { $null } else { & $az ml data show --name "california-housing-$split" --version 1 @ws --query name -o tsv --only-show-errors 2>$null }
    if (-not $exists) {
        Az ml data create --name "california-housing-$split" --version 1 --type mltable `
            --path "./data/california/$split" --description "California housing - $split split" @ws --query name -o tsv
    } else { Write-Host "data asset california-housing-$split already exists" }
}

if (-not $SkipFoundry) {
    if (-not $FoundryName) {
        # (no "|" or "&" inside --query: az.cmd would hand them to the Windows shell)
        $existing = if ($DryRun) { $null } else { & $az cognitiveservices account list -g $ResourceGroup --query "[?kind=='AIServices'].name" -o tsv --only-show-errors | Select-Object -First 1 }
        $FoundryName = if ($existing) { $existing } else { "dl-alk-foundry-" + (Suffix) }
    }
    Step "7. Azure AI Foundry $FoundryName ($FoundryLocation) + project + model deployments + roles"
    $found = if ($DryRun) { $null } else { & $az cognitiveservices account show -n $FoundryName -g $ResourceGroup --query name -o tsv --only-show-errors 2>$null }
    if (-not $found) {
        Az cognitiveservices account create --name $FoundryName --resource-group $ResourceGroup `
            --location $FoundryLocation --kind AIServices --sku S0 --custom-domain $FoundryName --yes `
            --query "properties.endpoint" -o tsv
    }
    # A Foundry *project* is what the Foundry portal (ai.azure.com) works with: playgrounds, agents, evaluations.
    $proj = if ($DryRun) { $null } else { & $az cognitiveservices account project show --name $FoundryName -g $ResourceGroup --project-name dl-alk-class --query name -o tsv --only-show-errors 2>$null }
    if (-not $proj) {
        Az cognitiveservices account project create --name $FoundryName --resource-group $ResourceGroup `
            --project-name dl-alk-class --location $FoundryLocation --display-name "Deep Learning ALK - class demos" `
            --query name -o tsv
    }

    # Model deployments - all pay-per-use (tokens / images / audio), nothing when idle.
    # The Language, Translator, Vision and Speech services need NO deployment - they come with the resource.
    $models = @(
        @{ n = "gpt-4.1-mini";           v = "2025-04-14"; f = "OpenAI";    s = "DataZoneStandard"; c = 50 },  # chat + understands images
        @{ n = "gpt-image-1.5";          v = "2025-12-16"; f = "OpenAI";    s = "DataZoneStandard"; c = 1 },   # image generation
        @{ n = "whisper";                v = "001";        f = "OpenAI";    s = "Standard";         c = 1 },   # speech -> text
        @{ n = "tts";                    v = "001";        f = "OpenAI";    s = "Standard";         c = 1 },   # text -> speech
        @{ n = "text-embedding-3-small"; v = "1";          f = "OpenAI";    s = "DataZoneStandard"; c = 50 },  # embeddings (search, RAG)
        @{ n = "Phi-4-mini-instruct";    v = "1";          f = "Microsoft"; s = "GlobalStandard";   c = 1 }    # small open-weight model
    )
    foreach ($m in $models) {
        $dep = if ($DryRun) { $null } else { & $az cognitiveservices account deployment show -g $ResourceGroup -n $FoundryName --deployment-name $m.n --query name -o tsv --only-show-errors 2>$null }
        if (-not $dep) {
            Az cognitiveservices account deployment create --resource-group $ResourceGroup --name $FoundryName `
                --deployment-name $m.n --model-name $m.n --model-version $m.v --model-format $m.f `
                --sku-name $m.s --sku-capacity $m.c --query name -o tsv
        } else { Write-Host "deployment $($m.n) already exists" }
    }
    if (-not $DryRun) {
        $me = & $az ad signed-in-user show --query id -o tsv --only-show-errors
        $scope = & $az cognitiveservices account show -n $FoundryName -g $ResourceGroup --query id -o tsv --only-show-errors
        foreach ($role in "Cognitive Services OpenAI User", "Cognitive Services User") {
            & $az role assignment create --assignee-object-id $me --assignee-principal-type User --role $role `
                --scope $scope --only-show-errors -o none 2>$null
        }
    }
}

if (-not $SkipExperiments) {
    Step "8. Experiments: MNIST job, sweep, AutoML"
    Az ml job create --file job-mnist-cpu.yml @ws --query name -o tsv
    Az ml job create --file sweep-mnist.yml @ws --query name -o tsv
    $automl = Az ml job create --file automl-housing.yml @ws --query name -o tsv
    Write-Host "AutoML job: $automl"
}

if (-not $SkipEndpoint) {
    Step "9. Best AutoML model as an API (runs and bills 24/7 until deleted!)"
    if (-not $automl -and -not $DryRun) {
        $jobs = & $az ml job list @ws --query "[?display_name=='automl-california-housing'].{name:name, status:status}" -o json --only-show-errors | ConvertFrom-Json
        $automl = $jobs | Where-Object { $_.status -eq "Completed" } | Select-Object -First 1 -ExpandProperty name
    }
    if (-not $DryRun) {
        if (-not $automl) { throw "No AutoML job found - run without -SkipExperiments first." }
        Write-Host "Waiting for AutoML job $automl to finish (up to ~45 min)..."
        do {
            $status = & $az ml job show -n $automl @ws --query status -o tsv --only-show-errors
            Write-Host "  $(Get-Date -Format HH:mm) $status"
            if ($status -notin "Completed", "Failed", "Canceled") { Start-Sleep -Seconds 120 }
        } until ($status -in "Completed", "Failed", "Canceled")
        if ($status -ne "Completed") { throw "AutoML job ended with status $status" }
        $best = & $az ml job show -n $automl @ws --query "tags.automl_best_child_run_id" -o tsv --only-show-errors
    } else { $best = "<best-child-run>" }

    $versions = if ($DryRun) { @() } else { & $az ml model list --name california-housing-automl @ws --query "[].version" -o tsv --only-show-errors 2>$null }
    $version = [string](([int](($versions | ForEach-Object { [int]$_ } | Measure-Object -Maximum).Maximum)) + 1)
    Az ml model create --name california-housing-automl --version $version --type mlflow_model `
        --path "azureml://jobs/$best/outputs/artifacts/outputs/mlflow-model" @ws --query "{name:name, version:version}" -o tsv

    if (-not $EndpointName) { $EndpointName = "california-housing-" + (Suffix) }
    Az ml online-endpoint create --file endpoint-housing.yml --name $EndpointName @ws --query name -o tsv
    Az ml online-deployment create --file deployment-housing.yml --endpoint-name $EndpointName `
        --set "model=azureml:california-housing-automl:$version" "instance_type=$EndpointVmSize" @ws --query name -o tsv
    Az ml online-endpoint update --name $EndpointName --traffic "lightgbm=100" @ws -o none
    Az ml online-endpoint invoke --name $EndpointName --request-file sample-request.json @ws
}

Step "Done"
Write-Host "Studio:  https://ml.azure.com  (workspace $Workspace)"
if (-not $SkipFoundry) { Write-Host "Foundry endpoint for the notebooks:  https://$FoundryName.cognitiveservices.azure.com/" }
if (-not $SkipEndpoint) { Write-Host "API endpoint: $EndpointName  - delete it when you're done: .\teardown-azure.ps1 -ResourceGroup $ResourceGroup -EndpointsOnly" }
