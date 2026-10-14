$ErrorActionPreference = "Stop"

# Run from E:\arcface-eeg-reference\recognition\arcface_torch.
# This script intentionally stages source/docs only. It does NOT stage datasets,
# manifests, checkpoints, logs, cache files, secrets, or model weights.

$repoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) { throw "Current directory is not inside a Git repository." }

# The clone uses sparse checkout; the working code lives under this subdirectory.
$workDir = Join-Path $repoRoot "recognition\arcface_torch"
if (-not (Test-Path (Join-Path $workDir "eeg_model.py"))) {
    throw "Expected EEG code under $workDir. Copy the adaptation files there first."
}

$branch = (git branch --show-current).Trim()
if ($branch -ne "eeg-adaptation") {
    Write-Host "Current branch is '$branch', not 'eeg-adaptation'." -ForegroundColor Yellow
    $confirm = Read-Host "Continue and push this current branch? Type YES to continue"
    if ($confirm -ne "YES") { throw "Stopped to avoid pushing the wrong branch." }
}

gh --version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "GitHub CLI (gh) is missing. Install it from https://cli.github.com/ and run gh auth login." }
gh auth status
if ($LASTEXITCODE -ne 0) { throw "GitHub CLI is not authenticated. Run gh auth login first." }

$files = @(
    "eeg_dataset.py",
    "eeg_model.py",
    "train_eeg.py",
    "smoke_test_eeg.py",
    "evaluate_eeg_arcface.py",
    "api.py",
    "static/index.html",
    "README_EEG.md",
    "EEG_ARCFACE_README.md"
)

$missing = @($files | Where-Object { -not (Test-Path (Join-Path $workDir $_)) })
if ($missing.Count -gt 0) {
    Write-Host "These files were not found under ${workDir}:" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "  $_" }
    Write-Host "Copy the adaptation/demo files into the current repository first, or edit this script's file list."
    throw "Stopped before staging."
}

# Show exactly what will be staged before committing.
git status --short
$repoRelativeFiles = @($files | ForEach-Object { "recognition/arcface_torch/" + ($_ -replace "\\", "/") })
git add -- $repoRelativeFiles
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw "Whitespace check failed; fix the staged diff." }

Write-Host ""
Write-Host "Review staged changes above. This script will now create a commit and push to datta256/insightface."
$confirm = Read-Host "Type PUSH to commit and push"
if ($confirm -ne "PUSH") { throw "Stopped before commit/push. Staged changes remain staged." }

git commit -m "Add EEG ArcFace biometric research adaptation"
if ($LASTEXITCODE -ne 0) { throw "Commit failed." }

# Create the fork if it does not already exist. gh uses the authenticated account.
$forkUrl = "https://github.com/datta256/insightface.git"
gh repo view datta256/insightface 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    gh repo fork deepinsight/insightface --clone=false
    if ($LASTEXITCODE -ne 0) { throw "Fork creation failed." }
}

$remotes = git remote
if ($remotes -contains "datta256") {
    git remote set-url datta256 $forkUrl
} else {
    git remote add datta256 $forkUrl
}

git push -u datta256 $branch
if ($LASTEXITCODE -ne 0) { throw "Push failed. Check branch protection and repository permissions." }

Write-Host ""
Write-Host "Pushed branch '$branch' to $forkUrl" -ForegroundColor Green
Write-Host "Open https://github.com/datta256/insightface/tree/$branch"

