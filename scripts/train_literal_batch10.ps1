# Literal-batch-10 baseline on Windows (8 seeds).
# Sets the CUDA allocator option needed to fit a batch of 10 on a 4 GB GPU and logs to
# results\logs\train_literal_batch10.log. Optional: -WaitPid <pid> waits for another
# process (for example scripts\run_experiments.py) to free the GPU first.
param([int]$WaitPid = 0)
$repo = Split-Path $PSScriptRoot -Parent
$logDir = Join-Path $repo "results\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "train_literal_batch10.log"
if ($WaitPid) { Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue }
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:64,garbage_collection_threshold:0.8"
"=== START $(Get-Date -Format s) ===" | Out-File $log -Append -Encoding utf8
python (Join-Path $PSScriptRoot "train_literal_batch10.py") *>> $log
"=== END $(Get-Date -Format s) ===" | Out-File $log -Append -Encoding utf8
