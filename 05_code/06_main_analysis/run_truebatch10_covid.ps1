# Literal-batch-10 O1 arm (paper §5.2, 8 seeds) on Zenodo COVID-19-CT-Seg.
# Waits for the main chain (PID passed as arg) to free the GPU, then runs with
# max_split_size_mb to avoid the allocator fragmentation that caused the first OOM
# (expandable_segments, tried first, is not supported on Windows). Numerics unchanged.
param([int]$WaitPid = 0)
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "..\..\07_analysis_outputs\covid19_ct_seg\logs\run_truebatch10_covid.log"
if ($WaitPid) { Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue }
$env:LUG_CONFIG = "config_covid.yaml"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:64,garbage_collection_threshold:0.8"
"=== START $(Get-Date -Format s) ===" | Out-File $log -Append -Encoding utf8
python train_faithful.py --runs 8 --out faithful_truebatch10_all.csv *>> $log
"=== END $(Get-Date -Format s) ===" | Out-File $log -Append -Encoding utf8
