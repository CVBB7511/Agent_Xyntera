param([ValidateSet('sync','bootstrap','prepare','preflight','smoke','campaign','run','grade','report','fetch')][string]$Action='sync')
$ErrorActionPreference='Stop'
$root='/mnt/sata1/beihang-cuda_backup'
$target='beihang@10.132.80.126'
$local=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
function Checked { if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE)" } }
if ($Action -eq 'sync') {
    ssh -o BatchMode=yes $target "cd $root && mountpoint -q /mnt/sata1 && mkdir -p code/memtrace_pilot vendor/MemTrace results/memtrace-pilot"
    Checked
    scp -q -r "$PSScriptRoot/." "${target}:$root/code/memtrace_pilot/"
    Checked
    $vendor=Join-Path $local 'paper/code_idx/25_memTrace/MemTrace-main'
    scp -q -r "$vendor/src" "$vendor/configs" "$vendor/pyproject.toml" "$vendor/README.md" "$vendor/LICENSE" "$vendor/NOTICE" "${target}:$root/vendor/MemTrace/"
    Checked
} elseif ($Action -eq 'fetch') {
    $tag=[guid]::NewGuid().ToString('N')
    $remote="$root/.conda/tmp/pilot-results-$tag.tar.gz"
    $archive=Join-Path ([System.IO.Path]::GetTempPath()) "pilot-results-$tag.tar.gz"
    try {
        ssh -o BatchMode=yes $target "cd $root && umask 077 && mountpoint -q /mnt/sata1 && tar -czf $remote -C results memtrace-pilot"
        Checked
        $expected=ssh -o BatchMode=yes $target "sha256sum $remote"
        Checked
        scp -q "${target}:$remote" $archive
        Checked
        if((Get-FileHash -Algorithm SHA256 $archive).Hash -ne ($expected -split ' ')[0]) { throw 'Result archive checksum mismatch' }
        tar -xzf $archive -C "$local/results"
        Checked
    } finally {
        ssh -o BatchMode=yes $target "rm -f -- $remote"
        if(Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive }
    }
} elseif ($Action -eq 'bootstrap') {
    ssh -o BatchMode=yes $target "cd $root && bash -o pipefail -c 'bash code/memtrace_pilot/bootstrap.sh 2>&1 | tee results/memtrace-pilot/bootstrap.log'"
    Checked
} elseif ($Action -eq 'campaign') {
    ssh -o BatchMode=yes $target "cd $root && bash -o pipefail -c 'bash code/memtrace_pilot/campaign.sh 2>&1 | tee -a results/memtrace-pilot/campaign.log'"
    Checked
} else {
    ssh -o BatchMode=yes $target "cd $root && .conda/v00/bin/python code/memtrace_pilot/pilot.py $Action"
    Checked
}
