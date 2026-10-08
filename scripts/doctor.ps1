<#Requires -Version 5.1>
<#
.SYNOPSIS
  LocalFind environment doctor (P0-T002). Reports toolchain versions on Windows.

.DESCRIPTION
  Read-only check: verifies the Windows 11 x64 target and the presence of the
  Phase 0 toolchain (Python, Rust, Node, package manager, FFmpeg, git) and
  reports GPU/CUDA capability when present. Missing REQUIRED tools produce
  actionable install guidance and a non-zero exit code.

  Privacy: this script intentionally never prints usernames, computer names,
  home-directory paths, or full file paths — only tool names, version strings,
  and hardware model identifiers. Do not add $env:USERNAME, $env:COMPUTERNAME,
  $env:USERPROFILE, $PROFILE, or `pip --version` raw output (it embeds paths).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/doctor.ps1
  Exit code 0 = all required checks pass; 1 = at least one required check failed.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$script:Failures = 0

function Write-Check {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][ValidateSet('OK', 'MISSING', 'INFO', 'WARN')][string]$Status,
        [Parameter(Mandatory)][string]$Detail,
        [string]$Hint = ''
    )
    $line = "[$Status] $Name : $Detail"
    Write-Output $line
    if ($Hint -ne '') { Write-Output "         -> $Hint" }
    if ($Status -eq 'MISSING') { $script:Failures += 1 }
}

# Runs "<command> <args>" and returns the first line of combined output,
# or $null when the command is not on PATH or exits non-zero.
function Get-FirstLine {
    param([Parameter(Mandatory)][string]$Command, [string]$Arguments = '--version')
    $cmd = Get-Command $Command -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { return $null }
    try {
        $out = & $cmd @($Arguments.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)) 2>&1 |
            Select-Object -First 1
        if ($LASTEXITCODE -ne 0) { return $null }
        return "$out".Trim()
    } catch {
        return $null
    }
}

Write-Output 'LocalFind doctor (P0-T002) — Windows toolchain check'
Write-Output '-----------------------------------------------------'

# --- OS: require Windows 64-bit, require Windows 11 build (>= 22000) ---
try {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
    $caption = "$($os.Caption)".Trim()
    $build = [int]$os.BuildNumber
    $arch = "$($os.OSArchitecture)".Trim()
    if ($arch -like '*64-bit*' -and $build -ge 22000) {
        Write-Check -Name 'OS' -Status 'OK' -Detail "$caption (build $build, $arch)"
    } else {
        Write-Check -Name 'OS' -Status 'MISSING' -Detail "$caption (build $build, $arch)" `
            -Hint 'LocalFind targets Windows 11 x64 (build >= 22000). Use a Windows 11 x64 machine.'
    }
} catch {
    Write-Check -Name 'OS' -Status 'MISSING' -Detail 'Could not query Win32_OperatingSystem.' `
        -Hint 'Run on Windows with WMI/CIM available.'
}

# --- CPU / RAM (informational; model strings only, no machine identity) ---
try {
    $cpu = Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1
    $cs = Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
    $ramGB = [math]::Round([double]$cs.TotalPhysicalMemory / 1GB, 1)
    Write-Check -Name 'CPU' -Status 'INFO' `
        -Detail "$($cpu.Name.Trim()) ($($cpu.NumberOfCores) cores / $($cpu.NumberOfLogicalProcessors) threads)"
    if ($ramGB -ge 8) {
        Write-Check -Name 'RAM' -Status 'INFO' -Detail "$ramGB GB"
    } else {
        Write-Check -Name 'RAM' -Status 'WARN' -Detail "$ramGB GB (below 8 GB; Phase 0 model runs may OOM)"
    }
} catch {
    Write-Check -Name 'CPU/RAM' -Status 'WARN' -Detail 'Could not query processor/memory via CIM.'
}

# --- GPU / CUDA capability (informational; safe query fields only) ---
$gpu = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -ne $gpu) {
    try {
        $rows = & $gpu --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $rows) {
            foreach ($row in @($rows)) {
                Write-Check -Name 'GPU' -Status 'INFO' -Detail "nvidia-smi: $("$row".Trim())"
            }
        } else {
            Write-Check -Name 'GPU' -Status 'INFO' -Detail 'nvidia-smi present but GPU query failed (no NVIDIA GPU or driver issue).'
        }
    } catch {
        Write-Check -Name 'GPU' -Status 'INFO' -Detail 'nvidia-smi present but GPU query failed.'
    }
} else {
    Write-Check -Name 'GPU' -Status 'INFO' -Detail 'No NVIDIA GPU detected (nvidia-smi absent). CPU-only runs expected.'
}

# --- Python (required >= 3.10 for Phase 0 model experiments) ---
$pyLine = Get-FirstLine -Command 'python' -Arguments '--version'
if ($null -eq $pyLine) { $pyLine = Get-FirstLine -Command 'py' -Arguments '-3 --version' }
if ($null -ne $pyLine -and $pyLine -match 'Python\s+(\d+)\.(\d+)\.(\d+)') {
    $pyVer = [version]"$($Matches[1]).$($Matches[2]).$($Matches[3])"
    if ($pyVer -ge [version]'3.10.0') {
        Write-Check -Name 'Python' -Status 'OK' -Detail $pyLine
    } else {
        Write-Check -Name 'Python' -Status 'MISSING' -Detail "$pyLine (requires >= 3.10)" `
            -Hint 'Install Python >= 3.10: winget install Python.Python.3.12  (then re-run doctor).'
    }
} else {
    Write-Check -Name 'Python' -Status 'MISSING' -Detail 'python/py not found on PATH.' `
        -Hint 'Install Python >= 3.10: winget install Python.Python.3.12  (then re-run doctor).'
}

# --- Rust toolchain (required from P1; rustup is the supported path) ---
$rustcLine = Get-FirstLine -Command 'rustc'
$cargoLine = Get-FirstLine -Command 'cargo'
if ($null -ne $rustcLine -and $null -ne $cargoLine) {
    Write-Check -Name 'Rust' -Status 'OK' -Detail "$rustcLine / $cargoLine"
} else {
    Write-Check -Name 'Rust' -Status 'MISSING' -Detail 'rustc/cargo not found on PATH.' `
        -Hint 'Install via rustup: winget install Rustlang.Rustup  then `rustup install stable`. Tauri builds also need MSVC Build Tools (VS 2022 C++ workload).'
}

# --- Node.js (required >= 20 for the Tauri/Svelte frontend) ---
$nodeLine = Get-FirstLine -Command 'node' -Arguments '--version'
if ($null -ne $nodeLine -and $nodeLine -match 'v(\d+)\.') {
    if ([int]$Matches[1] -ge 20) {
        Write-Check -Name 'Node' -Status 'OK' -Detail $nodeLine
    } else {
        Write-Check -Name 'Node' -Status 'MISSING' -Detail "$nodeLine (requires >= v20)" `
            -Hint 'Install Node 22 LTS: winget install OpenJS.NodeJS.LTS  (then re-run doctor).'
    }
} else {
    Write-Check -Name 'Node' -Status 'MISSING' -Detail 'node not found on PATH.' `
        -Hint 'Install Node 22 LTS: winget install OpenJS.NodeJS.LTS  (then re-run doctor).'
}

# --- Package manager: pnpm preferred (PLAN scripts use pnpm), npm accepted ---
$pnpmLine = Get-FirstLine -Command 'pnpm' -Arguments '--version'
$npmLine = Get-FirstLine -Command 'npm' -Arguments '--version'
if ($null -ne $pnpmLine) {
    Write-Check -Name 'pnpm' -Status 'OK' -Detail "pnpm $pnpmLine"
} elseif ($null -ne $npmLine) {
    Write-Check -Name 'PackageMgr' -Status 'OK' -Detail "npm $npmLine (pnpm preferred; run: npm install -g pnpm)"
} else {
    Write-Check -Name 'PackageMgr' -Status 'MISSING' -Detail 'Neither pnpm nor npm found on PATH.' `
        -Hint 'Install Node 22 LTS (bundles npm), then `npm install -g pnpm`. winget install OpenJS.NodeJS.LTS'
}

# --- FFmpeg + ffprobe (required for P0-T005 video probing/segments) ---
$ffLine = Get-FirstLine -Command 'ffmpeg' -Arguments '-version'
$fpLine = Get-FirstLine -Command 'ffprobe' -Arguments '-version'
if ($null -ne $ffLine) {
    Write-Check -Name 'ffmpeg' -Status 'OK' -Detail "$("$ffLine".Split([Environment]::NewLine)[0])"
} else {
    Write-Check -Name 'ffmpeg' -Status 'MISSING' -Detail 'ffmpeg not found on PATH.' `
        -Hint 'Install FFmpeg: winget install Gyan.FFmpeg  (then re-run doctor).'
}
if ($null -ne $fpLine) {
    Write-Check -Name 'ffprobe' -Status 'OK' -Detail "$("$fpLine".Split([Environment]::NewLine)[0])"
} else {
    Write-Check -Name 'ffprobe' -Status 'MISSING' -Detail 'ffprobe not found on PATH.' `
        -Hint 'Install FFmpeg (bundles ffprobe): winget install Gyan.FFmpeg  (then re-run doctor).'
}

# --- git (required; repo already initialized in P0-T001) ---
$gitLine = Get-FirstLine -Command 'git' -Arguments '--version'
if ($null -ne $gitLine) {
    Write-Check -Name 'git' -Status 'OK' -Detail $gitLine
} else {
    Write-Check -Name 'git' -Status 'MISSING' -Detail 'git not found on PATH.' `
        -Hint 'Install Git: winget install Git.Git  (then re-run doctor).'
}

Write-Output '-----------------------------------------------------'
if ($script:Failures -eq 0) {
    Write-Output 'doctor: all required checks passed.'
    exit 0
} else {
    Write-Output ("doctor: {0} required check(s) failed — follow the '->' hints above, then re-run." -f $script:Failures)
    exit 1
}
