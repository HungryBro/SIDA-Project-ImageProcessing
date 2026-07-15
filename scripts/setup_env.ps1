# PowerShell script to set up the python environment for YOLO and Grad-CAM

$envName = "yolo_gradcam"
$projectRoot = Split-Path $PSScriptRoot -Parent
$venvDir = Join-Path $projectRoot ".venv"
$requirementsFile = Join-Path $projectRoot "requirements.txt"

# 1. Search for Conda
$condaPaths = @(
    "conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\miniconda3\Scripts\conda.exe"
)

$condaPath = $null
foreach ($path in $condaPaths) {
    $resolved = Get-Command $path -ErrorAction SilentlyContinue
    if ($resolved) {
        $condaPath = $resolved.Source
        break
    }
}

# 2. Search for Python
$pythonPaths = @(
    "python.exe",
    "$env:USERPROFILE\AppData\Local\Programs\Python\Python311\python.exe",
    "$env:USERPROFILE\AppData\Local\Programs\Python\Python310\python.exe",
    "C:\Program Files\Python311\python.exe",
    "C:\Program Files\Python310\python.exe"
)

$pythonPath = $null
foreach ($path in $pythonPaths) {
    $resolved = Get-Command $path -ErrorAction SilentlyContinue
    if ($resolved) {
        $pythonPath = $resolved.Source
        break
    }
}

if ($condaPath) {
    Write-Host "[+] Found Conda at: $condaPath" -ForegroundColor Green
    Write-Host "[*] Creating Conda environment '$envName' with Python 3.11..." -ForegroundColor Cyan
    
    # Run conda create
    & $condaPath create -n $envName python=3.11 -y
    
    # Get the python executable path in the new environment
    # Typically: C:\Users\Acer\miniconda3\envs\yolo_gradcam\python.exe
    $condaBase = Split-Path (Split-Path $condaPath -Parent) -Parent
    $envPython = Join-Path $condaBase "envs\$envName\python.exe"
    
    if (-not (Test-Path $envPython)) {
        # Fallback to check under AppData or user profile envs
        $envPython = Join-Path $env:USERPROFILE ".conda\envs\$envName\python.exe"
    }

    if (Test-Path $envPython) {
        Write-Host "[+] Environment python found at: $envPython" -ForegroundColor Green
        Write-Host "[*] Installing dependencies..." -ForegroundColor Cyan
        & $envPython -m pip install --upgrade pip
        & $envPython -m pip install -r $requirementsFile
        Write-Host "[+] Environment setup successfully using Conda!" -ForegroundColor Green
        Write-Host "[*] To run your script, use: conda activate $envName; python src\gradcam_yolo.py" -ForegroundColor Yellow
    } else {
        Write-Error "Conda environment created but python.exe not found at $envPython"
    }
}
elseif ($pythonPath) {
    Write-Host "[+] Found Python at: $pythonPath" -ForegroundColor Green
    Write-Host "[*] Creating virtual environment (.venv)..." -ForegroundColor Cyan
    
    & $pythonPath -m venv $venvDir
    $venvPython = Join-Path $venvDir "Scripts\python.exe"
    
    if (Test-Path $venvPython) {
        Write-Host "[*] Upgrading pip and installing requirements..." -ForegroundColor Cyan
        & $venvPython -m pip install --upgrade pip
        & $venvPython -m pip install -r $requirementsFile
        Write-Host "[+] Environment setup successfully using virtualenv!" -ForegroundColor Green
        Write-Host "[*] To run your script, activate with: .venv\Scripts\Activate.ps1; python src\gradcam_yolo.py" -ForegroundColor Yellow
    } else {
        Write-Error "Virtual environment created but python.exe not found at $venvPython"
    }
}
else {
    Write-Error "Neither Python nor Conda could be found. Please install Python 3.10/3.11 or Miniconda first."
}
