@echo off
setlocal

rem === Project folders ===
set "PROJECT_DIR=C:\Users\Kevin Liu\Desktop\codes"
set "TRANSCR_DIR=C:\Users\Kevin Liu\Desktop\transcriptions"
set "CELL5_DIR=C:\Users\Kevin Liu\Desktop\excel for cell 5, 100 annot"

echo.
echo [1/9] Ensuring folders exist...
mkdir "%PROJECT_DIR%" 2>nul
mkdir "%TRANSCR_DIR%" 2>nul
mkdir "%CELL5_DIR%" 2>nul

echo.
echo [2/9] Choosing Python launcher...
set "PY_CMD=py -3.11"
%PY_CMD% -V >nul 2>&1 || set "PY_CMD=python"
echo Using: %PY_CMD%

echo.
echo [3/9] Creating virtual environment in %PROJECT_DIR%\ .venv ...
pushd "%PROJECT_DIR%"
%PY_CMD% -m venv .venv
if errorlevel 1 (
  echo FAILED to create venv. Make sure Python 3.11 is installed.
  goto :end
)
set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "VENV_PIP=%PROJECT_DIR%\.venv\Scripts\pip.exe"

echo.
echo [4/9] Upgrading pip/setuptools/wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel

echo.
echo [5/9] Installing packages (this can take a few minutes)...
"%VENV_PIP%" install ^
  numpy scipy numba llvmlite ^
  pandas openpyxl ^
  matplotlib seaborn ^
  librosa soundfile audioread ^
  scikit-learn ^
  mido pretty_midi ^
  basic-pitch onnxruntime ^
  tensorflow-cpu crepe ^
  ipykernel music21 lxml ^
  imageio-ffmpeg
rem If you later add an NVIDIA GPU for faster ONNX:
rem "%VENV_PIP%" install onnxruntime-gpu

echo.
echo [6/9] Registering Jupyter kernel "Python 3.11 (codes)"...
"%VENV_PY%" -m ipykernel install --user --name codes-venv --display-name "Python 3.11 (codes)"

echo.
echo [7/9] Verifying key imports and FFmpeg availability...
"%VENV_PY%" - <<PYEND
import sys, shutil
mods = ["pretty_midi","basic_pitch","music21","librosa","pandas","imageio_ffmpeg","lxml"]
failed = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        failed.append((m, str(e)))
if failed:
    print("[Verify] Some imports failed:")
    for m, err in failed:
        print("  -", m, "->", err)
    sys.exit(1)
print("[Verify] All key imports OK.")

# Show lxml version (optional)
import lxml
print("lxml version:", getattr(lxml, "__version__", "unknown"))

# FFmpeg where?
ff_on_path = shutil.which("ffmpeg")
try:
    import imageio_ffmpeg
    ff_img = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    ff_img = None

print("\n[FFmpeg check]")
if ff_on_path:
    print("  - Found on PATH at:", ff_on_path)
else:
    print("  - Not found on PATH.")

if ff_img:
    print("  - imageio-ffmpeg provides:", ff_img)
else:
    print("  - imageio-ffmpeg not available? (should be installed above)")

print("\nTip: In your code, you can set in Cell 1:")
print(r"  FFMPEG_BIN = r''  # or r'C:\ffmpeg\bin\ffmpeg.exe' (leave empty to auto-detect imageio-ffmpeg)")
PYEND
if errorlevel 1 (
  echo One or more Python packages failed to import. See messages above.
  goto :end
)

echo.
echo [8/9] Summary:
echo   Project:       %PROJECT_DIR%
echo   Transcripts:   %TRANSCR_DIR%
echo   Cell5 Excel:   %CELL5_DIR%
echo   Venv Python:   %VENV_PY%

where code >nul 2>&1
if %errorlevel%==0 (
  echo Launching VS Code in the project folder...
  code "%PROJECT_DIR%"
) else (
  echo VS Code CLI not found. Open this folder in VS Code manually: "%PROJECT_DIR%"
)

:end
popd
echo.
echo ✅ Environment ready. In VS Code: pick kernel "Python 3.11 (codes)" and run your cells.
echo If a terminal opens, activate venv with:  .\.venv\Scripts\activate
echo.
pause
