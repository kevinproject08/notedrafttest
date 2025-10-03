## Music to MIDI transcription using ML  (title)


# %% Cells
# # Music Transcription Pipeline – MIDI-only (Simplified, no feature changes)
# - Cell 1: Imports & Settings
# - Cell 2: Utilities (picker, ffmpeg, run dirs)
# - Cell 3: Quantizer & Hand Split
# - Cell 4: MIDI Writers
# - Cell 5: Transcribe 
# - Cell 6: Processing
# - Cell 7: Run everything

## Music to MIDI transcription using ML
## Music to MIDI transcription using ML with MusicXML support

# %% Cell 1 - Imports & Settings
from __future__ import annotations
import re, os, sys, json, shutil, subprocess, inspect, itertools
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import pretty_midi as pm
import mido

# Import notation module if available
try:
    import notation_export
    NOTATION_AVAILABLE = True
except ImportError:
    NOTATION_AVAILABLE = False
    print("[WARNING] notation_export not available - MusicXML export will be disabled")

# Optional: librosa for duration detection
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[WARNING] librosa not available - duration detection will be limited")

# ---- Configuration Management ----
CONFIG_FILE = Path.home() / ".music_transcription_config.json"

def _load_config() -> dict:
    """Load configuration from file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                if "transcription_root" in config:
                    if Path(config["transcription_root"]).exists():
                        return config
                    else:
                        print(f"[Config] Previous location no longer exists: {config['transcription_root']}")
                        print("[Config] Will prompt for new location...")
                        return {}
        except Exception as e:
            print(f"[Config] Error loading config: {e}")
    return {}

def _save_config(config: dict):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[Config] Settings saved")
    except Exception as e:
        print(f"[Config] Error saving config: {e}")

def _choose_transcription_location() -> Optional[Path]:
    """Let user choose where to create the transcriptions folder"""
    
    print("\n" + "="*70)
    print("SELECT OUTPUT LOCATION FOR TRANSCRIPTIONS")
    print("="*70)
    print("Please choose where to create the 'transcriptions' folder.")
    print("This location will be saved for all future transcriptions.")
    print("="*70 + "\n")
    
    # Try GUI first
    try:
        from tkinter import Tk, filedialog, messagebox
        
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        
        messagebox.showinfo(
            "Choose Location for Transcriptions",
            "Please select the folder where you want to create a 'transcriptions' directory.",
            parent=root
        )
        
        chosen_dir = filedialog.askdirectory(
            title="Select where to create 'transcriptions' folder",
            initialdir=str(Path.home()),
            parent=root
        )
        
        root.destroy()
        
        if chosen_dir:
            transcr_path = Path(chosen_dir) / "transcriptions"
            print(f"[Config] Selected location: {transcr_path}")
            return transcr_path
        else:
            print("[Config] No location selected.")
            return None
            
    except ImportError:
        pass
    
    # Fallback to CLI if tkinter is not available
    print("[Config] GUI not available, using command line input.")
    print("\nEnter the full path where you want to create the 'transcriptions' folder")
    user_input = input("\nPath: ").strip().strip('"').strip("'")
        
    if not user_input:
        print("[Config] No path entered. Cancelling...")
        return None
        
    try:
        parent_path = Path(user_input)
        if parent_path.exists() and parent_path.is_dir():
            return parent_path / "transcriptions"
        else:
            print("[Error] Path does not exist or is not a directory.")
            return None
    except Exception as e:
        print(f"[Error] Invalid path: {e}")
        return None

def _get_transcription_root(force_reconfigure: bool = False) -> Optional[Path]:
    """Get or set the transcription root directory"""
    config = _load_config() if not force_reconfigure else {}
    
    if force_reconfigure or "transcription_root" not in config:
        transcr_path = _choose_transcription_location()
        
        if transcr_path is None:
            return None
        
        config["transcription_root"] = str(transcr_path)
        _save_config(config)
        
        transcr_path.mkdir(parents=True, exist_ok=True)
        return transcr_path
    
    transcr_path = Path(config["transcription_root"])
    transcr_path.mkdir(parents=True, exist_ok=True)
    return transcr_path


# ---- Global Settings ----
TRANSCR_ROOT = None
PREFIX = "transcription"
FFMPEG_BIN = ""

USE_INTERACTIVE_TIME_SELECTION = True
DEFAULT_TIME_RANGE_START = 0.0
DEFAULT_TIME_RANGE_END = 0.0

USE_FILE_PICKER = True
FILE_PICKER_TOPMOST = True
FALLBACK_CLI_PROMPT = True

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".wma",
              ".aif", ".aiff", ".aifc", ".opus"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
MIDI_EXTS  = {".mid", ".midi"}
ACCEPTED_EXTS = AUDIO_EXTS | VIDEO_EXTS | MIDI_EXTS

ALL_FILE_FILTER = [
    ("Media & MIDI", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.mp4 *.mov *.mkv *.avi *.mid *.midi"),
    ("Audio", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg"),
    ("Video", "*.mp4 *.mov *.mkv *.avi"),
    ("MIDI", "*.mid *.midi"),
    ("All files", "*.*"),
]

TARGET_SR = 44100
FORCE_BPM = None
BPM_MIN, BPM_MAX = 50, 200
DEFAULT_BPM = 120
QUANTIZE_STRENGTH = 0.6
MIN_NOTE_DURATION_SEC = 0.06
CHORD_TOLERANCE_MS = 50
TREBLE_BASS_SPLIT = 60
BP_ONSET_THRESH = 0.4
BP_FRAME_THRESH = 0.3
BP_MIN_NOTE_LEN = 0.05
AUTO_CUT_TIME = True
CUTTIME_BPM_THRESHOLD = 160


# %% Cell 2 - Utilities

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _ffmpeg_path() -> str:
    if FFMPEG_BIN:
        cand = Path(FFMPEG_BIN.strip().strip('"'))
        if cand.is_file():
            return str(cand)
    for name in ("ffmpeg", "ffmpeg.exe"):
        cand = shutil.which(name)
        if cand and Path(cand).is_file():
            return cand
    try:
        import imageio_ffmpeg
        cand = imageio_ffmpeg.get_ffmpeg_exe()
        if cand and Path(cand).is_file():
            return cand
    except Exception:
        pass
    raise FileNotFoundError("FFmpeg not found.")

def _get_audio_duration(file_path: Path) -> float:
    try:
        if LIBROSA_AVAILABLE and file_path.suffix.lower() in AUDIO_EXTS:
            return float(librosa.get_duration(path=str(file_path)))
    except Exception:
        pass
    try:
        ffmpeg = _ffmpeg_path()
        result = subprocess.run([ffmpeg, "-i", str(file_path), "-f", "null", "-"],
                                capture_output=True, text=True, stderr=subprocess.STDOUT)
        text = result.stdout or result.stderr or ""
        for line in text.splitlines():
            if "Duration:" in line:
                d = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = d.split(":")
                return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        pass
    return 0.0

def _get_time_range_from_user(duration: float, filename: str = "") -> Tuple[float, float]:
    """Get time range using sliders"""
    if duration <= 0:
        print("[Duration] Unknown; using full file")
        return 0.0, 0.0
    if not USE_INTERACTIVE_TIME_SELECTION:
        s = DEFAULT_TIME_RANGE_START
        e = DEFAULT_TIME_RANGE_END or duration
        return max(0.0, s), min(duration, e)

    print(f"\n[Duration] {filename}: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    
    try:
        from tkinter import Tk, Scale, Button, Label, Frame, HORIZONTAL
        root = Tk()
        root.title(f"Select Time Range - {filename}")
        root.geometry("600x300")
        root.attributes("-topmost", True)
        
        result = {'start': 0.0, 'end': duration}
        
        Label(root, text=f"Select time range for: {filename}", font=("Arial", 12, "bold")).pack(pady=10)
        Label(root, text=f"Total duration: {duration:.1f} seconds").pack()
        
        slider_frame = Frame(root)
        slider_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        Label(slider_frame, text="Start Time:").grid(row=0, column=0, sticky="w", pady=5)
        start_value_label = Label(slider_frame, text="0.0 s", width=10)
        start_value_label.grid(row=0, column=2, pady=5)
        start_slider = Scale(slider_frame, from_=0, to=duration, orient=HORIZONTAL, 
                           resolution=0.1, length=400)
        start_slider.set(0)
        start_slider.grid(row=0, column=1, pady=5)
        
        Label(slider_frame, text="End Time:").grid(row=1, column=0, sticky="w", pady=5)
        end_value_label = Label(slider_frame, text=f"{duration:.1f} s", width=10)
        end_value_label.grid(row=1, column=2, pady=5)
        end_slider = Scale(slider_frame, from_=0, to=duration, orient=HORIZONTAL,
                         resolution=0.1, length=400)
        end_slider.set(duration)
        end_slider.grid(row=1, column=1, pady=5)
        
        range_label = Label(slider_frame, text="", font=("Arial", 10), fg="blue")
        range_label.grid(row=2, column=0, columnspan=3, pady=10)
        
        def update_range_display():
            s = start_slider.get()
            e = end_slider.get()
            if e <= s:
                e = min(s + 1, duration)
                end_slider.set(e)
            start_value_label.config(text=f"{s:.1f} s")
            end_value_label.config(text=f"{e:.1f} s")
            range_label.config(text=f"Selected: {s:.1f}s to {e:.1f}s (Duration: {e-s:.1f}s)")
        
        start_slider.config(command=lambda v: update_range_display())
        end_slider.config(command=lambda v: update_range_display())
        
        button_frame = Frame(root)
        button_frame.pack(pady=10)
        
        def use_selection():
            result['start'] = start_slider.get()
            result['end'] = end_slider.get()
            root.quit()
        
        def use_full():
            result['start'] = 0
            result['end'] = duration
            root.quit()
        
        Button(button_frame, text="Use Selected Range", command=use_selection, 
               bg="#4CAF50", fg="white", padx=20, pady=5).pack(side="left", padx=5)
        Button(button_frame, text="Use Full File", command=use_full,
               bg="#2196F3", fg="white", padx=20, pady=5).pack(side="left", padx=5)
        
        update_range_display()
        root.mainloop()
        root.destroy()
        
        print(f"[Range] Selected: {result['start']:.1f}s to {result['end']:.1f}s")
        return result['start'], result['end']
        
    except Exception as e:
        print(f"[GUI] Error, falling back to CLI: {e}")
        print("[Time Range] Enter to use full file, or specify start/end seconds.")
        s_in = input(f"Start (0 to {duration:.1f} sec): ").strip()
        e_in = input(f"End ({s_in or '0'} to {duration:.1f} sec): ").strip()
        s = float(s_in) if s_in else 0.0
        e = float(e_in) if e_in else duration
        return max(0.0, s), min(duration, e)

def _choose_files() -> List[Tuple[str, float, float]]:
    """Let user select files to process"""
    out: List[Tuple[str, float, float]] = []
    
    while True:
        path = None
        if USE_FILE_PICKER:
            try:
                from tkinter import Tk, filedialog
                root = Tk()
                root.withdraw()
                root.attributes("-topmost", FILE_PICKER_TOPMOST)
                
                file_num = len(out) + 1
                selected_path = filedialog.askopenfilename(
                    title=f"Select audio/video/MIDI file #{file_num}",
                    filetypes=ALL_FILE_FILTER
                )
                root.destroy()
                
                if selected_path:
                    path = selected_path
                else:
                    if len(out) == 0: return []
                    else: break
            except Exception:
                pass
        
        if not path:
            file_num = len(out) + 1
            raw = input(f"File #{file_num} path (Enter to finish): ").strip().strip("\"' ")
            if raw: path = raw
            else:
                if len(out) == 0: return []
                else: break

        if path:
            ext = Path(path).suffix.lower()
            if ext not in ACCEPTED_EXTS:
                print(f"[Error] Unsupported file type: {ext}")
                continue
            if ext in MIDI_EXTS:
                out.append((path, 0.0, 0.0))
                print(f"[Added] MIDI: {Path(path).name}")
            else:
                dur = _get_audio_duration(Path(path))
                s, e = _get_time_range_from_user(dur, Path(path).name) if dur > 0 else (0.0, 0.0)
                out.append((path, s, e))
                print(f"[Added] Media: {Path(path).name}")
            
            print(f"\n{len(out)} file(s) selected.")
            more = input("Add another file? (yes/no): ").strip().lower()
            if more not in ['yes', 'y']:
                return out
    
    return out

def _make_run_dir() -> Path:
    global TRANSCR_ROOT
    if TRANSCR_ROOT is None:
        raise RuntimeError("Root directory not initialized")
    
    nums = [int(m.group(1)) for p in TRANSCR_ROOT.glob(f"{PREFIX}*")
            if p.is_dir() and (m := re.search(rf"{re.escape(PREFIX)}(\d+)$", p.name))]
    idx = (max(nums) + 1) if nums else 1
    run_dir = TRANSCR_ROOT / f"{PREFIX}{idx}"
    _ensure_dir(run_dir)
    return run_dir

def _to_wav_via_ffmpeg(src: Path, dst_wav: Path, s: float = 0.0, e: float = 0.0) -> bool:
    ffm = _ffmpeg_path()
    cmd = [ffm, "-y"]
    if s > 0: cmd += ["-ss", str(s)]
    if e > s: cmd += ["-t", str(e - s)]
    cmd += ["-i", str(src), "-vn", "-ar", str(TARGET_SR), "-ac", "1", str(dst_wav)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0
    except Exception:
        return False

def _count_midis(d: Path) -> int:
    return sum(1 for _ in itertools.chain(d.glob("*.mid"), d.glob("*.midi")))

def _list_midis(d: Path) -> List[Path]:
    return sorted(itertools.chain(d.glob("*.mid"), d.glob("*.midi")), key=lambda p: p.name)


# %% Cell 3 - Quantization & BPM Detection

def _improved_bpm_detection(notes_array: np.ndarray) -> float:
    """Superior beat detection algorithm"""
    if FORCE_BPM:
        return float(FORCE_BPM)
    if len(notes_array) < 10:
        return DEFAULT_BPM

    onsets = np.sort(notes_array[:, 0])
    iois = np.diff(onsets)
    valid = iois[(iois >= 0.2) & (iois <= 2.0)]
    if len(valid) == 0:
        return DEFAULT_BPM

    hist, bins = np.histogram(valid, bins=50)
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append((bins[i] + bins[i+1]) / 2)

    candidates = []
    for peak_ioi in peaks[:3]:
        for division in [0.25, 0.5, 1, 2, 4]:
            bpm = (60.0 / peak_ioi) * division
            if BPM_MIN <= bpm <= BPM_MAX:
                beat = 60.0 / bpm
                score = 0.0
                for onset in onsets[:20]:
                    pos = (onset % beat) / beat
                    score += min(abs(pos - p) for p in (0, 0.25, 0.5, 0.75, 1.0))
                candidates.append((bpm, score))
    if candidates:
        best = min(candidates, key=lambda x: x[1])[0]
        print(f"[BPM] Detected: {best:.0f}")
        return best
    print(f"[BPM] Defaulting to {DEFAULT_BPM}")
    return DEFAULT_BPM

def _reduce_chord_clusters(notes_array: np.ndarray, max_notes_per_chord: int = 6) -> np.ndarray:
    if len(notes_array) == 0:
        return notes_array
    notes = notes_array[np.argsort(notes_array[:, 0])].copy()
    out = []
    i = 0
    while i < len(notes):
        start = notes[i, 0]
        idxs = [i]
        j = i + 1
        while j < len(notes) and (notes[j, 0] - start) * 1000 < CHORD_TOLERANCE_MS:
            idxs.append(j)
            j += 1
        cluster = notes[idxs]
        if len(cluster) > max_notes_per_chord:
            treble = cluster[cluster[:, 2] >= TREBLE_BASS_SPLIT]
            bass = cluster[cluster[:, 2] < TREBLE_BASS_SPLIT]
            keep = []
            if len(treble):
                s = treble[np.argsort(treble[:, 2])]
                keep.extend([s[0], s[-1]] if len(s) >= 2 else [s[-1]])
                if len(s) >= 3:
                    k = np.argmax(s[:, 3])
                    if k not in (0, len(s)-1): keep.append(s[k])
            if len(bass):
                s = bass[np.argsort(bass[:, 2])]
                keep.extend([s[0], s[-1]] if len(s) >= 2 else [s[0]])
                if len(s) >= 3:
                    k = np.argmax(s[:, 3])
                    if k not in (0, len(s)-1): keep.append(s[k])
            out.extend(keep)
        else:
            out.extend(cluster)
        i = j
    return np.array(out) if out else np.zeros((0, 4))

def _simple_quantization(notes_array: np.ndarray, bpm: float) -> np.ndarray:
    if len(notes_array) == 0:
        return notes_array
    cleaned = notes_array.copy()
    beat = 60.0 / bpm
    sixteenth = beat / 4

    cleaned = cleaned[np.argsort(cleaned[:, 0])]
    i = 0
    while i < len(cleaned):
        start = cleaned[i, 0]
        cluster = [i]
        j = i + 1
        while j < len(cleaned) and (cleaned[j, 0] - start) * 1000 < CHORD_TOLERANCE_MS:
            cluster.append(j)
            j += 1
        if len(cluster) > 1:
            snap = round(start / sixteenth) * sixteenth
            for k in cluster: cleaned[k, 0] = snap
        i = j

    for k in range(len(cleaned)):
        s0 = cleaned[k, 0]
        grid_s = round(s0 / sixteenth) * sixteenth
        cleaned[k, 0] = grid_s
        dur = max(MIN_NOTE_DURATION_SEC, cleaned[k, 1] - s0)
        qn = max(1, round(dur / sixteenth))
        cleaned[k, 1] = grid_s + qn * sixteenth

    for pitch in np.unique(cleaned[:, 2]):
        mask = cleaned[:, 2] == pitch
        seg = cleaned[mask]
        if len(seg) <= 1: continue
        order = np.argsort(seg[:, 0])
        seg = seg[order]
        for i in range(len(seg) - 1):
            gap = seg[i+1, 0] - seg[i, 1]
            if 0 < gap < sixteenth:
                seg[i, 1] = seg[i+1, 0]
            elif gap < 0:
                seg[i, 1] = seg[i+1, 0] - sixteenth/2
        cleaned[mask] = seg

    for k in range(len(cleaned)):
        if cleaned[k, 1] - cleaned[k, 0] < MIN_NOTE_DURATION_SEC:
            cleaned[k, 1] = cleaned[k, 0] + MIN_NOTE_DURATION_SEC
    return cleaned


# %% Cell 4 - MIDI Writers

def _write_format1_dual_track(notes_array: np.ndarray, bpm: float, output_path: Path) -> bool:
    """Format 1: Two instrument tracks, no conductor track 0."""
    if len(notes_array) == 0: return False

    TICKS_PER_BEAT = 480
    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT, type=1)
    tempo = mido.bpm2tempo(max(1.0, float(bpm)))

    def sec_to_ticks(s: float) -> int:
        return max(0, int(mido.second2tick(max(0.0, float(s)), TICKS_PER_BEAT, tempo)))

    num, den = (4, 4)
    if AUTO_CUT_TIME and bpm >= CUTTIME_BPM_THRESHOLD: num, den = (2, 2)

    na = np.round(notes_array, 3)
    treble_notes = na[na[:, 2] >= TREBLE_BASS_SPLIT]
    bass_notes = na[na[:, 2] < TREBLE_BASS_SPLIT]

    def create_track(notes: np.ndarray, name: str, inject_meta: bool = False):
        events = []
        for s, e, p, v in notes:
            pitch = int(p)
            vel = int(min(127, max(1, v)))
            events.append((sec_to_ticks(s), 'on', pitch, vel))
            events.append((sec_to_ticks(e), 'off', pitch, 0))
        
        events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1, x[2]))
        
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=name, time=0))
        track.append(mido.Message('program_change', channel=0, program=0, time=0))

        if inject_meta:
            track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
            track.append(mido.MetaMessage('time_signature', numerator=num, denominator=den,
                                          clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
        
        cur_tick = 0
        for abs_tick, msg_type, pitch, vel in events:
            delta = max(0, abs_tick - cur_tick)
            msg = mido.Message(f'note_{msg_type}', channel=0, note=pitch, velocity=vel, time=delta)
            track.append(msg)
            cur_tick = abs_tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track

    if len(treble_notes) > 0:
        midi.tracks.append(create_track(treble_notes, 'Piano Treble', inject_meta=True))
        if len(bass_notes) > 0:
            midi.tracks.append(create_track(bass_notes, 'Piano Bass', inject_meta=False))
    else:
        if len(bass_notes) > 0:
            midi.tracks.append(create_track(bass_notes, 'Piano Bass', inject_meta=True))

    midi.save(str(output_path))
    print(f"  [Type 1] Wrote {len(treble_notes)} treble, {len(bass_notes)} bass notes")
    return True


# %% Cell 5 - Transcription (BasicPitch)

def _basicpitch_python(wavs: List[str], out_dir: Path) -> int:
    try:
        from basic_pitch.inference import predict_and_save
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError:
        print("[BasicPitch] Module not found. Install with: pip install basic-pitch")
        return 0

    sig = inspect.signature(predict_and_save)
    kw = {}
    for k in ("output_directory", "midi_output_directory"):
        if k in sig.parameters: kw[k] = str(out_dir)
    if "audio_path_list" in sig.parameters:
        kw["audio_path_list"] = wavs
    elif "input_paths" in sig.parameters:
        kw["input_paths"] = wavs
    for pname in ("model_or_model_path", "model_path", "model"):
        if pname in sig.parameters: kw[pname] = ICASSP_2022_MODEL_PATH
    for k, v in [
        ("save_midi", True), ("save_model_outputs", False),
        ("onset_threshold", BP_ONSET_THRESH), ("frame_threshold", BP_FRAME_THRESH),
        ("minimum_note_length", BP_MIN_NOTE_LEN), ("midi_tempo", DEFAULT_BPM)
    ]:
        if k in sig.parameters: kw[k] = v

    try:
        predict_and_save(**kw)
        return _count_midis(out_dir)
    except Exception as e:
        print(f"[BasicPitch] Python API Error: {e}")
        return 0

def _basicpitch_cli(wavs: List[str], out_dir: Path) -> int:
    cli = shutil.which("basic-pitch")
    if not cli: return 0
    before = _count_midis(out_dir)
    for w in wavs:
        cmd = [cli, str(out_dir), w]
        try:
            subprocess.run(cmd, capture_output=True, timeout=300)
        except Exception:
            pass
    return _count_midis(out_dir) - before


# %% Cell 6 - Processing

def _get_original_filename_for_midi(midi_path: Path, media_dir: Path) -> str:
    stem = midi_path.stem
    for suf in ["_basic_pitch", "_transcribed", "_output", "_bp"]:
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break
    if media_dir.exists():
        for f in media_dir.glob("*"):
            if f.stem == stem: return f.stem
    return midi_path.stem

def _process_all_midis(raw_dir: Path, run_dir: Path) -> List[str]:
    finals: List[str] = []
    raw_midis = _list_midis(raw_dir)
    media_dir = run_dir / "_tmp_media"
    
    print(f"\n[Process] Processing {len(raw_midis)} raw MIDI file(s)")
    if not raw_midis: return finals

    for midi_path in raw_midis:
        original_name = _get_original_filename_for_midi(midi_path, media_dir)
        try:
            pmid = pm.PrettyMIDI(str(midi_path))
            arr = [
                [n.start, n.end, n.pitch, n.velocity]
                for inst in pmid.instruments for n in inst.notes
            ]
            if not arr: continue
            
            notes = np.array(arr)
            print(f"\n[File] {midi_path.name}: {len(notes)} raw notes")

            # Remove leading silence by shifting all notes to start at time 0
            if len(notes) > 0:
                min_start = np.min(notes[:, 0])
                if min_start > 0:
                    notes[:, 0] -= min_start  # Shift start times
                    notes[:, 1] -= min_start  # Shift end times
                    print(f"[Trim] Removed {min_start:.2f}s of leading silence")

            bpm = _improved_bpm_detection(notes)
            cleaned = _simple_quantization(notes, bpm)
            cleaned = _reduce_chord_clusters(cleaned, max_notes_per_chord=6)
            print(f"[Clean] {len(cleaned)} notes after processing")

            # Write MIDI file
            out_type1 = run_dir / f"Transcribed_{original_name}.mid"
            if _write_format1_dual_track(cleaned, bpm, out_type1):
                finals.append(str(out_type1))
                print(f"  Saved MIDI: {out_type1.name}")
            
            # Write MusicXML file if notation module is available
            if NOTATION_AVAILABLE:
                try:
                    out_musicxml = run_dir / f"Transcribed_{original_name}"
                    musicxml_path = notation_export.notes_to_musicxml(cleaned, bpm, out_musicxml)
                    print(f"  Saved MusicXML: {musicxml_path.name}")
                except Exception as e:
                    print(f"  [Warning] Could not create MusicXML: {e}")

        except Exception as e:
            print(f"  Error processing {midi_path.name}: {e}")
    
    return finals

def _cleanup_run_directory(run_dir: Path, raw_dir: Path, media_dir: Path):
    for d in (raw_dir, media_dir):
        if d.exists():
            try: shutil.rmtree(d)
            except: pass


# %% Cell 7 - Main Execution

def main(reconfigure: bool = False, serve: bool = False):
    global TRANSCR_ROOT
    
    # Server mode
    if serve:
        print("=" * 70)
        print("STARTING NOTATION SERVER MODE")
        print("=" * 70)
        
        TRANSCR_ROOT = _get_transcription_root(force_reconfigure=False)
        if TRANSCR_ROOT is None:
            print("\n[Error] Cannot start server without a valid transcription folder.")
            return
        
        # Find the most recent run directory
        runs = sorted([d for d in TRANSCR_ROOT.glob(f"{PREFIX}*") if d.is_dir()],
                     key=lambda x: x.stat().st_mtime, reverse=True)
        
        if runs:
            current_run = runs[0]
            print(f"[Server] Using run directory: {current_run}")
            
            # Import and start server
            try:
                import notation_server
                notation_server.set_transcription_root(TRANSCR_ROOT, current_run)
                
                print("\n[Server] Starting Flask server...")
                print("[UI] After server starts, open: http://localhost:3000")
                print("[UI] Or visit: http://localhost:5000 for API")
                print("\nPress Ctrl+C to stop the server.")
                
                notation_server.run_server(debug=False)
            except ImportError:
                print("[Error] notation_server module not found.")
                print("Make sure notation_server.py is in the same directory.")
        else:
            print("[Error] No transcription runs found. Run the pipeline first.")
        return
    
    # Normal transcription mode
    print("=" * 70)
    print("UNIFIED MUSIC TO MIDI TRANSCRIPTION PIPELINE")
    if NOTATION_AVAILABLE:
        print("With MusicXML Export Support")
    print("=" * 70)
    
    TRANSCR_ROOT = _get_transcription_root(force_reconfigure=reconfigure)
    if TRANSCR_ROOT is None:
        print("\n[Error] Cannot proceed without a valid transcription folder. Exiting.")
        return
    
    print(f"\n[Config] Output location: {TRANSCR_ROOT}")
    print("(To change this, run: main(reconfigure=True))")
    
    print("\nThis script generates:")
    print("  • MIDI files with automatic leading silence removal")
    print("  • Dual Track format: Explicitly splits treble/bass for MuseScore")
    print("  • No Track 0: Tempo/time signature embedded in first instrument track")
    if NOTATION_AVAILABLE:
        print("  • MusicXML files for notation viewing/editing")
    print("=" * 70)

    selected_files = _choose_files()
    if not selected_files:
        print("\nNo files selected. Exiting.")
        return

    run_dir = _make_run_dir()
    tmp_media_dir = _ensure_dir(run_dir / "_tmp_media")
    tmp_raw_dir = _ensure_dir(run_dir / "_tmp_raw")
    print(f"\n[Run] Created new directory for this session: {run_dir}")

    media_files = []
    for file_path, start_time, end_time in selected_files:
        path = Path(file_path)
        if path.suffix.lower() in MIDI_EXTS:
            shutil.copy2(path, tmp_raw_dir / path.name)
        else:
            dst = tmp_media_dir / f"{path.stem}.wav"
            if _to_wav_via_ffmpeg(path, dst, start_time, end_time):
                media_files.append(dst)

    if media_files:
        print(f"\n[Transcribe] Transcribing {len(media_files)} audio file(s)...")
        wav_paths = [str(p) for p in media_files]
        produced = _basicpitch_python(wav_paths, tmp_raw_dir)
        if produced == 0:
            print("[Transcribe] Python API failed, trying CLI fallback...")
            produced = _basicpitch_cli(wav_paths, tmp_raw_dir)
        print(f"[Transcribe] Generated {produced} raw MIDI file(s).")

    final_files = _process_all_midis(tmp_raw_dir, run_dir)
    _cleanup_run_directory(run_dir, tmp_raw_dir, tmp_media_dir)

    print("\n" + "=" * 70)
    print("TRANSCRIPTION COMPLETE!")
    print("=" * 70)
    
    if final_files:
        print(f"\nGenerated files in: {run_dir}")
        print("\nOutput files:")
        for f in sorted(final_files):
            print(f"  • {Path(f).name} (MIDI)")
            if NOTATION_AVAILABLE:
                xml_path = Path(f).with_suffix('.mxl')
                if not xml_path.exists():
                    xml_path = Path(f).with_suffix('.musicxml')
                if xml_path.exists():
                    print(f"  • {xml_path.name} (MusicXML)")
        
        print("\n📝 Next steps:")
        print("  1. Import MIDI files into MuseScore for detailed editing")
        if NOTATION_AVAILABLE:
            print("  2. Or run with --serve flag to launch the notation viewer/editor:")
            print(f"     python '{__file__}' --serve")
    else:
        print("\nNo final MIDI files were generated. Please check for errors above.")
    
    print("=" * 70)


if __name__ == "__main__":
    reconfigure = any(arg in sys.argv for arg in ["--reconfigure", "-r"])
    serve = any(arg in sys.argv for arg in ["--serve", "-s"])
    main(reconfigure=reconfigure, serve=serve)