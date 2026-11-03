import os
import sys
import re
import json
import shutil
import subprocess
import threading
import time
import tempfile
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio_config.json")

SAFE_WORDS = {
    "или", "в", "общем", "чего", "там", "человек", "люди", "что", "его", "без", "просто", 
    "либо", "женщина", "девушка", "парень", "буду", "будет", "будут", "виде", "вид", 
    "хоть", "как", "так", "все", "всё", "где", "куда", "когда", "зачем", "почему", 
    "да", "нет", "не", "ни", "он", "она", "они", "мы", "вы", "я", "ты", "был", "была", 
    "были", "есть", "то", "это", "этот", "эта", "эти", "тот", "та", "те", "уже", "еще", 
    "ещё", "тоже", "также", "только", "даже", "вдруг", "сейчас", "потом", "тогда", 
    "здесь", "тут", "туда", "сюда", "откуда", "оттуда", "очень", "много", "мало", 
    "больше", "меньше", "хорошо", "плохо", "можно", "надо", "нужно", "хочу", "хочет", 
    "могу", "может", "хлеб", "колебания", "потребность", "ребенок", "дети", "рубль", 
    "себя", "тебя", "меня", "какого", "такого"
}

DONATION_TRIGGERS = {
    "донат", "задонатил", "рублей", "сообщение", "подписку", "отправил", "голос", 
    "скинул", "спасибо за", "братан", "донейшн", "сотка", "тыща", "озвучка", "подгон",
    "сообщением", "доната", "донатик", "рубликов", "сабку"
}

def find_ffmpeg():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_ffmpeg = os.path.join(script_dir, "ffmpeg.exe")
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg
    cwd_ffmpeg = os.path.join(os.getcwd(), "ffmpeg.exe")
    if os.path.exists(cwd_ffmpeg):
        return cwd_ffmpeg
    return shutil.which("ffmpeg") or "ffmpeg"

def _config_value(key, default):
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get(key, default)
    except Exception:
        return default

def _pick_temp_dir():
    """Временные файлы: env DOTA2STUDIO_TEMP -> temp_dir из конфига -> D:\\Dota2StudioTemp -> системный TEMP."""
    candidates = [
        os.environ.get("DOTA2STUDIO_TEMP", "").strip(),
        str(_config_value("temp_dir", "") or "").strip(),
        "D:\\Dota2StudioTemp",
    ]
    for c in candidates:
        if not c:
            continue
        try:
            os.makedirs(c, exist_ok=True)
            probe = os.path.join(c, f".probe_{os.getpid()}")
            with open(probe, "w", encoding="utf-8") as pf:
                pf.write("ok")
            os.remove(probe)
            return c
        except Exception:
            continue
    return tempfile.gettempdir()

TEMP_DIR = _pick_temp_dir()
try:
    tempfile.tempdir = TEMP_DIR
except Exception:
    pass

_QSV_CACHE = {}

def qsv_available(ffmpeg_bin):
    """Проверка, что Intel Quick Sync реально инициализируется на этой машине."""
    key = ffmpeg_bin or "ffmpeg"
    if key in _QSV_CACHE:
        return _QSV_CACHE[key]
    ok = False
    if not os.environ.get("DOTA2STUDIO_NO_GPU"):
        try:
            # проверяем ровно те опции, которыми потом реально кодируем:
            # некоторые драйверы инициализируются, но падают на global_quality
            res = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "nullsrc=s=320x180:d=1",
                 "-c:v", "h264_qsv", "-preset", "veryfast",
                 "-global_quality", "22", "-look_ahead", "0",
                 "-f", "null", "-"],
                capture_output=True, timeout=25
            )
            ok = res.returncode == 0
        except Exception:
            ok = False
    _QSV_CACHE[key] = ok
    return ok

def use_qsv(ffmpeg_bin):
    """GPU-кодирование включено по умолчанию; отключается use_gpu_qsv=false в конфиге."""
    if not _config_value("use_gpu_qsv", True):
        return False
    return qsv_available(ffmpeg_bin)

_QSV_PRESETS = {"ultrafast": "veryfast", "superfast": "veryfast", "veryfast": "veryfast",
                "faster": "medium", "fast": "fast", "medium": "medium",
                "slow": "slow", "slower": "slower", "veryslow": "veryslow"}

def video_enc_args(ffmpeg_bin, preset="ultrafast", crf=22, fps=60):
    """Аргументы видеокодека: Intel QSV, если доступен, иначе libx264 (как раньше)."""
    if use_qsv(ffmpeg_bin):
        args = ["-c:v", "h264_qsv",
                "-preset", _QSV_PRESETS.get(preset, "veryfast"),
                "-global_quality", str(crf), "-look_ahead", "0"]
    else:
        args = ["-c:v", "libx264", "-preset", preset,
                "-tune", "fastdecode", "-threads", "0", "-crf", str(crf)]
    if fps:
        args += ["-fps_mode", "cfr", "-r", str(fps)]
    return args

def swap_to_cpu(cmd):
    """Переписывает команду ffmpeg с h264_qsv на libx264 — для повторной попытки без GPU."""
    out, i = [], 0
    while i < len(cmd):
        if cmd[i] == "-c:v" and i + 1 < len(cmd) and cmd[i + 1] == "h264_qsv":
            out += ["-c:v", "libx264", "-preset", "ultrafast",
                    "-tune", "fastdecode", "-threads", "0", "-crf", "22"]
            i += 2
            while i < len(cmd) and cmd[i] in ("-preset", "-global_quality", "-look_ahead"):
                i += 2
            continue
        out.append(cmd[i])
        i += 1
    return out

def ffmpeg_reason(stderr):
    """Достаёт из вывода ffmpeg самую информативную строку об ошибке."""
    lines = [l.strip() for l in (stderr or "").splitlines() if l.strip()]
    for l in lines:
        low = l.lower()
        if "error" in low or "invalid" in low or "no such" in low or "not found" in low:
            return l
    return lines[-1] if lines else "неизвестная ошибка"

def ffmpeg_run(cmd, ffmpeg_bin, log=None):
    """Запуск ffmpeg: если GPU-энкодер отказал — один повтор на CPU, чтобы не терять прогон."""
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0 and "h264_qsv" in cmd and use_qsv(ffmpeg_bin):
        if log:
            log("[СИСТЕМА] QSV не смог закодировать — повторяю на CPU (libx264)...")
        r = subprocess.run(swap_to_cpu(cmd), capture_output=True, text=True, errors="replace")
    return r

def ad_probe(ffmpeg_bin, ad_path):
    """(доля кадра под роликом, доля зелёного фона, длительность) — чтобы выбрать способ вставки."""
    try:
        info = subprocess.run([ffmpeg_bin, "-i", ad_path], capture_output=True, text=True, errors="replace").stderr
        m = re.search(r"Video: .*?(\d{2,5})x(\d{2,5})", info)
        d = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", info)
        dur = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)) if d else None
        cov = green = None
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            s = min(1080 / w, 1920 / h)
            cov = (w * s) * (h * s) / (1080 * 1920) * 100
        if CV2_AVAILABLE:
            tmp = os.path.join(TEMP_DIR, f"ad_probe_{int(time.time()*1000)}.png")
            subprocess.run([ffmpeg_bin, "-y", "-ss", "1", "-i", ad_path, "-frames:v", "1",
                            "-vf", "scale=270:480", tmp], capture_output=True)
            img = imread_unicode(tmp)
            if img is not None:
                b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
                green = ((g > 90) & (g - r > 50) & (g - b > 50)).mean() * 100
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
        return cov, green, dur
    except Exception:
        return None, None, None

def shorts_ad_overlay_filter(stack, ad_at, ad_dur):
    """Хромакей: на время ролика шортс встаёт на паузу — кадр замирает, звук замолкает, потом идёт дальше."""
    end = ad_at + ad_dur
    ms = int(ad_at * 1000)
    return stack + (
        ",fps=60,setsar=1,format=yuv420p,split=2[v1][v2];"
        f"[v1]trim=0:{ad_at:.2f},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={ad_dur:.2f}[vf];"
        f"[v2]trim=start={ad_at:.2f},setpts=PTS-STARTPTS[v2t];"
        "[vf][v2t]concat=n=2:v=1:a=0[base];"
        "[1:v]scale=1080:1920,chromakey=0x00FF00:0.12:0.06,fps=60,"
        f"setpts=PTS+{ad_at:.2f}/TB,format=yuva420p[ad];"
        f"[base][ad]overlay=0:0:eof_action=pass:enable='between(t,{ad_at:.2f},{end:.2f})'[v];"
        "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asplit=2[a1][a2];"
        f"[a1]atrim=0:{ad_at:.2f},asetpts=PTS-STARTPTS,apad=pad_dur={ad_dur:.2f}[af];"
        f"[a2]atrim=start={ad_at:.2f},asetpts=PTS-STARTPTS[a2t];"
        "[af][a2t]concat=n=2:v=0:a=1[am];"
        "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"adelay={ms}|{ms}[aa];"
        "[am][aa]amix=inputs=2:duration=first:normalize=0[a]"
    )

def shorts_ad_splice_filter(stack, ad_at):
    """Ролик без прозрачности: врезается в таймлайн целиком, остальное сдвигается на его длину."""
    return stack + (
        ",fps=60,setsar=1,format=yuv420p,split=2[v1][v2];"
        f"[v1]trim=0:{ad_at},setpts=PTS-STARTPTS,format=yuv420p[va];"
        f"[v2]trim=start={ad_at},setpts=PTS-STARTPTS,format=yuv420p[vb];"
        "[1:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=60,format=yuv420p[v_ad];"
        "[va][v_ad][vb]concat=n=3:v=1:a=0[v];"
        "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asplit=2[a1][a2];"
        f"[a1]atrim=0:{ad_at},asetpts=PTS-STARTPTS[aa];"
        f"[a2]atrim=start={ad_at},asetpts=PTS-STARTPTS[ab];"
        "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[aad];"
        "[aa][aad][ab]concat=n=3:v=0:a=1[a]"
    )

def imread_unicode(path):
    try:
        with open(path, "rb") as f:
            bytes_data = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(bytes_data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None

def imwrite_unicode(path, img):
    try:
        ext = os.path.splitext(path)[1] or ".png"
        success, buffer = cv2.imencode(ext, img)
        if success:
            with open(path, "wb") as f:
                f.write(buffer)
            return True
    except Exception:
        pass
    return False

def remove_fake_checkerboard_transparency(image_path, logger=None):
    if not CV2_AVAILABLE or not os.path.exists(image_path):
        return image_path

    img = imread_unicode(image_path)
    if img is None:
        return image_path

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    b, g, r, a = cv2.split(img)

    r_f = r.astype(np.float32)
    g_f = g.astype(np.float32)
    b_f = b.astype(np.float32)

    color_diff = np.maximum(np.abs(r_f - g_f), np.maximum(np.abs(r_f - b_f), np.abs(g_f - b_f)))
    brightness = (r_f + g_f + b_f) / 3.0

    checker_mask = (color_diff < 18) & (brightness > 150)

    if np.count_nonzero(checker_mask) > 100:
        if logger:
            logger("Вырезаем нарисованные квадраты (делаем фон прозрачным)...")
        a[checker_mask] = 0

        cleaned = cv2.merge([b, g, r, a])
        out_name = os.path.join(os.path.dirname(image_path), "clean_transparent_logo.png")
        if imwrite_unicode(out_name, cleaned):
            return out_name

    return image_path

def find_system_font():
    candidates = [
        "C\\:/Windows/Fonts/segoeui.ttf",
        "C\\:/Windows/Fonts/arial.ttf",
        "C\\:/Windows/Fonts/calibri.ttf"
    ]
    for c in candidates:
        normal_p = c.replace("\\:", ":")
        if os.path.exists(normal_p):
            return c
    return "Arial"

def find_default_swears_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = ["ru_curse_words.txt", "swears.txt", "russian_swears.txt", "mat.txt"]
    for fname in candidates:
        path = os.path.join(script_dir, fname)
        if os.path.exists(path):
            return path
    return ""

def find_default_overlay_image():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_name = "png-transparent-dota-2-the-international-2016-electronic-sports-video-game-others-game-dragon-video-game.png"
    target_path = os.path.join(script_dir, target_name)
    if os.path.exists(target_path):
        return target_path

    for f in os.listdir(script_dir):
        if f.lower().endswith(".png") and ("dragon" in f.lower() or "dota" in f.lower() or "clean" in f.lower()):
            return os.path.join(script_dir, f)
    return ""

def load_user_swears(file_path):
    if not file_path or not os.path.exists(file_path):
        return set()
    words = set()
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w and len(w) >= 2 and w not in SAFE_WORDS:
                words.add(w)
    return words

def is_bad_word(word_str, swear_dict):
    clean = re.sub(r"[^\w]", "", word_str).lower().strip()
    if not clean or len(clean) < 2 or clean in SAFE_WORDS:
        return False
    if clean in swear_dict:
        return True
    for bad in swear_dict:
        if len(bad) >= 3 and bad in clean:
            return True
    return False

def parse_time_str(time_str):
    if not time_str or not time_str.strip():
        return None
    parts = time_str.strip().split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
    except ValueError:
        return None
    return None

def format_time_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def detect_fonbet_box_single_frame(frame):
    if frame is None:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 75, 65]), np.array([16, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 75, 65]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    max_score = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (90 <= w <= 650) and (20 <= h <= 180):
            aspect = w / float(h)
            if 1.8 <= aspect <= 7.0:
                roi = red_mask[y:y+h, x:x+w]
                density = cv2.countNonZero(roi) / float(w * h)
                if density > 0.18:
                    score = (w * h) * density
                    if score > max_score:
                        max_score = score
                        best_box = (x, y, w, h)
    return best_box

def detect_dota_kills(video_path, start_sec, end_sec, logger=None):
    """
    Убийства Dota 2 по смене счёта в верхней панели.
    Читаем только боксы с цифрами (часы и портреты героев в них не попадают),
    4 кадра/с через ffmpeg — быстро даже на двухчасовом VOD.
    """
    if not os.path.exists(video_path) or end_sec - start_sec < 5:
        return []

    fps = 4
    BW, BH = 30, 16
    # доли кадра: узкие боксы ровно по цифрам (в 1920x1080 это x 872-902 и 1020-1050, y 12-28).
    # Портреты героев и часы в них не попадают, фон панели занимает минимум площади.
    boxes = ((0.45417, 0.011111), (0.53125, 0.011111))
    chains = [
        f"[0:v]crop=iw*0.015625:ih*0.014815:iw*{fx}:ih*{fy},scale={BW}:{BH},format=gray[b{i}]"
        for i, (fx, fy) in enumerate(boxes)
    ]
    fc = ";".join(chains) + ";[b0][b1]hstack,format=gray,fps=4[out]"

    cmd = [find_ffmpeg(), "-v", "error", "-ss", f"{start_sec:.2f}", "-to", f"{end_sec:.2f}",
           "-i", video_path, "-filter_complex", fc, "-map", "[out]",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True).stdout
    except Exception:
        return []

    fw = BW * 2
    n = len(raw) // (fw * BH)
    if n < 5:
        if logger:
            logger("[CV-ДЕТЕКТОР] Полосу счёта прочитать не удалось.")
        return []

    frames = np.frombuffer(raw[: n * fw * BH], dtype=np.uint8).reshape(n, BH, fw)
    # только белые цифры: панель счёта полупрозрачная, и без бинаризации фон «шевелится» от игры
    mask = (frames > 200).astype(np.int16)
    delta = np.abs(np.diff(mask, axis=0))
    # боксы считаем раздельно: драка шумит в обоих, а смена цифры — только в одном
    per_box = (delta[:, :, :BW].sum(axis=(1, 2)), delta[:, :, BW:].sum(axis=(1, 2)))

    # Смена счёта — это устойчивый шаг: цифры поменялись и дальше стоят на месте.
    kill_times = []
    last_t = -999.0
    for i in range(1, len(delta) - 2):
        stepped = any(d[i] >= 8 and d[i + 1] <= 4 and d[i + 2] <= 4 for d in per_box)
        if not stepped:
            continue
        t = start_sec + (i + 1) / float(fps)
        if t - last_t < 1.5:
            continue
        kill_times.append(t)
        last_t = t

    # Захватываем 15 секунд ДО момента убийства (завязка) и 10 секунд ПОСЛЕ
    kill_segments = [[max(start_sec, kt - 15.0), min(end_sec, kt + 10.0)] for kt in kill_times]

    # Склеиваем непрерывные драки и серии фрагов
    merged_kills = []
    for seg in kill_segments:
        if not merged_kills:
            merged_kills.append(seg[:])
        elif seg[0] <= merged_kills[-1][1] + 8.0:
            merged_kills[-1][1] = max(merged_kills[-1][1], seg[1])
        else:
            merged_kills.append(seg[:])

    if logger:
        logger(f"[CV-ДЕТЕКТОР] Смен счёта: {len(kill_times)}, после склейки драк: {len(merged_kills)}.")

    return merged_kills

def transcribe_speech_intervals(video_path, windows, logger=None):
    """Фразы речи вокруг указанных моментов. windows: [(t0, t1)]. Возвращает [(начало, конец)] в секундах исходника."""
    if not WHISPER_AVAILABLE or not windows:
        return []

    ff = find_ffmpeg()
    tmp_wav = os.path.join(TEMP_DIR, f"speech_{int(time.time()*1000)}.wav")
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass

    model = None
    phrases = []
    try:
        for i, (w0, w1) in enumerate(windows):
            subprocess.run([ff, "-y", "-ss", f"{w0:.2f}", "-to", f"{w1:.2f}", "-i", video_path,
                            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp_wav],
                           capture_output=True)
            if not os.path.exists(tmp_wav):
                continue
            if model is None:
                model = WhisperModel("base", device="cuda" if has_cuda else "cpu",
                                     compute_type="float16" if has_cuda else "int8", cpu_threads=4)
            segments, _ = model.transcribe(tmp_wav, language="ru", word_timestamps=True,
                                           vad_filter=True, beam_size=1)
            for seg in segments:
                for w in (seg.words or []):
                    a, b = w0 + w.start, w0 + w.end
                    if phrases and a - phrases[-1][1] < 0.7:
                        phrases[-1][1] = max(phrases[-1][1], b)
                    else:
                        phrases.append([a, b])
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
            if logger and (i + 1) % 10 == 0:
                logger(f"[РЕЧЬ] Проверено склеек: {i+1}/{len(windows)}")
    except Exception as e:
        if logger:
            logger(f"[РЕЧЬ] Ошибка распознавания: {e}")
    finally:
        if os.path.exists(tmp_wav):
            try: os.remove(tmp_wav)
            except: pass

    return [tuple(p) for p in phrases]

def extend_cuts_to_speech(segments, phrases, limit, tail=3.0):
    """Если конец куска попадает на речь — тянем до конца фразы + tail (но не дальше limit)."""
    result = []
    for s, e in segments:
        end = e
        for a, b in phrases:
            if a <= e <= b:
                end = min(limit, max(end, b + tail))
        result.append([s, end])
    return result

def make_tg_intro(avatar_path, link, out_path, ffmpeg_bin, dur=4.0):
    """Анимированная вставка-заставка с логотипом и ссылкой на ТГК: 1920x1080, 60 fps, dur секунд."""
    font = "C\\:/Windows/Fonts/arialbd.ttf"
    if not os.path.exists("C:/Windows/Fonts/arialbd.ttf"):
        font = find_system_font()
    link = (link or "").replace("'", "").replace(":", r"\:")

    fc = (
        f"color=c=black:s=1920x1080:r=60:d={dur}[bg];"
        "[0:v]split=2[src1][src2];"
        "[src1]scale=1500:1500:force_original_aspect_ratio=increase,crop=1500:1500,"
        "gblur=sigma=80,format=rgba,colorchannelmixer=aa=0.30[glow];"
        "[src2]scale=760:760:force_original_aspect_ratio=decrease,"
        "pad=760:760:(ow-iw)/2:(oh-ih)/2:color=black,"
        "colorkey=0x000000:0.14:0.03,format=rgba[logo];"
        "[bg][glow]overlay=(W-w)/2:(H-h)/2[g1];"
        "[g1][logo]overlay=(W-w)/2:80[base];"
        "[base]zoompan=z='if(lt(on,18),1+0.35*pow(1-on/18,2),1+0.05*(on-18)/222)':"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=60[z];"
        f"color=c=0xFFD400:s=900x9:r=60:d={dur}[s1];"
        "[z][s1]overlay=x='-900+3000*t':y=300:enable='between(t,0.28,0.85)'[k1];"
        f"color=c=0xFF00AA:s=700x7:r=60:d={dur}[s2];"
        "[k1][s2]overlay=x='1920-2600*t':y=760:enable='between(t,0.42,0.95)'[k2];"
        f"color=c=0x00E5FF:s=700x7:r=60:d={dur}[s3];"
        "[k2][s3]overlay=x='-700+2600*t':y=800:enable='between(t,0.52,1.05)'[k3];"
        f"color=c=white:s=1920x1080:r=60:d={dur},format=rgba,colorchannelmixer=aa=0.8[fl];"
        "[k3][fl]overlay=enable='between(t,0.28,0.33)'[k4];"
        f"[k4]drawtext=fontfile='{font}':text='ТГК':fontcolor=white:fontsize=44:"
        "x=(w-tw)/2:y=880:alpha='if(lt(t,1.0),0,min(1,(t-1.0)/0.3))'[d1];"
        f"[d1]drawtext=fontfile='{font}':text='{link}':fontcolor=0xFFD400:fontsize=64:"
        "x=(w-tw)/2:y='945+40*(1-min(1,(t-1.15)/0.35))':"
        "alpha='if(lt(t,1.15),0,min(1,(t-1.15)/0.35))'[d2];"
        "[d2]fade=t=in:st=0:d=0.12,fade=t=out:st=3.70:d=0.30,format=yuv420p[v]"
    )

    cmd = [ffmpeg_bin, "-y",
           "-loop", "1", "-t", str(dur), "-i", avatar_path,
           "-f", "lavfi", "-t", str(dur), "-i", "anullsrc=r=44100:cl=stereo",
           "-filter_complex", fc, "-map", "[v]", "-map", "1:a",
           *video_enc_args(ffmpeg_bin, "ultrafast", 22, 60),
           "-c:a", "pcm_s16le", "-ar", "44100",
           "-t", str(dur), out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError("Не собралась вставка ТГК:\n" + r.stderr.strip()[-400:])
    return out_path

def probe_media(ffmpeg_bin, path):
    """(ширина, высота, fps, есть_аудио, длительность) — из заголовка файла."""
    try:
        info = subprocess.run([ffmpeg_bin, "-i", path], capture_output=True, text=True,
                              errors="replace").stderr
    except Exception:
        return None, None, None, False, None
    w = h = fps = dur = None
    m = re.search(r"Video: .*?(\d{2,5})x(\d{2,5})", info)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?) fps", info)
    if m:
        fps = float(m.group(1))
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", info)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return w, h, fps, ("Audio:" in info), dur

def output_to_source_time(segments, t_out):
    """Переводит время готового ролика во время исходного VOD по списку склеек."""
    acc = 0.0
    for s, e in segments:
        d = max(0.0, e - s)
        if t_out <= acc + d:
            return s + (t_out - acc)
        acc += d
    return None

def insert_ad_at(video_path, ad_path, at_sec, out_path, ffmpeg_bin, total_hint=None, log_callback=None):
    """Вставляет рекламный ролик в видео на заданном времени. Один проход, без промежуточных файлов."""
    W, H, fps, _, dur = probe_media(ffmpeg_bin, video_path)
    _, _, _, ad_audio, ad_dur = probe_media(ffmpeg_bin, ad_path)
    if not W or not H:
        raise RuntimeError("Не удалось прочитать параметры видео")
    fps = fps or 60
    at = max(0.1, float(at_sec))
    if dur:
        at = min(at, max(0.1, dur - 0.1))

    fc = (
        f"[0:v]trim=0:{at:.3f},setpts=PTS-STARTPTS,fps={fps:g},setsar=1[v1];"
        f"[0:v]trim=start={at:.3f},setpts=PTS-STARTPTS,fps={fps:g},setsar=1[v2];"
        f"[1:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps:g},format=yuv420p[vad];"
        "[v1][vad][v2]concat=n=3:v=1:a=0[v];"
        f"[0:a]atrim=0:{at:.3f},asetpts=PTS-STARTPTS,aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[aa1];"
        f"[0:a]atrim=start={at:.3f},asetpts=PTS-STARTPTS,aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[aa2];"
    )
    src_a = "1:a" if ad_audio else "2:a"
    fc += f"[{src_a}]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[aad];"
    fc += "[aa1][aad][aa2]concat=n=3:v=0:a=1[a]"

    cmd = [ffmpeg_bin, "-y", "-i", video_path, "-i", ad_path]
    if not ad_audio:
        cmd += ["-f", "lavfi", "-t", f"{(ad_dur or 1.0):.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]"]
    cmd += video_enc_args(ffmpeg_bin, "faster", 20, fps)
    cmd += ["-c:a", "aac", "-b:a", "192k", out_path]

    total = total_hint or ((dur or 0) + (ad_dur or 0))
    if log_callback:
        run_ffmpeg_with_progress(cmd, total, log_callback)
    else:
        r = ffmpeg_run(cmd, ffmpeg_bin)
        if r.returncode != 0:
            raise RuntimeError("Не удалось вставить рекламу:\n" + r.stderr.strip()[-400:])
    return out_path

def generate_cuts_report(report_path, teaser_segs, story_segs, output_video_path):
    """Создает подробный текстовый отчет со склейками и главами YouTube."""
    try:
        all_segs = []
        for s, e in teaser_segs:
            all_segs.append({"type": "⚡ Тизер (YouTube-Хук)", "start": s, "end": e})
        for s, e in story_segs:
            all_segs.append({"type": "🎮 Эпизод", "start": s, "end": e})

        out_time = 0.0
        prev_src_end = None

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"ОТЧЕТ О МОНТАЖЕ: ТАЙМКОДЫ ГОТОВОГО ВИДЕО И МЕСТА СКЛЕЕК\n")
            f.write(f"Итоговый файл: {os.path.basename(output_video_path)}\n")
            f.write("=" * 80 + "\n\n")

            f.write("📋 ГОТОВЫЕ ГЛАВЫ ДЛЯ ОПИСАНИЯ НА YOUTUBE:\n")
            temp_time = 0.0
            for idx, item in enumerate(all_segs):
                dur = item["end"] - item["start"]
                f.write(f"{format_time_hms(temp_time)} - {item['type']} #{idx+1}\n")
                temp_time += dur

            f.write("\n" + "=" * 80 + "\n")
            f.write("✂️ ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О СКЛЕЙКАХ (ЧТО И ГДЕ ВЫРЕЗАНО):\n")
            f.write("=" * 80 + "\n\n")

            for idx, item in enumerate(all_segs):
                s_orig, e_orig = item["start"], item["end"]
                seg_dur = e_orig - s_orig
                out_start = out_time
                out_end = out_time + seg_dur

                if prev_src_end is not None and s_orig > prev_src_end:
                    skipped = s_orig - prev_src_end
                    f.write(f"--- [СКЛЕЙКА НА {format_time_hms(out_start)}] ВЫРЕЗАНО ИЗ СТРИМА: {format_time_hms(prev_src_end)} -> {format_time_hms(s_orig)} (пропущено {format_time_hms(skipped)}) ---\n")

                f.write(f"• [{format_time_hms(out_start)} - {format_time_hms(out_end)}] {item['type']} #{idx+1} (Длина: {int(seg_dur)} сек)\n")
                f.write(f"    └─ Таймкод в исходном стриме: {format_time_hms(s_orig)} -> {format_time_hms(e_orig)}\n")

                out_time += seg_dur
                prev_src_end = e_orig

        return report_path
    except Exception:
        return None

def run_ffmpeg_with_progress(cmd, total_duration_sec, log_callback):
    """Рендер с прогрессом; при отказе GPU-энкодера повторяет на CPU."""
    try:
        return _run_ffmpeg_progress(cmd, total_duration_sec, log_callback)
    except RuntimeError:
        if "h264_qsv" in cmd and use_qsv(find_ffmpeg()):
            log_callback("[СИСТЕМА] QSV не смог закодировать — повторяю на CPU (libx264)...")
            return _run_ffmpeg_progress(swap_to_cpu(cmd), total_duration_sec, log_callback)
        raise

def _run_ffmpeg_progress(cmd, total_duration_sec, log_callback):
    cmd_with_progress = cmd[:-1] + ["-progress", "pipe:1", cmd[-1]]
    err_log_path = os.path.join(TEMP_DIR, "ffmpeg_render_stderr.log")

    with open(err_log_path, "w", encoding="utf-8", errors="replace") as err_file:
        process = subprocess.Popen(
            cmd_with_progress,
            stdout=subprocess.PIPE,
            stderr=err_file,
            universal_newlines=True,
            bufsize=1
        )

        last_log_time = time.time()
        out_time_sec = 0.0
        fps = "0"

        for line in process.stdout:
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    us = int(line.split("=")[1])
                    out_time_sec = us / 1000000.0
                except:
                    pass
            elif line.startswith("fps="):
                try:
                    fps = line.split("=")[1].strip()
                except:
                    fps = "0"

        now = time.time()
        if now - last_log_time >= 2.0 and total_duration_sec > 0:
            last_log_time = now
            pct = min(99.0, (out_time_sec / total_duration_sec) * 100.0)
            cur_m, cur_s = int(out_time_sec // 60), int(out_time_sec % 60)
            tot_m, tot_s = int(total_duration_sec // 60), int(total_duration_sec % 60)
            log_callback(f"[ПРОГРЕСС РЕНДЕРА] {pct:.1f}% ({cur_m}:{cur_s:02d} / {tot_m}:{tot_s:02d}) | Скорость: {fps} FPS")

        process.wait()

    if process.returncode != 0:
        err_msg = ""
        if os.path.exists(err_log_path):
            with open(err_log_path, "r", encoding="utf-8", errors="replace") as ef:
                err_msg = ef.read().strip()[-500:]
        raise RuntimeError(f"FFmpeg ошибка (код {process.returncode}):\n{err_msg}")

    if os.path.exists(err_log_path):
        try: os.remove(err_log_path)
        except: pass

    return process.returncode

def download_top_twitch_clips_ytdlp(channel_name, range_code, limit, temp_dir, log_callback):
    url = f"https://www.twitch.tv/{channel_name}/clips?filter=clips&range={range_code}"
    log_callback(f"Поиск клипов: {url}")

    os.makedirs(temp_dir, exist_ok=True)
    out_template = os.path.join(temp_dir, "clip_%(autonumber)03d.%(ext)s")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--playlist-end", str(limit),
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", out_template,
        "--print", "after_move:title",
        "--no-warnings",
        url
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            bufsize=1
        )
    except FileNotFoundError:
        raise RuntimeError("Утилита yt-dlp не установлена! Выполните: pip install yt-dlp")

    clip_titles = []
    for line in process.stdout:
        line_s = line.strip()
        if line_s and not line_s.startswith("["):
            clip_titles.append(line_s)
            log_callback(f"[{len(clip_titles)}/{limit}] Загружен: «{line_s}»")
        elif "Downloading item" in line_s or "Downloading" in line_s:
            log_callback(f"   {line_s}")

    process.wait()

    mp4_files = sorted([
        os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
        if f.endswith(".mp4") and os.path.getsize(os.path.join(temp_dir, f)) > 50000
    ])

    results = []
    for i, file_p in enumerate(mp4_files):
        t = clip_titles[i] if i < len(clip_titles) else f"Клип #{i+1}"
        results.append({"path": file_p, "title": t})

    return results

def _ff_path(p):
    """Путь для фильтра ffmpeg: прямые слэши и экранированное двоеточие диска."""
    return "'" + os.path.abspath(p).replace("\\", "/").replace(":", r"\:") + "'"

def detect_banner_box(ffmpeg_bin, path, frames=(0.2, 0.5, 0.8)):
    """Плашка букмекера в правом верхнем или нижнем углу.
    Возвращает (box, чисто_ли_найдена). Координаты исходника."""
    if not CV2_AVAILABLE or not os.path.exists(path):
        return None, False
    best = None
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * f))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            H, W = frame.shape[:2]
            for fx, fy, fw, fh in ((0.78, 0.015, 0.20, 0.09), (0.74, 0.70, 0.24, 0.12)):
                x, y, w, h = int(W * fx), int(H * fy), int(W * fw), int(H * fh)
                box = detect_fonbet_box_single_frame(frame[y:y + h, x:x + w])
                if not box:
                    continue
                bx, by, bw, bh = box
                # если «плашкой» оказался весь угол — фон тоже красный, точных границ нет
                if bw * bh < w * h * 0.7:
                    return (x + bx, y + by, bw, bh), True
                if best is None:
                    best = (x + bx, y + by, bw, bh)
    except Exception:
        pass
    finally:
        if cap is not None:
            cap.release()
    return best, False

def build_clips_compilation(clip_items, out_path, ffmpeg_bin, titles=True, fade=True,
                            overlay_png=None, log=None):
    """Нормализует клипы, накладывает титры и заглушку баннера, склеивает в один ролик."""
    if not clip_items:
        raise RuntimeError("Клипы не скачались")

    W, H, fps, _, _ = probe_media(ffmpeg_bin, clip_items[0]["path"])
    W, H = W or 1920, H or 1080
    fps = fps or 60

    tmp_dir = os.path.join(TEMP_DIR, "clips_norm")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    logo = None
    if overlay_png and os.path.exists(overlay_png):
        logo = remove_fake_checkerboard_transparency(overlay_png, log)

    font_file = "C\\:/Windows/Fonts/arialbd.ttf"
    if not os.path.exists("C:/Windows/Fonts/arialbd.ttf"):
        font_file = find_system_font()

    names = []
    entries = []
    acc = 0.0

    # один раз ищем плашку по всем клипам: раскладка у стрима одинаковая,
    # поэтому надёжный бокс из любого клипа годится для остальных
    boxes = [(None, False)] * len(clip_items)
    canonical = None
    if logo:
        for j, item in enumerate(clip_items):
            b, clean = detect_banner_box(ffmpeg_bin, item["path"])
            boxes[j] = (b, clean)
            if clean and canonical is None:
                canonical = b
        if canonical and log:
            log(f"[КЛИПЫ] Плашка: x={canonical[0]}, y={canonical[1]}, {canonical[2]}x{canonical[3]}")
        elif log:
            log("[КЛИПЫ] Плашку не нашёл — баннер не закрываю")

    for i, item in enumerate(clip_items):
        path = item["path"]
        title = " ".join((item.get("title") or "").split())
        src_w, src_h, _, has_audio, dur = probe_media(ffmpeg_bin, path)
        if not src_w or not src_h or not dur:
            if log:
                log(f"[КЛИПЫ] Пропуск {os.path.basename(path)}: файл не читается")
            continue
        src_w, src_h = src_w or W, src_h or H

        inputs = ["-i", path]
        chains = [f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={fps:g}[base]"]
        cur = "base"
        title_y = 40

        if logo:
            b, clean = boxes[i]
            box = b if clean else (canonical or b)
            if box:
                bx, by, bw, bh = box
                bx, by = int(bx * W / src_w), int(by * H / src_h)
                bw, bh = int(bw * W / src_w), int(bh * H / src_h)
                tw, th = int(bw * 1.5), int(bh * 1.5)
                px = max(0, min(W - tw, bx + bw // 2 - tw // 2))
                py = max(0, min(H - th, by + bh // 2 - th // 2))
                # картинка обычно вырезана с прозрачным фоном — под неё кладём плотную подложку,
                # иначе плашка просвечивает по краям. Берём с запасом: контур может быть на пару пикселей уже
                mx = max(4, int(bw * 0.04))
                my = max(4, int(bh * 0.08))
                fx = max(0, bx - mx)
                fy = max(0, by - my)
                fw2 = min(W - fx, bw + mx * 2)
                fh2 = min(H - fy, bh + my * 2)
                chains.append(f"[{cur}]drawbox=x={fx}:y={fy}:w={fw2}:h={fh2}:color=black:t=fill[bgb]")
                inputs += ["-i", logo]
                chains.append(f"[1:v]scale={tw}:{th}[logo]")
                chains.append(f"[bgb][logo]overlay={px}:{py}[ov]")
                cur = "ov"
                title_y = min(H - 120, py + th + 24)

        if titles and title:
            tf = os.path.join(tmp_dir, f"title_{i:03d}.txt")
            with open(tf, "w", encoding="utf-8") as f:
                f.write(title[:80])
            fs = max(26, int(H * 0.042))
            chains.append(
                f"[{cur}]drawtext=fontfile='{font_file}':textfile={_ff_path(tf)}:"
                f"fontcolor=white:fontsize={fs}:x=w-tw-40:y={title_y}:"
                f"box=1:boxcolor=black@0.6:boxborderw=14:fix_bounds=1[txt]"
            )
            cur = "txt"

        if fade and dur > 1.2:
            chains.append(f"[{cur}]fade=t=in:st=0:d=0.2,fade=t=out:st={max(0.0, dur - 0.2):.2f}:d=0.2[v]")
            cur = "v"

        a_idx = "0:a"
        if not has_audio:
            a_idx = str(len(inputs) // 2)
            inputs += ["-f", "lavfi", "-t", f"{max(1.0, dur):.3f}", "-i", "anullsrc=r=44100:cl=stereo"]

        name = f"norm_{i:03d}.mkv"
        out_i = os.path.join(tmp_dir, name)
        head = ["-filter_complex", ";".join(chains), "-map", f"[{cur}]"]
        tail = video_enc_args(ffmpeg_bin, "ultrafast", 22, fps) + [
            "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}", out_i
        ]

        r = ffmpeg_run([ffmpeg_bin, "-y"] + inputs + head + ["-map", a_idx] + tail, ffmpeg_bin, log)
        if r.returncode != 0 and has_audio:
            # частая причина нулевых кадров у клипов Twitch — битая аудиодорожка
            if log:
                log(f"[КЛИПЫ] Клип {i + 1}: звук не читается, подставляю тишину")
            s_idx = len(inputs) // 2
            inputs2 = inputs + ["-f", "lavfi", "-t", f"{max(1.0, dur):.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
            r = ffmpeg_run([ffmpeg_bin, "-y"] + inputs2 + head + ["-map", str(s_idx)] + tail, ffmpeg_bin, log)

        if r.returncode != 0:
            if log:
                log(f"[КЛИПЫ] Клип {i + 1} ({os.path.basename(path)}) не обработался — пропускаю: "
                    f"{ffmpeg_reason(r.stderr)}")
            continue
        names.append(name)
        real_dur = probe_media(ffmpeg_bin, os.path.join(tmp_dir, name))[4] or dur
        entries.append((acc, real_dur, title))
        acc += real_dur
        if log:
            log(f"[КЛИПЫ] {i + 1}/{len(clip_items)} готов" + (f": {title[:40]}" if title else ""))

    if not names:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Ни один клип не удалось обработать — смотри причины в логе")

    with open(os.path.join(tmp_dir, "list.txt"), "w", encoding="utf-8") as f:
        for n in names:
            f.write(f"file '{n}'\n")
    r = subprocess.run([ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
                        "-c", "copy", os.path.abspath(out_path)],
                       cwd=tmp_dir, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError("Не удалось склеить клипы:\n" + r.stderr.strip()[-300:])

    report = os.path.splitext(out_path)[0] + "_таймкоды.txt"
    try:
        with open(report, "w", encoding="utf-8") as f:
            f.write("СБОРНИК КЛИПОВ — ТАЙМКОДЫ\n")
            f.write(f"Файл: {os.path.basename(out_path)}\n")
            f.write(f"Клипов: {len(entries)}, длительность: {format_time_hms(acc)}\n\n")
            f.write("ГОТОВЫЕ ГЛАВЫ ДЛЯ ОПИСАНИЯ НА YOUTUBE:\n")
            for st, d, t in entries:
                f.write(f"{format_time_hms(st)} - {t or 'клип'} ({format_time_hms(d)})\n")
        if log:
            log(f"[КЛИПЫ] Таймкоды: {os.path.basename(report)}")
    except Exception as e:
        if log:
            log(f"[КЛИПЫ] Не удалось записать таймкоды: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

def deepseek_chat(prompt, api_key, system=None, model="deepseek-chat", max_tokens=2000, timeout=120):
    """Запрос к DeepSeek (OpenAI-совместимый API). Без сторонних библиотек."""
    key = (api_key or "").strip()
    if not key:
        raise RuntimeError("Не указан ключ DeepSeek — впиши его в поле или задай DEEPSEEK_API_KEY")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(DEEPSEEK_URL, data=body, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        if e.code == 401:
            raise RuntimeError("DeepSeek: неверный ключ (401)")
        if e.code == 402:
            raise RuntimeError("DeepSeek: на балансе нет денег (402)")
        raise RuntimeError(f"DeepSeek вернул {e.code}: {detail or e.reason}")
    except Exception as e:
        raise RuntimeError(f"Нет связи с DeepSeek: {e}")
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError("DeepSeek ответил неожиданным форматом")

def transcribe_segments(video_path, log=None):
    """Расшифровка речи с таймкодами: список (секунды, текст)."""
    if not WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper не установлен — расшифровать нечем")
    ff = find_ffmpeg()
    tmp_wav = os.path.join(TEMP_DIR, f"desc_{int(time.time()*1000)}.wav")
    subprocess.run([ff, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", tmp_wav], capture_output=True)
    if not os.path.exists(tmp_wav):
        raise RuntimeError("Не удалось вытащить звук из видео")
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    try:
        if log:
            log("[ОПИСАНИЕ] Распознаю речь (это самый долгий шаг)...")
        model = WhisperModel("base", device="cuda" if has_cuda else "cpu",
                             compute_type="float16" if has_cuda else "int8", cpu_threads=4)
        segments, _ = model.transcribe(tmp_wav, language="ru", vad_filter=True, beam_size=1)
        rows = []
        for s in segments:
            txt = (s.text or "").strip()
            if txt:
                rows.append((s.start, txt))
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass
    if log:
        log(f"[ОПИСАНИЕ] Распознано фрагментов: {len(rows)}")
    return rows

def read_chapters(video_path):
    """Ищет рядом с видео файл таймкодов от монтажа/сборника и достаёт главы [(секунды, название)]."""
    base = os.path.splitext(os.path.abspath(video_path))[0]
    d, name = os.path.split(base)
    candidates = []
    try:
        for f in os.listdir(d or "."):
            low = f.lower()
            if low.startswith(name.lower()) and low.endswith(".txt") and ("таймкод" in low or "глав" in low):
                candidates.append(os.path.join(d, f))
    except Exception:
        return []
    if not candidates:
        return []

    chapters = []
    in_block = False
    try:
        with open(sorted(candidates)[0], encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.rstrip("\n")
                if "ГОТОВЫЕ ГЛАВЫ" in s:
                    in_block = True
                    continue
                if not in_block:
                    continue
                if not s.strip():
                    if chapters:
                        break
                    continue
                m = re.match(r"\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*(.+)", s)
                if m:
                    t = parse_time_str(m.group(1))
                    if t is not None:
                        title = re.sub(r"\s*\(\d{1,2}:\d{2}(?::\d{2})?\)\s*$", "", m.group(2)).strip()
                        chapters.append((t, title))
    except Exception:
        return []
    return chapters

def _fix_chapters(answer, chapters):
    """Ставит в главы реальные таймкоды, оставляя названия от модели."""
    lines = answer.splitlines()
    out, i = [], 0
    while i < len(lines) and not lines[i].strip().upper().startswith("ГЛАВЫ"):
        out.append(lines[i])
        i += 1

    names = []
    if i < len(lines):
        i += 1
        while i < len(lines):
            s = lines[i].strip()
            i += 1
            if not s:
                continue
            s = re.sub(r"^[\d:]{3,9}\s*[-–—]?\s*", "", s)
            if s:
                names.append(s)

    out.append("ГЛАВЫ:")
    for idx, (t, default_name) in enumerate(chapters):
        out.append(f"{format_time_hms(t)} {names[idx] if idx < len(names) else default_name}")
    return "\n".join(out).rstrip()

def make_description(video_path, out_path, api_key, log=None, max_chars=30000):
    """Расшифровка -> DeepSeek -> текстовый файл с заголовком, описанием, тегами и главами."""
    rows = transcribe_segments(video_path, log)
    if not rows:
        raise RuntimeError("Речь не распознана — отправлять нечего")

    text = "\n".join(f"[{format_time_hms(t)}] {x}" for t, x in rows)
    if len(text) > max_chars:
        text = text[:max_chars - 8000] + "\n...(середина пропущена)...\n" + text[-7000:]

    chapters = read_chapters(video_path)
    if chapters:
        fixed = "\n".join(f"{format_time_hms(t)} {name}" for t, name in chapters)
        if log:
            log(f"[ОПИСАНИЕ] Взял готовые главы из файла таймкодов: {len(chapters)} шт.")
        chapter_task = (
            "Главы уже известны — это реальные границы эпизодов готового ролика. "
            "Придумай каждой короткое название (2-5 слов), порядок и количество не меняй.\n"
            "Вот они:\n" + fixed + "\n"
        )
    else:
        chapter_task = (
            "- Главы: 5-10 штук, формат «ММ:СС Название», первая строго «00:00 ...». "
            "Таймкоды бери из расшифровки, не выдумывай.\n"
        )

    prompt = (
        "Ниже расшифровка готового YouTube-ролика с нарезками стримов Dota 2 "
        "(стример Daxak, канал «Бородка Дахака»).\n\n"
        "Сделай оформление для публикации.\n\n"
        "Требования:\n"
        "- Заголовок: до 90 символов, по-русски, цепляющий, но без обмана.\n"
        "- Описание: 2-3 абзаца по 2-3 предложения, живым языком, без воды и без выдумок.\n"
        "- Хэштеги: 12-15 штук через пробел, по теме Dota 2.\n"
        + chapter_task +
        "\nОтвет строго в таком виде, без пояснений:\n"
        "ЗАГОЛОВОК: ...\n"
        "ОПИСАНИЕ:\n...\n"
        "ХЭШТЕГИ: ...\n"
        "ГЛАВЫ:\n00:00 ...\n\n"
        f"РАСШИФРОВКА:\n{text}"
    )

    if log:
        log("[ОПИСАНИЕ] Отправляю в DeepSeek...")
    answer = deepseek_chat(prompt, api_key,
                           system="Ты редактор YouTube-канала с нарезками Dota 2. Пишешь по-русски, коротко и по делу.")
    answer = (answer or "").strip()
    if not answer:
        raise RuntimeError("DeepSeek вернул пустой ответ")

    if chapters:
        answer = _fix_chapters(answer, chapters)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(answer + "\n")
    if log:
        log(f"[ОПИСАНИЕ] Готово: {os.path.basename(out_path)}")
    return answer

# ==================== ГРАФИЧЕСКИЙ ИНТЕРФЕЙС ====================
class UnifiedDotaStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dota 2 Studio | Монтаж матчей + Реакции/Кейсы + Shorts + Цензор")

        # Окно подстраивается под экран (в т.ч. при масштабировании Windows)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        win_w = max(760, min(940, sw - 40))
        win_h = max(520, min(920, sh - 80))
        pos_x = max(0, (sw - win_w) // 2)
        pos_y = max(0, (sh - win_h) // 3)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(max(700, min(860, sw - 120)), max(420, min(600, sh - 160)))

        self.ffmpeg_bin = find_ffmpeg()
        self.temp_dir_var = tk.StringVar(value=TEMP_DIR)
        self.react_fragment_rows = []

        self.setup_ui()
        self.load_settings()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Вкладки занимают всё свободное место, консоль — фиксированную высоту снизу
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(6, 4))

        self.tab_cutter = self._add_scrollable_tab(" 🎬 1. Монтаж ")
        self.tab_react = self._add_scrollable_tab(" 📺 2. Реакции ")
        self.tab_clips = self._add_scrollable_tab(" 🔥 3. Клипы ")
        self.tab_shorts = self._add_scrollable_tab(" 📱 4. Shorts ")
        self.tab_censor = self._add_scrollable_tab(" 🔇 5. Цензура ")
        self.tab_overlay = self._add_scrollable_tab(" 🖼 6. Баннер ")
        self.tab_ads = self._add_scrollable_tab(" 📢 7. Реклама ")
        self.tab_desc = self._add_scrollable_tab(" 🤖 8. Описание ")

        self.build_cutter_tab()
        self.build_react_tab()
        self.build_clips_tab()
        self.build_shorts_tab()
        self.build_censor_tab()
        self.build_overlay_tab()
        self.build_ad_tab()
        self.build_desc_tab()

        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

        bottom_frame = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        bottom_frame.pack(fill=tk.X, expand=False)

        temp_row = ttk.Frame(bottom_frame)
        temp_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(temp_row, text="Временные файлы:").pack(side=tk.LEFT)
        self.temp_dir_entry = ttk.Entry(temp_row, textvariable=self.temp_dir_var)
        self.temp_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(temp_row, text="Выбрать...", command=self.browse_temp_dir).pack(side=tk.LEFT)
        ttk.Button(temp_row, text="Открыть", command=self.open_temp_dir).pack(side=tk.LEFT, padx=(5, 0))

        self.progress_bar = ttk.Progressbar(bottom_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 4))

        log_frame = ttk.LabelFrame(bottom_frame, text=" Консоль событий ", padding="4")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, bg="#1e1e1e", fg="#00ff66",
            font=("Consolas", 10), insertbackground="white", height=7
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log("[СИСТЕМА] Программа готова к работе.")
        self.log(f"[СИСТЕМА] Временные файлы: {TEMP_DIR}")
        threading.Thread(target=self._report_gpu_mode, daemon=True).start()

    def _add_scrollable_tab(self, text):
        """Вкладка с вертикальной прокруткой, чтобы всё влезало на маленьком экране."""
        outer = ttk.Frame(self.notebook)
        bg = ttk.Style().lookup("TFrame", "background") or "#dcdad5"
        canvas = tk.Canvas(outer, highlightthickness=0, bd=0, background=bg)
        vbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        inner = ttk.Frame(canvas, padding="8")
        inner._scroll_canvas = canvas
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        self.notebook.add(outer, text=text)
        return inner

    def _on_mousewheel(self, event):
        starts = []
        w = getattr(event, "widget", None)
        if w is not None:
            starts.append(w)
        try:
            w2 = self.root.winfo_containing(event.x_root, event.y_root)
            if w2 is not None:
                starts.append(w2)
        except Exception:
            pass
        for start in starts:
            w = start
            while w is not None:
                canvas = getattr(w, "_scroll_canvas", None)
                if canvas is not None:
                    canvas.yview_scroll(-3 if event.delta > 0 else 3, "units")
                    return "break"
                w = getattr(w, "master", None)

    def browse_temp_dir(self):
        cur = self.temp_dir_var.get().strip() or TEMP_DIR
        d = filedialog.askdirectory(title="Папка для временных файлов",
                                    initialdir=cur if os.path.isdir(cur) else None)
        if d:
            self.apply_temp_dir(d)

    def apply_temp_dir(self, d):
        global TEMP_DIR
        d = os.path.normpath(d)
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, f".probe_{os.getpid()}")
            with open(probe, "w", encoding="utf-8") as pf:
                pf.write("ok")
            os.remove(probe)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось использовать папку:\n{d}\n\n{e}")
            return
        TEMP_DIR = d
        tempfile.tempdir = d
        self.temp_dir_var.set(d)
        self.save_settings()
        self.log(f"[СИСТЕМА] Временные файлы теперь в: {d}")

    def open_temp_dir(self):
        d = self.temp_dir_var.get().strip() or TEMP_DIR
        try:
            os.makedirs(d, exist_ok=True)
            os.startfile(d)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _report_gpu_mode(self):
        if use_qsv(self.ffmpeg_bin):
            self.log("[СИСТЕМА] Кодирование: Intel Quick Sync (GPU) — CPU разгружен.")
        else:
            self.log("[СИСТЕМА] Кодирование: CPU (libx264). QSV недоступен или отключён (use_gpu_qsv=false).")

    # ---------------- СОХРАНЕНИЕ И ВОССТАНОВЛЕНИЕ НАСТРОЕК ----------------
    def save_settings(self):
        react_frags = []
        for r in getattr(self, 'react_fragment_rows', []):
            react_frags.append({
                "start": r['start_entry'].get().strip(),
                "end": r['end_entry'].get().strip()
            })

        def _get(attr, default=""):
            w = getattr(self, attr, None)
            if w is None:
                return default
            if hasattr(w, 'get'):
                return w.get()
            return default

        config = {
            "temp_dir": self.temp_dir_var.get() if hasattr(self, "temp_dir_var") else TEMP_DIR,
            "use_gpu_qsv": bool(_config_value("use_gpu_qsv", True)),
            "cut_input": _get("cut_input_entry"),
            "cut_chat_vid": _get("cut_chat_vid_entry"),
            "cut_output": _get("cut_output_entry"),
            "use_chat": _get("use_chat_var", True),
            "chat_x": _get("cut_chat_x_entry", "35"),
            "chat_y": _get("cut_chat_y_entry", "240"),
            "chat_w": _get("cut_chat_w_entry", "340"),
            "chat_h": _get("cut_chat_h_entry", "550"),
            "use_overlay": _get("use_overlay_var", True),
            "overlay_file": _get("overlay_file_entry"),
            "legacy_cutter_mode": _get("legacy_cutter_mode_var", True),
            "cut_x": _get("cut_x_entry", "1580"),
            "cut_y": _get("cut_y_entry", "30"),
            "cut_w": _get("cut_w_entry", "290"),
            "cut_h": _get("cut_h_entry", "70"),
            "swears_file": _get("swears_file_entry"),
            "series_mode": _get("series_mode_var", "series"),
            "start_time": _get("start_time_entry", "00:00:00"),
            "end_time": _get("end_time_entry", ""),
            "m1_s": _get("m1_s", "00:25:00"), "m1_e": _get("m1_e", "01:15:00"),
            "m2_s": _get("m2_s", "01:40:00"), "m2_e": _get("m2_e", "02:35:00"),
            "m3_s": _get("m3_s", "03:00:00"), "m3_e": _get("m3_e", "03:55:00"),
            "m4_s": _get("m4_s", "04:20:00"), "m4_e": _get("m4_e", "05:15:00"),
            "m5_s": _get("m5_s", "05:40:00"), "m5_e": _get("m5_e", "06:35:00"),
            "draft_min": [sp.get() for sp in getattr(self, "draft_min_spins", [])],
            "cut_duration": _get("cut_slider", 120),
            "keep_teaser": _get("keep_teaser_var", True),
            "keep_intro": _get("keep_intro_var", True),
            "keep_outro": _get("keep_outro_var", True),
            "keep_kills": _get("keep_kills_var", True),
            "smooth_cuts": _get("smooth_cuts_var", True),
            "censor_after_cut": _get("censor_after_cut_var", True),
            "audio_delay": _get("audio_delay_spin", "0"),
            "tg_intro": _get("tg_intro_var", False),
            "tg_avatar": _get("tg_avatar_entry"),
            "tg_link": _get("tg_link_entry"),
            "ad_cut_on": _get("ad_cut_var", False),
            "ad_cut_file": _get("ad_cut_entry"),
            "ad_cut_time": _get("ad_cut_time_entry", "01:00"),
            "ad_src": _get("ad_src_entry"),
            "ad_file": _get("ad_file_entry"),
            "ad_out": _get("ad_out_entry"),
            "ad_time": _get("ad_time_entry", "01:00"),
            "desc_video": _get("desc_video_entry"),
            "desc_out": _get("desc_out_entry"),
            "deepseek_key": _get("deepseek_key_entry"),

            # Реакции
            "react_input": _get("react_input_entry"),
            "react_chat_vid": _get("react_chat_vid_entry"),
            "react_output": _get("react_output_entry"),
            "react_use_chat": _get("react_use_chat_var", True),
            "react_chat_x": _get("react_chat_x_entry", "35"),
            "react_chat_y": _get("react_chat_y_entry", "240"),
            "react_chat_w": _get("react_chat_w_entry", "340"),
            "react_chat_h": _get("react_chat_h_entry", "550"),
            "react_use_overlay": _get("react_use_overlay_var", True),
            "react_overlay_file": _get("react_overlay_file_entry"),
            "react_x": _get("react_x_entry", "1580"),
            "react_y": _get("react_y_entry", "30"),
            "react_w": _get("react_w_entry", "290"),
            "react_h": _get("react_h_entry", "70"),
            "react_fragments": react_frags,
            "react_mode": _get("react_mode_combo", "🎬 Реакция на видео (100% сохранение видео без пропусков)"),
            "react_cut_donations": _get("react_cut_donations_var", True),
            "react_max_pause": _get("react_max_pause_spin", "10"),

            # Клипы
            "clips_channel": _get("clips_channel_entry", "daxak"),
            "clips_period": _get("clips_period_combo", "За месяц (30 дней)"),
            "clips_count": _get("clips_count_spin", "15"),
            "clips_out": _get("clips_out_entry"),
            "clips_titles": _get("clips_titles_var", True),
            "clips_fade": _get("clips_fade_var", True),
            "clips_censor": _get("clips_censor_var", True),
            "clips_swears": _get("clips_swears_entry"),
            "clips_overlay": _get("clips_overlay_var", True),
            "clips_png": _get("clips_png_entry"),

            # Shorts
            "shorts_input": _get("shorts_input_entry"),
            "shorts_out_dir": _get("shorts_out_dir_entry"),
            "shorts_cam_x": _get("shorts_cam_x", "1520"),
            "shorts_cam_y": _get("shorts_cam_y", "720"),
            "shorts_cam_w": _get("shorts_cam_w", "400"),
            "shorts_cam_h": _get("shorts_cam_h", "300"),
            "shorts_count": _get("shorts_count_spin", "5"),
            "shorts_duration": _get("shorts_dur_slider", 30),
            "shorts_censor": _get("shorts_censor_var", True),
            "shorts_swears": _get("shorts_swears_entry"),
            "shorts_ad": _get("shorts_ad_var", False),
            "shorts_ad_file": _get("shorts_ad_entry"),

            # Цензор / Оверлей
            "censor_input": _get("censor_input_entry"),
            "censor_output": _get("censor_output_entry"),
            "censor_swears": _get("censor_swears_entry"),
            "single_vid": _get("single_vid_entry"),
            "single_out": _get("single_out_entry"),
            "single_overlay_file": _get("single_overlay_file_entry"),
            "single_x": _get("single_x_entry", "1580"),
            "single_y": _get("single_y_entry", "30"),
            "single_w": _get("single_w_entry", "290"),
            "single_h": _get("single_h_entry", "70"),
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.log(f"[СИСТЕМА] Ошибка при сохранении настроек: {e}")

    def load_settings(self):
        def _set_entry(attr, val):
            w = getattr(self, attr, None)
            if w is not None and val is not None:
                w.delete(0, tk.END)
                w.insert(0, str(val))

        if not os.path.exists(CONFIG_FILE):
            def_swears = find_default_swears_file()
            if def_swears:
                for a in ["swears_file_entry", "censor_swears_entry", "clips_swears_entry", "shorts_swears_entry"]:
                    _set_entry(a, def_swears)
            def_ovl = find_default_overlay_image()
            if def_ovl:
                for a in ["overlay_file_entry", "react_overlay_file_entry", "clips_png_entry", "single_overlay_file_entry"]:
                    _set_entry(a, def_ovl)
            if not self.react_fragment_rows:
                self.add_react_fragment("00:00:00", "")
            return

        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                c = json.load(f)

            _set_entry("cut_input_entry", c.get("cut_input", ""))
            _set_entry("cut_chat_vid_entry", c.get("cut_chat_vid", ""))
            _set_entry("cut_output_entry", c.get("cut_output", ""))
            if hasattr(self, 'use_chat_var'): self.use_chat_var.set(c.get("use_chat", True))
            _set_entry("cut_chat_x_entry", c.get("chat_x", "35"))
            _set_entry("cut_chat_y_entry", c.get("chat_y", "240"))
            _set_entry("cut_chat_w_entry", c.get("chat_w", "340"))
            _set_entry("cut_chat_h_entry", c.get("chat_h", "550"))
            if hasattr(self, 'use_overlay_var'): self.use_overlay_var.set(c.get("use_overlay", True))
            _set_entry("overlay_file_entry", c.get("overlay_file", find_default_overlay_image()))
            _set_entry("cut_x_entry", c.get("cut_x", "1580"))
            _set_entry("cut_y_entry", c.get("cut_y", "30"))
            _set_entry("cut_w_entry", c.get("cut_w", "290"))
            _set_entry("cut_h_entry", c.get("cut_h", "70"))
            _set_entry("swears_file_entry", c.get("swears_file", find_default_swears_file()))
            if hasattr(self, 'series_mode_var'):
                self.series_mode_var.set(c.get("series_mode", "series"))
                self.toggle_mode_view()
            _set_entry("start_time_entry", c.get("start_time", "00:00:00"))
            _set_entry("end_time_entry", c.get("end_time", ""))
            _set_entry("m1_s", c.get("m1_s", "00:25:00")); _set_entry("m1_e", c.get("m1_e", "01:15:00"))
            _set_entry("m2_s", c.get("m2_s", "01:40:00")); _set_entry("m2_e", c.get("m2_e", "02:35:00"))
            _set_entry("m3_s", c.get("m3_s", "03:00:00")); _set_entry("m3_e", c.get("m3_e", "03:55:00"))
            _set_entry("m4_s", c.get("m4_s", "04:20:00")); _set_entry("m4_e", c.get("m4_e", "05:15:00"))
            _set_entry("m5_s", c.get("m5_s", "05:40:00")); _set_entry("m5_e", c.get("m5_e", "06:35:00"))
            _draft_mins = c.get("draft_min")
            if _draft_mins and hasattr(self, "draft_min_spins"):
                for _sp, _v in zip(self.draft_min_spins, _draft_mins):
                    try: _sp.set(str(_v))
                    except Exception: pass
            
            dur = c.get("cut_duration", 120)
            if hasattr(self, 'cut_slider'):
                self.cut_slider.set(dur)
                self.on_cut_slider(dur)
            
            if hasattr(self, 'keep_teaser_var'): self.keep_teaser_var.set(c.get("keep_teaser", True))
            if hasattr(self, 'keep_intro_var'): self.keep_intro_var.set(c.get("keep_intro", True))
            if hasattr(self, 'keep_outro_var'): self.keep_outro_var.set(c.get("keep_outro", True))
            if hasattr(self, 'keep_kills_var'): self.keep_kills_var.set(c.get("keep_kills", True))
            if hasattr(self, 'legacy_cutter_mode_var'): self.legacy_cutter_mode_var.set(c.get("legacy_cutter_mode", True))
            if hasattr(self, 'smooth_cuts_var'): self.smooth_cuts_var.set(c.get("smooth_cuts", True))
            if hasattr(self, 'censor_after_cut_var'): self.censor_after_cut_var.set(c.get("censor_after_cut", True))
            _set_entry("audio_delay_spin", c.get("audio_delay", "0"))
            if hasattr(self, 'tg_intro_var'): self.tg_intro_var.set(c.get("tg_intro", False))
            _set_entry("tg_avatar_entry", c.get("tg_avatar", ""))
            _set_entry("tg_link_entry", c.get("tg_link", "t.me/borodkadaxaka"))
            if hasattr(self, 'ad_cut_var'): self.ad_cut_var.set(c.get("ad_cut_on", False))
            _set_entry("ad_cut_entry", c.get("ad_cut_file", ""))
            _set_entry("ad_cut_time_entry", c.get("ad_cut_time", "01:00"))
            _set_entry("ad_src_entry", c.get("ad_src", ""))
            _set_entry("ad_file_entry", c.get("ad_file", ""))
            _set_entry("ad_out_entry", c.get("ad_out", ""))
            _set_entry("ad_time_entry", c.get("ad_time", "01:00"))
            _set_entry("desc_video_entry", c.get("desc_video", ""))
            _set_entry("desc_out_entry", c.get("desc_out", ""))
            _set_entry("deepseek_key_entry", c.get("deepseek_key", ""))

            _t = str(c.get("temp_dir", "") or "").strip()
            if _t and os.path.isdir(_t) and hasattr(self, "temp_dir_var"):
                self.temp_dir_var.set(os.path.normpath(_t))

            # Реакции
            _set_entry("react_input_entry", c.get("react_input", ""))
            _set_entry("react_chat_vid_entry", c.get("react_chat_vid", ""))
            _set_entry("react_output_entry", c.get("react_output", ""))
            if hasattr(self, 'react_use_chat_var'): self.react_use_chat_var.set(c.get("react_use_chat", True))
            _set_entry("react_chat_x_entry", c.get("chat_x", "35"))
            _set_entry("react_chat_y_entry", c.get("chat_y", "240"))
            _set_entry("react_chat_w_entry", c.get("chat_w", "340"))
            _set_entry("react_chat_h_entry", c.get("chat_h", "550"))
            if hasattr(self, 'react_use_overlay_var'): self.react_use_overlay_var.set(c.get("react_use_overlay", True))
            _set_entry("react_overlay_file_entry", c.get("react_overlay_file", find_default_overlay_image()))
            _set_entry("react_x_entry", c.get("react_x", "1580"))
            _set_entry("react_y_entry", c.get("react_y", "30"))
            _set_entry("react_w_entry", c.get("react_w", "290"))
            _set_entry("react_h_entry", c.get("react_h", "70"))
            if hasattr(self, 'react_mode_combo') and c.get("react_mode"):
                self.react_mode_combo.set(c["react_mode"])
            if hasattr(self, 'react_cut_donations_var'):
                self.react_cut_donations_var.set(c.get("react_cut_donations", True))
            _set_entry("react_max_pause_spin", c.get("react_max_pause", "10"))

            for row in list(self.react_fragment_rows):
                self.remove_react_fragment(row)

            saved_frags = c.get("react_fragments", [])
            if saved_frags:
                for item in saved_frags:
                    self.add_react_fragment(item.get("start", "00:00:00"), item.get("end", ""))
            else:
                self.add_react_fragment("00:00:00", "")

            # Клипы
            _set_entry("clips_channel_entry", c.get("clips_channel", "daxak"))
            if hasattr(self, 'clips_period_combo') and c.get("clips_period"):
                self.clips_period_combo.set(c["clips_period"])
            _set_entry("clips_count_spin", c.get("clips_count", "15"))
            _set_entry("clips_out_entry", c.get("clips_out", os.path.abspath("DAXAK_TOP_CLIPS_MONTH.mp4")))
            if hasattr(self, 'clips_titles_var'): self.clips_titles_var.set(c.get("clips_titles", True))
            if hasattr(self, 'clips_fade_var'): self.clips_fade_var.set(c.get("clips_fade", True))
            if hasattr(self, 'clips_censor_var'): self.clips_censor_var.set(c.get("clips_censor", True))
            _set_entry("clips_swears_entry", c.get("clips_swears", find_default_swears_file()))
            if hasattr(self, 'clips_overlay_var'): self.clips_overlay_var.set(c.get("clips_overlay", True))
            _set_entry("clips_png_entry", c.get("clips_png", find_default_overlay_image()))

            # Shorts
            _set_entry("shorts_input_entry", c.get("shorts_input", ""))
            _set_entry("shorts_out_dir_entry", c.get("shorts_out_dir", os.path.abspath("Shorts_Result")))
            _set_entry("shorts_cam_x", c.get("shorts_cam_x", "1520"))
            _set_entry("shorts_cam_y", c.get("shorts_cam_y", "720"))
            _set_entry("shorts_cam_w", c.get("shorts_cam_w", "400"))
            _set_entry("shorts_cam_h", c.get("shorts_cam_h", "300"))
            _set_entry("shorts_count_spin", c.get("shorts_count", "5"))
            s_dur = c.get("shorts_duration", 30)
            if hasattr(self, 'shorts_dur_slider'):
                self.shorts_dur_slider.set(s_dur)
                self.on_shorts_slider(s_dur)
            if hasattr(self, 'shorts_censor_var'): self.shorts_censor_var.set(c.get("shorts_censor", True))
            _set_entry("shorts_swears_entry", c.get("shorts_swears", find_default_swears_file()))
            if hasattr(self, 'shorts_ad_var'): self.shorts_ad_var.set(c.get("shorts_ad", False))
            _set_entry("shorts_ad_entry", c.get("shorts_ad_file", ""))

            # Цензор / Оверлей
            _set_entry("censor_input_entry", c.get("censor_input", ""))
            _set_entry("censor_output_entry", c.get("censor_output", ""))
            _set_entry("censor_swears_entry", c.get("censor_swears", find_default_swears_file()))

            _set_entry("single_vid_entry", c.get("single_vid", ""))
            _set_entry("single_out_entry", c.get("single_out", ""))
            _set_entry("single_overlay_file_entry", c.get("single_overlay_file", find_default_overlay_image()))
            _set_entry("single_x_entry", c.get("single_x", "1580"))
            _set_entry("single_y_entry", c.get("single_y", "30"))
            _set_entry("single_w_entry", c.get("single_w", "290"))
            _set_entry("single_h_entry", c.get("single_h", "70"))

            self.log("[СИСТЕМА] Настройки успешно загружены.")
        except Exception as e:
            self.log(f"[СИСТЕМА] Ошибка чтения studio_config.json: {e}")

    def on_close(self):
        self.save_settings()
        self.root.destroy()

    # ---------------- ВКЛАДКА 1: МОНТАЖ МАТЧА ----------------
    def build_cutter_tab(self):
        f_box = ttk.LabelFrame(self.tab_cutter, text=" 1. Исходные файлы ", padding="6")
        f_box.pack(fill=tk.X, pady=3)

        ttk.Label(f_box, text="Исходный VOD:").grid(row=0, column=0, sticky=tk.W)
        self.cut_input_entry = ttk.Entry(f_box, width=58)
        self.cut_input_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(f_box, text="Обзор...", command=self.browse_cut_input).grid(row=0, column=2)

        ttk.Label(f_box, text="Видео чата (опционально):").grid(row=1, column=0, sticky=tk.W)
        self.cut_chat_vid_entry = ttk.Entry(f_box, width=58)
        self.cut_chat_vid_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(f_box, text="Обзор...", command=self.browse_cut_chat_vid).grid(row=1, column=2)

        ttk.Label(f_box, text="Сохранить как:").grid(row=2, column=0, sticky=tk.W)
        self.cut_output_entry = ttk.Entry(f_box, width=58)
        self.cut_output_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(f_box, text="Обзор...", command=self.browse_cut_output).grid(row=2, column=2)
        f_box.columnconfigure(1, weight=1)

        chat_box = ttk.LabelFrame(self.tab_cutter, text=" 2. Позиция чата Twitch на экране ", padding="6")
        chat_box.pack(fill=tk.X, pady=3)

        ch_r = ttk.Frame(chat_box)
        ch_r.pack(fill=tk.X, pady=2)
        self.use_chat_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ch_r, text="Накладывать чат", variable=self.use_chat_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(ch_r, text="🔍 Указать положение на кадре (мышкой)", command=lambda: self.open_interactive_chat_selector(is_react=False)).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(ch_r, text="X:").pack(side=tk.LEFT)
        self.cut_chat_x_entry = ttk.Entry(ch_r, width=5); self.cut_chat_x_entry.insert(0, "35"); self.cut_chat_x_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(ch_r, text="Y:").pack(side=tk.LEFT)
        self.cut_chat_y_entry = ttk.Entry(ch_r, width=5); self.cut_chat_y_entry.insert(0, "240"); self.cut_chat_y_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(ch_r, text="W:").pack(side=tk.LEFT)
        self.cut_chat_w_entry = ttk.Entry(ch_r, width=5); self.cut_chat_w_entry.insert(0, "340"); self.cut_chat_w_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(ch_r, text="H:").pack(side=tk.LEFT)
        self.cut_chat_h_entry = ttk.Entry(ch_r, width=5); self.cut_chat_h_entry.insert(0, "550"); self.cut_chat_h_entry.pack(side=tk.LEFT, padx=(2, 10))

        banner_box = ttk.LabelFrame(self.tab_cutter, text=" 3. Закрытие баннера FONBET картинкой (1.5x) ", padding="6")
        banner_box.pack(fill=tk.X, pady=3)

        b_top = ttk.Frame(banner_box)
        b_top.pack(fill=tk.X, pady=2)
        self.use_overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(b_top, text="Закрывать баннер (увеличение 1.5x)", variable=self.use_overlay_var).pack(side=tk.LEFT)

        b_file = ttk.Frame(banner_box)
        b_file.pack(fill=tk.X, pady=2)
        ttk.Label(b_file, text="Файл PNG:").pack(side=tk.LEFT)
        self.overlay_file_entry = ttk.Entry(b_file, width=48)
        self.overlay_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(b_file, text="Обзор...", command=self.browse_overlay_file).pack(side=tk.LEFT)

        b_detect_row = ttk.Frame(banner_box)
        b_detect_row.pack(fill=tk.X, pady=3)
        ttk.Button(b_detect_row, text="🔍 Указать баннер мышкой", command=lambda: self.open_interactive_detector(self.cut_input_entry.get())).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(b_detect_row, text="X:").pack(side=tk.LEFT)
        self.cut_x_entry = ttk.Entry(b_detect_row, width=5); self.cut_x_entry.insert(0, "1580"); self.cut_x_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(b_detect_row, text="Y:").pack(side=tk.LEFT)
        self.cut_y_entry = ttk.Entry(b_detect_row, width=5); self.cut_y_entry.insert(0, "30"); self.cut_y_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(b_detect_row, text="W:").pack(side=tk.LEFT)
        self.cut_w_entry = ttk.Entry(b_detect_row, width=5); self.cut_w_entry.insert(0, "290"); self.cut_w_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(b_detect_row, text="H:").pack(side=tk.LEFT)
        self.cut_h_entry = ttk.Entry(b_detect_row, width=5); self.cut_h_entry.insert(0, "70"); self.cut_h_entry.pack(side=tk.LEFT, padx=(2, 10))

        dict_box = ttk.LabelFrame(self.tab_cutter, text=" 4. Словарь матов ru_curse_words.txt ", padding="6")
        dict_box.pack(fill=tk.X, pady=3)
        ttk.Label(dict_box, text="Словарь (.txt):").grid(row=0, column=0, sticky=tk.W)
        self.swears_file_entry = ttk.Entry(dict_box, width=58)
        self.swears_file_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(dict_box, text="Обзор...", command=self.browse_swears_file).grid(row=0, column=2)
        dict_box.columnconfigure(1, weight=1)

        bounds_box = ttk.LabelFrame(self.tab_cutter, text=" 5. Режим матча / Финал турнира (Таймкоды карт) ", padding="6")
        bounds_box.pack(fill=tk.X, pady=3)

        mode_row = ttk.Frame(bounds_box)
        mode_row.pack(fill=tk.X, pady=2)
        self.series_mode_var = tk.StringVar(value="series")
        ttk.Radiobutton(mode_row, text="🏆 Серия карт (Финал турнира из 2–4 карт)", value="series", variable=self.series_mode_var, command=self.toggle_mode_view).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(mode_row, text="Одиночный матч (1 игра)", value="single", variable=self.series_mode_var, command=self.toggle_mode_view).pack(side=tk.LEFT)

        self.single_frame = ttk.Frame(bounds_box)
        ttk.Label(self.single_frame, text="Начало драфта:").pack(side=tk.LEFT)
        self.start_time_entry = ttk.Entry(self.single_frame, width=9); self.start_time_entry.insert(0, "00:00:00"); self.start_time_entry.pack(side=tk.LEFT, padx=(5, 20))
        ttk.Label(self.single_frame, text="Падение трона:").pack(side=tk.LEFT)
        self.end_time_entry = ttk.Entry(self.single_frame, width=9); self.end_time_entry.pack(side=tk.LEFT, padx=(5, 10))

        self.series_frame = ttk.Frame(bounds_box)
        self.series_frame.pack(fill=tk.X, pady=4)

        r_m1 = ttk.Frame(self.series_frame); r_m1.pack(fill=tk.X, pady=2)
        ttk.Label(r_m1, text="Карта 1:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.m1_s = ttk.Entry(r_m1, width=8); self.m1_s.insert(0, "00:25:00"); self.m1_s.pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(r_m1, text="—").pack(side=tk.LEFT)
        self.m1_e = ttk.Entry(r_m1, width=8); self.m1_e.insert(0, "01:15:00"); self.m1_e.pack(side=tk.LEFT, padx=(2, 20))
        ttk.Label(r_m1, text="Карта 2:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.m2_s = ttk.Entry(r_m1, width=8); self.m2_s.insert(0, "01:40:00"); self.m2_s.pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(r_m1, text="—").pack(side=tk.LEFT)
        self.m2_e = ttk.Entry(r_m1, width=8); self.m2_e.insert(0, "02:35:00"); self.m2_e.pack(side=tk.LEFT, padx=2)

        r_m2 = ttk.Frame(self.series_frame); r_m2.pack(fill=tk.X, pady=2)
        ttk.Label(r_m2, text="Карта 3:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.m3_s = ttk.Entry(r_m2, width=8); self.m3_s.insert(0, "03:00:00"); self.m3_s.pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(r_m2, text="—").pack(side=tk.LEFT)
        self.m3_e = ttk.Entry(r_m2, width=8); self.m3_e.insert(0, "03:55:00"); self.m3_e.pack(side=tk.LEFT, padx=(2, 20))
        ttk.Label(r_m2, text="Карта 4:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.m4_s = ttk.Entry(r_m2, width=8); self.m4_s.insert(0, "04:20:00"); self.m4_s.pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(r_m2, text="—").pack(side=tk.LEFT)
        self.m4_e = ttk.Entry(r_m2, width=8); self.m4_e.insert(0, "05:15:00"); self.m4_e.pack(side=tk.LEFT, padx=2)
        ttk.Label(r_m2, text="Карта 5:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.m5_s = ttk.Entry(r_m2, width=8); self.m5_s.insert(0, "05:40:00"); self.m5_s.pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(r_m2, text="—").pack(side=tk.LEFT)
        self.m5_e = ttk.Entry(r_m2, width=8); self.m5_e.insert(0, "06:35:00"); self.m5_e.pack(side=tk.LEFT, padx=2)

        draft_row = ttk.Frame(self.series_frame); draft_row.pack(fill=tk.X, pady=2)
        ttk.Label(draft_row, text="Драфт (пики), мин:", font=("Arial", 8, "bold")).pack(side=tk.LEFT)
        self.draft_min_spins = []
        for _i in range(5):
            ttk.Label(draft_row, text=f"К{_i + 1}", font=("Arial", 8)).pack(side=tk.LEFT, padx=(6, 0))
            _sp = ttk.Spinbox(draft_row, from_=0.5, to=30.0, increment=0.5, width=4)
            _sp.set("1.5")
            _sp.pack(side=tk.LEFT)
            self.draft_min_spins.append(_sp)

        ttk.Label(self.series_frame, text="* В каждую карту гарантированно попадут ДРАФТ (сколько указано в «Драфт, мин») и ТРОН (120с)", font=("Arial", 8, "italic"), foreground="#00aa00").pack(anchor=tk.W, pady=2)

        story_box = ttk.LabelFrame(self.tab_cutter, text=" 6. Хронометраж и Синхронизация звука ", padding="6")
        story_box.pack(fill=tk.X, pady=3)
        t_sub = ttk.Frame(story_box); t_sub.pack(fill=tk.X)
        self.target_label = ttk.Label(t_sub, text="Длина ролика: 120 минут (2 часа)", font=("Arial", 10, "bold"))
        self.target_label.pack(side=tk.LEFT)

        btn_box = ttk.Frame(t_sub); btn_box.pack(side=tk.RIGHT)
        for m in [30, 60, 90, 120, 150]:
            lbl_b = f"{m} мин" if m < 60 else f"{m//60}ч" if m % 60 == 0 else f"{m}м"
            ttk.Button(btn_box, text=lbl_b, width=5, command=lambda v=m: self.set_cut_duration(v)).pack(side=tk.LEFT, padx=2)

        self.cut_slider = ttk.Scale(story_box, from_=20, to=180, orient=tk.HORIZONTAL, command=self.on_cut_slider)
        self.cut_slider.set(120)
        self.cut_slider.pack(fill=tk.X, pady=3)

        opts_box = ttk.Frame(story_box); opts_box.pack(fill=tk.X, pady=2)
        self.keep_teaser_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts_box, text="⚡ YouTube-Хук (3 момента)", variable=self.keep_teaser_var).pack(side=tk.LEFT, padx=(0, 15))
        self.keep_intro_var = tk.BooleanVar(value=True); ttk.Checkbutton(opts_box, text="Все пики", variable=self.keep_intro_var).pack(side=tk.LEFT, padx=(0, 15))
        self.keep_outro_var = tk.BooleanVar(value=True); ttk.Checkbutton(opts_box, text="Все троны", variable=self.keep_outro_var).pack(side=tk.LEFT, padx=(0, 15))
        self.keep_kills_var = tk.BooleanVar(value=True); ttk.Checkbutton(opts_box, text="⚔️ Все килы (CV-счёт)", variable=self.keep_kills_var).pack(side=tk.LEFT, padx=(0, 15))
        self.legacy_cutter_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts_box, text="📼 Старый режим (максимум килов, без жесткого лимита времени)", variable=self.legacy_cutter_mode_var).pack(side=tk.LEFT, padx=(0, 15))
        self.smooth_cuts_var = tk.BooleanVar(value=True); ttk.Checkbutton(opts_box, text="✨ Плавные склейки (Fade)", variable=self.smooth_cuts_var).pack(side=tk.LEFT, padx=(0, 15))
        self.censor_after_cut_var = tk.BooleanVar(value=True); ttk.Checkbutton(opts_box, text="🔇 Mute", variable=self.censor_after_cut_var).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(opts_box, text="Сдвиг звука (мс):").pack(side=tk.LEFT)
        self.audio_delay_spin = ttk.Spinbox(opts_box, from_=-1000, to=1000, increment=50, width=5)
        self.audio_delay_spin.set(0)
        self.audio_delay_spin.pack(side=tk.LEFT, padx=(2, 0))

        tg_box = ttk.LabelFrame(self.tab_cutter, text=" 7. Вставка-заставка ТГК (4 сек, после хука) ", padding="6")
        tg_box.pack(fill=tk.X, pady=3)

        tg_top = ttk.Frame(tg_box); tg_top.pack(fill=tk.X, pady=2)
        self.tg_intro_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tg_top, text="Вставлять заставку с логотипом", variable=self.tg_intro_var).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(tg_top, text="Ссылка:").pack(side=tk.LEFT)
        self.tg_link_entry = ttk.Entry(tg_top, width=28); self.tg_link_entry.insert(0, "t.me/borodkadaxaka"); self.tg_link_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(tg_top, text="👁 Превью (4 сек)", command=self.preview_tg_intro).pack(side=tk.LEFT)

        tg_row = ttk.Frame(tg_box); tg_row.pack(fill=tk.X, pady=2)
        ttk.Label(tg_row, text="Аватарка (PNG):").pack(side=tk.LEFT)
        self.tg_avatar_entry = ttk.Entry(tg_row, width=48); self.tg_avatar_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(tg_row, text="Обзор...", command=self.browse_tg_avatar).pack(side=tk.LEFT)

        adv_box = ttk.LabelFrame(self.tab_cutter, text=" 8. Реклама в готовый ролик (только если тишина) ", padding="6")
        adv_box.pack(fill=tk.X, pady=3)

        adv_top = ttk.Frame(adv_box); adv_top.pack(fill=tk.X, pady=2)
        self.ad_cut_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv_top, text="Вставлять рекламный ролик", variable=self.ad_cut_var).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(adv_top, text="На:").pack(side=tk.LEFT)
        self.ad_cut_time_entry = ttk.Entry(adv_top, width=10); self.ad_cut_time_entry.insert(0, "01:00"); self.ad_cut_time_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(adv_top, text="(если стример в этот момент молчит)", font=("Arial", 8, "italic"), foreground="#00aa00").pack(side=tk.LEFT)

        adv_row = ttk.Frame(adv_box); adv_row.pack(fill=tk.X, pady=2)
        ttk.Label(adv_row, text="Ролик (MP4):").pack(side=tk.LEFT)
        self.ad_cut_entry = ttk.Entry(adv_row, width=48); self.ad_cut_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(adv_row, text="Обзор...", command=self.browse_ad_cut_file).pack(side=tk.LEFT)

        self.btn_run_cutter = ttk.Button(self.tab_cutter, text="▶ СОБРАТЬ ФИНАЛ (БЕЗ РАССИНХРОНА)", command=self.start_cutter)
        self.btn_run_cutter.pack(fill=tk.X, pady=6, ipady=5)

    def browse_cut_input(self):
        fn = filedialog.askopenfilename(title="Выберите VOD", filetypes=[("Видео", "*.mp4 *.mkv *.avi *.ts *.flv"), ("Все", "*.*")])
        if fn:
            self.cut_input_entry.delete(0, tk.END); self.cut_input_entry.insert(0, os.path.normpath(fn))
            d, b = os.path.split(fn); name, _ = os.path.splitext(b)
            self.cut_output_entry.delete(0, tk.END); self.cut_output_entry.insert(0, os.path.normpath(os.path.join(d, f"{name}_FINAL_SERIES.mp4")))
            self.save_settings()

    def browse_cut_chat_vid(self):
        fn = filedialog.askopenfilename(title="Выберите видео чата", filetypes=[("Видео", "*.mp4 *.mov *.mkv *.webm"), ("Все", "*.*")])
        if fn:
            self.cut_chat_vid_entry.delete(0, tk.END); self.cut_chat_vid_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def browse_cut_output(self):
        fn = filedialog.asksaveasfilename(title="Сохранить как", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if fn:
            self.cut_output_entry.delete(0, tk.END); self.cut_output_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def browse_tg_avatar(self):
        fn = filedialog.askopenfilename(title="Выберите аватарку", filetypes=[("Картинка", "*.png *.jpg *.jpeg *.webp"), ("Все", "*.*")])
        if fn:
            self.tg_avatar_entry.delete(0, tk.END); self.tg_avatar_entry.insert(0, os.path.normpath(fn))
            self.tg_intro_var.set(True)
            self.save_settings()

    def preview_tg_intro(self):
        av = self.tg_avatar_entry.get().strip()
        if not av or not os.path.exists(av):
            messagebox.showerror("Ошибка", "Сначала выберите аватарку PNG!"); return
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tg_intro_preview.mp4")
        try:
            make_tg_intro(av, self.tg_link_entry.get().strip(), out, self.ffmpeg_bin)
            self.log(f"[ТГК] Превью собрано: {out}")
            os.startfile(out)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def start_cutter(self):
        self.save_settings()
        inp = self.cut_input_entry.get().strip()
        chat_vid = self.cut_chat_vid_entry.get().strip()
        out = self.cut_output_entry.get().strip()
        sw_file = self.swears_file_entry.get().strip()
        png_file = self.overlay_file_entry.get().strip()

        if not inp or not os.path.exists(inp): messagebox.showerror("Ошибка", "Исходный VOD не найден!"); return
        if not out: messagebox.showerror("Ошибка", "Укажите путь сохранения!"); return

        if self.use_overlay_var.get() and (not png_file or not os.path.exists(png_file)):
            messagebox.showerror("Ошибка", "PNG файл не найден!"); return

        if self.censor_after_cut_var.get() and (not sw_file or not os.path.exists(sw_file)):
            messagebox.showerror("Ошибка", "ru_curse_words.txt не найден!"); return

        if self.tg_intro_var.get():
            tg_av = self.tg_avatar_entry.get().strip()
            if not tg_av or not os.path.exists(tg_av):
                messagebox.showerror("Ошибка", "Выберите аватарку для вставки ТГК!"); return

        if getattr(self, "ad_cut_var", None) is not None and self.ad_cut_var.get():
            adf = self.ad_cut_entry.get().strip()
            if not adf or not os.path.exists(adf):
                messagebox.showerror("Ошибка", "Выберите рекламный ролик для вставки!"); return
            if parse_time_str(self.ad_cut_time_entry.get()) is None:
                messagebox.showerror("Ошибка", "Тайминг рекламы в формате 01:00"); return

        out_dir = os.path.dirname(os.path.abspath(out))
        if out_dir: os.makedirs(out_dir, exist_ok=True)

        self.set_busy(True)
        threading.Thread(target=self.run_cutter_pipeline, args=(inp, chat_vid, out, sw_file, png_file), daemon=True).start()

    def run_cutter_pipeline(self, video_path, chat_video_path, output_path, swears_file, png_overlay):
        temp_audio = os.path.join(TEMP_DIR, "temp_analysis_audio.wav")
        temp_clips_dir = os.path.join(TEMP_DIR, "temp_clips_vod")
        temp_list = os.path.join(temp_clips_dir, "list.txt")
        temp_concat_video = os.path.join(TEMP_DIR, "temp_concat_raw.mkv")

        temp_chat_clips_dir = os.path.join(TEMP_DIR, "temp_clips_chat")
        temp_chat_list = os.path.join(temp_chat_clips_dir, "list_chat.txt")
        temp_concat_chat = os.path.join(TEMP_DIR, "temp_concat_chat.mkv")

        do_chat = self.use_chat_var.get() and chat_video_path and os.path.exists(chat_video_path)
        do_teaser = self.keep_teaser_var.get()
        do_censor = self.censor_after_cut_var.get()
        do_overlay = self.use_overlay_var.get() and os.path.exists(png_overlay)
        do_keep_kills = self.keep_kills_var.get()
        do_smooth = self.smooth_cuts_var.get()

        clean_audio_wav = None

        try:
            target_mins = int(self.cut_slider.get())
            total_target_sec = target_mins * 60.0

            self.log("=" * 60)
            self.log(f"СТАРТ МОНТАЖА СЕРИИ: {os.path.basename(video_path)}")
            if do_chat: self.log(f"Подключено видео чата Twitch: {os.path.basename(chat_video_path)}")
            self.log(f"Целевой хронометраж: {target_mins} минут ({target_mins//60}ч {target_mins%60}м).")

            cmd_audio = [self.ffmpeg_bin, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio]
            subprocess.run(cmd_audio, capture_output=True)

            rate, data = wavfile.read(temp_audio); data = data.astype(np.float32)
            window_size = 16000; num_windows = len(data) // window_size
            total_video_dur = float(num_windows)

            rms_values = np.zeros(num_windows)
            for i in range(num_windows):
                w = data[i * window_size : (i + 1) * window_size]
                rms_values[i] = np.sqrt(np.mean(w**2))

            map_list = self.get_parsed_maps()
            if not map_list: map_list = [(1, 0.0, total_video_dur)]

            self.log(f"Определено карт в серии: {len(map_list)} шт.")

            total_raw_maps_duration = 0.0
            parsed_maps_info = []
            for map_num, m_start, m_end in map_list:
                m_end_c = min(total_video_dur, m_end if m_end is not None else total_video_dur)
                raw_dur = max(10.0, m_end_c - m_start)
                total_raw_maps_duration += raw_dur
                parsed_maps_info.append((map_num, m_start, m_end_c, raw_dur))

            compression_ratio = min(1.0, total_target_sec / max(1.0, total_raw_maps_duration))
            self.log(f"[БАЛАНС] Исходная длина карт: {int(total_raw_maps_duration//60)} мин. Сжатие до: {int(total_target_sec//60)} мин (коэффициент {compression_ratio:.2f})")

            # 1. Тизер 12 сек
            teaser_segments = []
            if do_teaser and len(map_list) >= 1:
                self.log("Поиск 3 супер-моментов со всей серии для тизера...")
                all_series_peaks = []
                for _, ms, me in map_list:
                    me_c = me or total_video_dur
                    pks, _ = find_peaks(rms_values[int(ms):int(me_c)], distance=20)
                    for p in pks: all_series_peaks.append((int(ms) + p, rms_values[int(ms) + p]))

                all_series_peaks.sort(key=lambda x: x[1], reverse=True)
                chosen_pks = []
                for pt, score in all_series_peaks:
                    if all(abs(pt - cp) > 80.0 for cp in chosen_pks): chosen_pks.append(pt)
                    if len(chosen_pks) >= 3: break

                for cp in chosen_pks:
                    s_t = max(0.0, cp - 1.5)
                    teaser_segments.append([s_t, s_t + 4.0])

            # 2. Карты финала
# 2. Карты финала
            is_legacy_mode = self.legacy_cutter_mode_var.get()
            if is_legacy_mode:
                self.log("[РЕЖИМ] Включен СТАРЫЙ РЕЖИМ: сохраняем максимум килов без жесткого лимита времени.")
            else:
                self.log("[РЕЖИМ] Включен НОВЫЙ РЕЖИМ: строгий отбор под хронометраж.")

            all_story_segments = []
            for map_num, m_start, m_end_c, m_dur in parsed_maps_info:
                this_map_budget = m_dur * compression_ratio
                self.log(f"--- ОБРАБОТКА КАРТЫ {map_num} (длина {int(m_dur//60)} мин) ---")

                map_segs = []
                used_map_budget = 0.0

                # 1. Драфт (пики) и Трон
                draft_min = 1.5
                try:
                    if getattr(self, "draft_min_spins", None):
                        draft_min = float(self.draft_min_spins[map_num - 1].get())
                except Exception:
                    draft_min = 1.5
                intro_len = max(0.0, draft_min * 60.0) if self.keep_intro_var.get() else 0.0
                intro_len = min(intro_len, max(0.0, m_dur - 60.0))
                if intro_len > 0 and m_dur > 180:
                    map_segs.append({'range': [m_start, m_start + intro_len], 'priority': 9999})
                    used_map_budget += intro_len
                    self.log(f"[ДРАФТ] Карта {map_num}: сохранено {intro_len / 60.0:.1f} мин с начала карты.")

                outro_len = 120.0 if self.keep_outro_var.get() else 0.0
                if outro_len > 0 and m_dur > (intro_len + outro_len + 60):
                    map_segs.append({'range': [m_end_c - outro_len, m_end_c], 'priority': 9999})
                    used_map_budget += outro_len

                # 2. Детектор килов (в старом режиме порог мягче, чтобы ничего не пропустить)
                cv_kill_segments = []
                if do_keep_kills:
                    cv_kill_segments = detect_dota_kills(video_path, m_start + intro_len, m_end_c - outro_len, self.log)
                    if cv_kill_segments:
                        # не обрываем кусок на полуслове: если стример говорит — тянем до конца фразы + 3 с
                        windows = [(max(m_start, s - 6.0), min(m_end_c, e + 30.0)) for s, e in cv_kill_segments]
                        phrases = transcribe_speech_intervals(video_path, windows, self.log)
                        if phrases:
                            extended = extend_cuts_to_speech(cv_kill_segments, phrases, m_end_c)
                            grown = sum(1 for a, b in zip(cv_kill_segments, extended) if b[1] > a[1] + 0.01)
                            cv_kill_segments = []
                            for s, e in extended:
                                if cv_kill_segments and s <= cv_kill_segments[-1][1] + 2.0:
                                    cv_kill_segments[-1][1] = max(cv_kill_segments[-1][1], e)
                                else:
                                    cv_kill_segments.append([s, e])
                            self.log(f"[РЕЧЬ] Дотянуто до конца фразы: {grown} из {len(extended)} кусков.")
                    for k_seg in cv_kill_segments:
                        k_s = k_seg[0]
                        k_e = k_seg[1] if is_legacy_mode else min(k_seg[1], k_s + 90.0)
                        sub_k = rms_values[int(k_s):int(k_e)]
                        prio = 500 + (np.max(sub_k) if len(sub_k) > 0 else 0)
                        map_segs.append({'range': [k_s, k_e], 'priority': prio})
                        used_map_budget += (k_e - k_s)

                # 3. Аудио-пики — в старом режиме только если килы не найдены (иначе карта раздувается вдвое)
                rem_map_budget = this_map_budget - used_map_budget
                need_filler = (not is_legacy_mode and rem_map_budget > 30.0) or (is_legacy_mode and not cv_kill_segments)
                if need_filler:
                    mid_start = m_start + intro_len
                    mid_end = m_end_c - outro_len
                    mid_dur = max(10.0, mid_end - mid_start)
                    e_end = mid_start + (mid_dur * 0.35)
                    mf_end = mid_start + (mid_dur * 0.80)

                    tgt_calc = (rem_map_budget if rem_map_budget > 0 else this_map_budget * 0.4)

                    def get_peaks(s_sec, e_sec, tgt_sec, min_dist=25, is_early=False):
                        sub = rms_values[int(s_sec):int(e_sec)]
                        if len(sub) < 30 or tgt_sec <= 0: return []
                        pks, _ = find_peaks(sub, distance=min_dist)
                        cands = []
                        for p in pks:
                            peak_abs = int(s_sec) + p
                            pre_roll = 35.0 if not is_early else 25.0
                            post_roll = 15.0
                            act_start = max(s_sec, peak_abs - pre_roll)
                            act_end = min(e_sec, peak_abs + post_roll)
                            cands.append({'range': [act_start, act_end], 'priority': float(sub[p])})
                        cands.sort(key=lambda x: x['priority'], reverse=True)
                        sel = []
                        acc_dur = 0.0
                        for c in cands:
                            sel.append(c)
                            acc_dur += (c['range'][1] - c['range'][0])
                            if acc_dur >= tgt_sec: break
                        return sel

                    map_segs.extend(get_peaks(mid_start, e_end, tgt_calc * 0.35, min_dist=20, is_early=True))
                    map_segs.extend(get_peaks(e_end, mf_end, tgt_calc * 0.45, min_dist=25))
                    map_segs.extend(get_peaks(mf_end, mid_end, tgt_calc * 0.20, min_dist=25))

                # Склейка перекрывающихся сцен
                raw_intervals = sorted(map_segs, key=lambda x: x['range'][0])
                merged_intervals = []
                for item in raw_intervals:
                    r = item['range']
                    p = item['priority']
                    if not merged_intervals:
                        merged_intervals.append({'range': r[:], 'priority': p})
                    elif r[0] <= merged_intervals[-1]['range'][1] + (6.0 if is_legacy_mode else 3.0):
                        merged_intervals[-1]['range'][1] = max(merged_intervals[-1]['range'][1], r[1])
                        merged_intervals[-1]['priority'] = max(merged_intervals[-1]['priority'], p)
                    else:
                        merged_intervals.append({'range': r[:], 'priority': p})

                # В СТАРОМ РЕЖИМЕ берем все килы и сцены карты без вырезания
                if is_legacy_mode:
                    selected_for_map = [item['range'] for item in merged_intervals]
                else:
                    # В НОВОМ РЕЖИМЕ укладываемся в бюджет карты
                    merged_intervals.sort(key=lambda x: x['priority'], reverse=True)
                    selected_for_map = []
                    cur_map_total = 0.0
                    for item in merged_intervals:
                        dur = item['range'][1] - item['range'][0]
                        if item['priority'] >= 9000 or (cur_map_total + dur) <= this_map_budget:
                            selected_for_map.append(item['range'])
                            cur_map_total += dur

                selected_for_map.sort(key=lambda x: x[0])
                all_story_segments.extend(selected_for_map)

            # Финальная проверка
            teaser_total_s = len(teaser_segments) * 4.0
            if not is_legacy_mode:
                allowed_story_s = max(60.0, total_target_sec - teaser_total_s)
                final_story_capped = []
                acc_sec = 0.0
                for seg in all_story_segments:
                    dur = seg[1] - seg[0]
                    if acc_sec + dur <= allowed_story_s:
                        final_story_capped.append(seg)
                        acc_sec += dur
                    else:
                        rem = allowed_story_s - acc_sec
                        if rem > 12.0:
                            final_story_capped.append([seg[0], seg[0] + rem])
                        break
                all_story_segments = final_story_capped

            total_s = teaser_total_s + sum(e - s for s, e in all_story_segments)
            self.log("=" * 60)
            self.log(f"ВСЕ КАРТЫ СОБРАНЫ: {len(all_story_segments)} сцен. Хронометраж: {int(total_s//60)} мин ({int(total_s//3600)}ч {int((total_s%3600)//60)}м).")

            # 8. Реклама в готовый ролик: ставим только если в этот момент стример молчит
            ad_cut_at_out = None
            ad_cut_dur = 0.0
            ad_cut_has_audio = False
            ad_cut_time = parse_time_str(self.ad_cut_time_entry.get()) if hasattr(self, "ad_cut_time_entry") else None
            ad_cut_file = self.ad_cut_entry.get().strip() if hasattr(self, "ad_cut_entry") else ""
            if (self.ad_cut_var.get() if hasattr(self, "ad_cut_var") else False) and ad_cut_time and os.path.exists(ad_cut_file):
                src_t = output_to_source_time(teaser_segments + all_story_segments, ad_cut_time)
                if src_t is None:
                    self.log("[РЕКЛАМА] Тайминг за пределами ролика — вставка пропущена.")
                else:
                    phrases = transcribe_speech_intervals(video_path, [(max(0.0, src_t - 4.0), src_t + 6.0)], self.log)
                    if any(a <= src_t <= b for a, b in phrases):
                        self.log(f"[РЕКЛАМА] На {format_time_hms(ad_cut_time)} стример говорит — вставка пропущена.")
                    else:
                        _, _, _, ad_cut_has_audio, ad_cut_dur = probe_media(self.ffmpeg_bin, ad_cut_file)
                        ad_cut_at_out = float(ad_cut_time)
                        total_s += (ad_cut_dur or 0.0)
                        self.log(f"[РЕКЛАМА] Тишина — ставлю ролик на {format_time_hms(ad_cut_time)}.")

            os.makedirs(temp_clips_dir, exist_ok=True)
            if do_chat: os.makedirs(temp_chat_clips_dir, exist_ok=True)

            tg_avatar = self.tg_avatar_entry.get().strip()
            do_tg_intro = self.tg_intro_var.get() and os.path.exists(tg_avatar)
            if do_tg_intro:
                self.log("[ТГК] Собираю вставку-заставку 4 с...")
                make_tg_intro(tg_avatar, self.tg_link_entry.get().strip(),
                              os.path.join(temp_clips_dir, "tg_intro.mkv"), self.ffmpeg_bin)

            clip_names = []
            chat_clip_names = []

            self.log("Покадрово точная нарезка (Frame-Accurate) с плавными переходами...")

            for tidx, (ts, te) in enumerate(teaser_segments + all_story_segments):
                fn = f"cut_{tidx:04d}.mkv"
                dur = te - ts

                base_vf = "setpts=PTS-STARTPTS"
                base_af = "asetpts=PTS-STARTPTS,aresample=44100:first_pts=0"

                # Плавное микро-затухание по краям каждого отрезка
                if do_smooth and dur > 1.2:
                    base_vf += f",fade=t=in:st=0:d=0.18,fade=t=out:st={dur - 0.18:.2f}:d=0.18"
                    base_af += f",afade=t=in:st=0:d=0.15,afade=t=out:st={dur - 0.15:.2f}:d=0.15"

                slice_enc_args = [
                    "-vf", base_vf,
                    "-af", base_af,
                    *video_enc_args(self.ffmpeg_bin, "ultrafast", 22, 60),
                    "-c:a", "pcm_s16le", "-ar", "44100"
                ]

                cmd_cut = [self.ffmpeg_bin, "-y", "-ss", str(ts), "-to", str(te), "-i", video_path] + slice_enc_args + [os.path.join(temp_clips_dir, fn)]
                res_cut = ffmpeg_run(cmd_cut, self.ffmpeg_bin, self.log)
                if res_cut.returncode != 0:
                    raise RuntimeError(f"Кусок {fn} не вырезался:\n{res_cut.stderr.strip()[-300:]}")
                clip_names.append(fn)

                if do_chat:
                    cfn = f"chat_{tidx:04d}.mkv"
                    chat_vf = "setpts=PTS-STARTPTS"
                    if do_smooth and dur > 1.2:
                        chat_vf += f",fade=t=in:st=0:d=0.18,fade=t=out:st={dur - 0.18:.2f}:d=0.18"

                    cmd_ccut = [self.ffmpeg_bin, "-y", "-ss", str(ts), "-to", str(te), "-i", chat_video_path,
                                "-vf", chat_vf] + video_enc_args(self.ffmpeg_bin, "ultrafast", 22, 60) + [
                                "-an", os.path.join(temp_chat_clips_dir, cfn)]
                    res_ccut = ffmpeg_run(cmd_ccut, self.ffmpeg_bin, self.log)
                    if res_ccut.returncode != 0:
                        raise RuntimeError(f"Чат {cfn} не вырезался:\n{res_ccut.stderr.strip()[-300:]}")
                    chat_clip_names.append(cfn)

            if do_tg_intro:
                clip_names.insert(len(teaser_segments), "tg_intro.mkv")
                if do_chat:
                    cts = all_story_segments[0][0] if all_story_segments else 0.0
                    cfn = "chat_tg_intro.mkv"
                    ffmpeg_run([self.ffmpeg_bin, "-y", "-ss", str(cts), "-to", str(cts + 4.0),
                                "-i", chat_video_path, "-vf", "setpts=PTS-STARTPTS"] +
                               video_enc_args(self.ffmpeg_bin, "ultrafast", 22, 60) + [
                                "-an", os.path.join(temp_chat_clips_dir, cfn)], self.ffmpeg_bin, self.log)
                    chat_clip_names.insert(len(teaser_segments), cfn)
                self.log(f"[ТГК] Заставка вставлена после хука (после {len(teaser_segments)} роликов).")

            with open(temp_list, "w", encoding="utf-8") as f:
                for name in clip_names: f.write(f"file '{name}'\n")
            cmd_concat = [self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", os.path.abspath(temp_concat_video)]
            res_c = subprocess.run(cmd_concat, cwd=temp_clips_dir, capture_output=True, text=True, errors="replace")
            if res_c.returncode != 0: raise RuntimeError(f"Ошибка склейки фрагментов:\n{res_c.stderr[-300:]}")

            if do_chat:
                with open(temp_chat_list, "w", encoding="utf-8") as f:
                    for cname in chat_clip_names: f.write(f"file '{cname}'\n")
                cmd_cconcat = [self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", "list_chat.txt", "-c", "copy", os.path.abspath(temp_concat_chat)]
                res_cc = subprocess.run(cmd_cconcat, cwd=temp_chat_clips_dir, capture_output=True, text=True, errors="replace")
                if res_cc.returncode != 0: raise RuntimeError(f"Ошибка склейки чата:\n{res_cc.stderr[-300:]}")

            # 4. Цензура матов
            if do_censor:
                self.log("=" * 60)
                clean_audio_wav, muted_count = self.mute_audio_data(temp_concat_video, swears_file)

            # 5. ФИНАЛЬНЫЙ РЕНДЕР
            self.log("=" * 60)
            self.log("ФИНАЛЬНЫЙ РЕНДЕР: наложение слоев и экспорт в MP4...")

            inputs = ["-i", temp_concat_video]
            filter_chains = []
            cur_v = "0:v"

            if do_chat and os.path.exists(temp_concat_chat):
                chat_idx = len(inputs) // 2
                inputs.extend(["-i", temp_concat_chat])
                try:
                    cx = int(self.cut_chat_x_entry.get()); cy = int(self.cut_chat_y_entry.get())
                    cw = int(self.cut_chat_w_entry.get()); ch_h = int(self.cut_chat_h_entry.get())
                except:
                    cx, cy, cw, ch_h = 35, 240, 340, 550

                filter_chains.append(f"[{chat_idx}:v]scale={cw}:{ch_h}[chat_scaled]")
                filter_chains.append(f"[{cur_v}][chat_scaled]overlay={cx}:{cy}[v_chat]")
                cur_v = "v_chat"

            if do_overlay:
                try:
                    bx = int(self.cut_x_entry.get())
                    by = int(self.cut_y_entry.get())
                    bw = int(self.cut_w_entry.get())
                    bh = int(self.cut_h_entry.get())
                except:
                    bx, by, bw, bh = 1580, 30, 290, 70

                real_png = remove_fake_checkerboard_transparency(png_overlay, self.log)
                logo_idx = len(inputs) // 2
                inputs.extend(["-i", real_png])

                target_w = int(bw * 1.5)
                target_h = int(bh * 1.5)
                cx_pt = bx + bw // 2
                cy_pt = by + bh // 2
                pos_x = max(0, min(1920 - target_w, cx_pt - target_w // 2))
                pos_y = max(0, min(1080 - target_h, cy_pt - target_h // 2))

                self.log(f"[ОВЕРЛЕЙ 1.5x] Баннер {bw}x{bh} -> Закрываем оверлеем {target_w}x{target_h} px в X={pos_x}, Y={pos_y}")

                filter_chains.append(f"[{logo_idx}:v]scale={target_w}:{target_h}[logo]")
                filter_chains.append(f"[{cur_v}][logo]overlay={pos_x}:{pos_y}[v_out]")
                cur_v = "v_out"

            if clean_audio_wav and os.path.exists(clean_audio_wav):
                aud_idx = len(inputs) // 2
                inputs.extend(["-i", clean_audio_wav])
                audio_src = f"{aud_idx}:a:0"
            else:
                audio_src = "0:a:0"

            audio_delay_ms = 0
            try:
                audio_delay_ms = int(self.audio_delay_spin.get())
            except:
                audio_delay_ms = 0

            audio_filter = "aresample=44100:async=1000:first_pts=0"
            if audio_delay_ms > 0:
                audio_filter += f",adelay={audio_delay_ms}|{audio_delay_ms}"
            elif audio_delay_ms < 0:
                trim_s = abs(audio_delay_ms) / 1000.0
                audio_filter = f"atrim=start={trim_s:.3f},asetpts=PTS-STARTPTS," + audio_filter

            enc_flags = video_enc_args(self.ffmpeg_bin, "ultrafast", 22, 60)

            # вставка рекламы в готовый ролик (без второго прохода кодирования)
            if ad_cut_at_out is not None:
                W, H, _, _, _ = probe_media(self.ffmpeg_bin, temp_concat_video)
                W, H = W or 1920, H or 1080
                ad_idx = len(inputs) // 2
                inputs.extend(["-i", ad_cut_file])
                if ad_cut_has_audio:
                    ad_a_idx = ad_idx
                else:
                    ad_a_idx = len(inputs) // 2
                    inputs.extend(["-f", "lavfi", "-t", f"{(ad_cut_dur or 1.0):.3f}", "-i", "anullsrc=r=44100:cl=stereo"])

                filter_chains.append(f"[{cur_v}]fps=60,setsar=1,format=yuv420p,split=2[adt1][adt2]")
                filter_chains.append(f"[adt1]trim=0:{ad_cut_at_out:.3f},setpts=PTS-STARTPTS[adp1]")
                filter_chains.append(f"[adt2]trim=start={ad_cut_at_out:.3f},setpts=PTS-STARTPTS[adp2]")
                filter_chains.append(
                    f"[{ad_idx}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                    f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=60,format=yuv420p[adv]"
                )
                filter_chains.append("[adp1][adv][adp2]concat=n=3:v=1:a=0[v_ad]")
                cur_v = "v_ad"

            if filter_chains:
                if ad_cut_at_out is not None:
                    filter_chains.append(f"[{audio_src}]{audio_filter},asplit=2[aat1][aat2]")
                    filter_chains.append(f"[aat1]atrim=0:{ad_cut_at_out:.3f},asetpts=PTS-STARTPTS[apa1]")
                    filter_chains.append(f"[aat2]atrim=start={ad_cut_at_out:.3f},asetpts=PTS-STARTPTS[apa2]")
                    filter_chains.append(f"[{ad_a_idx}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[aada]")
                    filter_chains.append("[apa1][aada][apa2]concat=n=3:v=0:a=1[a_out]")
                else:
                    filter_chains.append(f"[{audio_src}]{audio_filter}[a_out]")
                cmd_final = [self.ffmpeg_bin, "-y"] + inputs + [
                    "-filter_complex", ";".join(filter_chains),
                    "-map", f"[{cur_v}]",
                    "-map", "[a_out]",
                ] + enc_flags + [
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest", output_path
                ]
                run_ffmpeg_with_progress(cmd_final, total_s, self.log)
            else:
                cmd_mux = [
                    self.ffmpeg_bin, "-y", "-i", temp_concat_video
                ]
                if clean_audio_wav and os.path.exists(clean_audio_wav):
                    cmd_mux.extend(["-i", clean_audio_wav, "-map", "0:v:0", "-map", "1:a:0"])
                else:
                    cmd_mux.extend(["-map", "0:v:0", "-map", "0:a:0"])

                cmd_mux.extend([
                    "-af", audio_filter,
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest", output_path
                ])
                res_m = subprocess.run(cmd_mux, capture_output=True, text=True, errors="replace")
                if res_m.returncode != 0: raise RuntimeError(f"FFmpeg ошибка:\n{res_m.stderr[-300:]}")

            if not os.path.exists(output_path) or os.path.getsize(output_path) < 10000:
                raise RuntimeError(f"Файл не был создан! Проверьте путь сохранения: {output_path}")

            # Генерация текстового файла с таймкодами и склейками
            report_file = os.path.splitext(output_path)[0] + "_таймкоды_и_склейки.txt"
            saved_report = generate_cuts_report(report_file, teaser_segments, all_story_segments, output_path)
            if saved_report:
                self.log(f"[ОТЧЕТ] Текстовый файл со склейками сохранен: {os.path.basename(saved_report)}")

            self.log("=" * 60)
            self.log(f"[УСПЕХ] Ролик успешно смонтирован: {output_path}")
            self.root.after(0, lambda: messagebox.showinfo(
                "Успех!", 
                f"Ролик готов!\nХронометраж: {int(total_s//60)} мин ({int(total_s//3600)}ч {int((total_s%3600)//60)}м)\nФайл с таймкодами и склейками сохранен рядом с видео!"
            ))

        except Exception as e:
            err_msg = str(e)
            self.log(f"[ОШИБКА] {err_msg}")
            self.root.after(0, lambda msg=err_msg: messagebox.showerror("Ошибка", msg))

        finally:
            if os.path.exists(temp_audio):
                try: os.remove(temp_audio)
                except: pass
            if os.path.exists(temp_concat_video):
                try: os.remove(temp_concat_video)
                except: pass
            if os.path.exists(temp_concat_chat):
                try: os.remove(temp_concat_chat)
                except: pass
            if clean_audio_wav and os.path.exists(clean_audio_wav):
                try: os.remove(clean_audio_wav)
                except: pass
            if os.path.exists(temp_clips_dir):
                shutil.rmtree(temp_clips_dir, ignore_errors=True)
            if os.path.exists(temp_chat_clips_dir):
                shutil.rmtree(temp_chat_clips_dir, ignore_errors=True)
            self.root.after(0, self.set_busy, False)

    # ---------------- ВКЛАДКА 2: РЕАКЦИИ / КЕЙСЫ / IRL ----------------
    def build_react_tab(self):
        f_box = ttk.LabelFrame(self.tab_react, text=" 1. Исходные файлы ", padding="8")
        f_box.pack(fill=tk.X, pady=3)

        ttk.Label(f_box, text="Видео стрима:").grid(row=0, column=0, sticky=tk.W)
        self.react_input_entry = ttk.Entry(f_box, width=55)
        self.react_input_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(f_box, text="Обзор...", command=self.browse_react_input).grid(row=0, column=2)

        ttk.Label(f_box, text="Видео чата (опционально):").grid(row=1, column=0, sticky=tk.W)
        self.react_chat_vid_entry = ttk.Entry(f_box, width=55)
        self.react_chat_vid_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(f_box, text="Обзор...", command=self.browse_react_chat_vid).grid(row=1, column=2)

        ttk.Label(f_box, text="Куда сохранить:").grid(row=2, column=0, sticky=tk.W)
        self.react_output_entry = ttk.Entry(f_box, width=55)
        self.react_output_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(f_box, text="Обзор...", command=self.browse_react_output).grid(row=2, column=2)
        f_box.columnconfigure(1, weight=1)

        chat_box = ttk.LabelFrame(self.tab_react, text=" 2. Позиция чата Twitch на экране ", padding="6")
        chat_box.pack(fill=tk.X, pady=3)

        ch_r = ttk.Frame(chat_box)
        ch_r.pack(fill=tk.X, pady=2)
        self.react_use_chat_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ch_r, text="Накладывать чат", variable=self.react_use_chat_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(ch_r, text="🔍 Указать положение на кадре (мышкой)", command=lambda: self.open_interactive_chat_selector(is_react=True)).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(ch_r, text="X:").pack(side=tk.LEFT)
        self.react_chat_x_entry = ttk.Entry(ch_r, width=5); self.react_chat_x_entry.insert(0, "35"); self.react_chat_x_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(ch_r, text="Y:").pack(side=tk.LEFT)
        self.react_chat_y_entry = ttk.Entry(ch_r, width=5); self.react_chat_y_entry.insert(0, "240"); self.react_chat_y_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(ch_r, text="W:").pack(side=tk.LEFT)
        self.react_chat_w_entry = ttk.Entry(ch_r, width=5); self.react_chat_w_entry.insert(0, "340"); self.react_chat_w_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(ch_r, text="H:").pack(side=tk.LEFT)
        self.react_chat_h_entry = ttk.Entry(ch_r, width=5); self.react_chat_h_entry.insert(0, "550"); self.react_chat_h_entry.pack(side=tk.LEFT, padx=(2, 10))

        banner_box = ttk.LabelFrame(self.tab_react, text=" 3. Закрытие баннера FONBET картинкой (1.5x) ", padding="6")
        banner_box.pack(fill=tk.X, pady=3)

        b_top = ttk.Frame(banner_box)
        b_top.pack(fill=tk.X, pady=2)
        self.react_use_overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(b_top, text="Закрывать баннер (увеличение 1.5x)", variable=self.react_use_overlay_var).pack(side=tk.LEFT)

        b_file = ttk.Frame(banner_box)
        b_file.pack(fill=tk.X, pady=2)
        ttk.Label(b_file, text="Файл PNG:").pack(side=tk.LEFT)
        self.react_overlay_file_entry = ttk.Entry(b_file, width=48)
        self.react_overlay_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(b_file, text="Обзор...", command=self.browse_react_overlay_file).pack(side=tk.LEFT)

        b_detect_row = ttk.Frame(banner_box)
        b_detect_row.pack(fill=tk.X, pady=3)
        ttk.Button(b_detect_row, text="🔍 Указать баннер мышкой", command=lambda: self.open_interactive_detector(self.react_input_entry.get(), is_react_tab=True)).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(b_detect_row, text="X:").pack(side=tk.LEFT)
        self.react_x_entry = ttk.Entry(b_detect_row, width=5); self.react_x_entry.insert(0, "1580"); self.react_x_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(b_detect_row, text="Y:").pack(side=tk.LEFT)
        self.react_y_entry = ttk.Entry(b_detect_row, width=5); self.react_y_entry.insert(0, "30"); self.react_y_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(b_detect_row, text="W:").pack(side=tk.LEFT)
        self.react_w_entry = ttk.Entry(b_detect_row, width=5); self.react_w_entry.insert(0, "290"); self.react_w_entry.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(b_detect_row, text="H:").pack(side=tk.LEFT)
        self.react_h_entry = ttk.Entry(b_detect_row, width=5); self.react_h_entry.insert(0, "70"); self.react_h_entry.pack(side=tk.LEFT, padx=(2, 10))

        self.react_time_box = ttk.LabelFrame(self.tab_react, text=" 4. Отрезки времени на стриме (Фрагменты) ", padding="6")
        self.react_time_box.pack(fill=tk.X, pady=3)

        top_btn_bar = ttk.Frame(self.react_time_box)
        top_btn_bar.pack(fill=tk.X, pady=(0, 3))
        ttk.Button(top_btn_bar, text="➕ Добавить фрагмент", command=lambda: self.add_react_fragment("", "")).pack(side=tk.LEFT)
        ttk.Label(top_btn_bar, text=" (например: кейсы 01:10:00 - 01:45:00, потом снова кейсы 03:20:00 - 03:55:00)", font=("Arial", 8, "italic"), foreground="#00aa00").pack(side=tk.LEFT, padx=10)

        self.react_frags_container = ttk.Frame(self.react_time_box)
        self.react_frags_container.pack(fill=tk.X)

        mode_box = ttk.LabelFrame(self.tab_react, text=" 5. Режим контента и правила ", padding="6")
        mode_box.pack(fill=tk.X, pady=3)

        m_row = ttk.Frame(mode_box)
        m_row.pack(fill=tk.X, pady=2)
        ttk.Label(m_row, text="Тип контента:").pack(side=tk.LEFT)
        self.react_mode_combo = ttk.Combobox(m_row, values=[
            "🎬 Реакция на видео (100% сохранение видео без пропусков)",
            "🎁 Открытие сундуков / кейсов (Динамично, без пауз)",
            "💬 Разговоры / Подкаст (Удаление донатов и затиший)"
        ], state="readonly", width=55)
        self.react_mode_combo.set("🎬 Реакция на видео (100% сохранение видео без пропусков)")
        self.react_mode_combo.pack(side=tk.LEFT, padx=(5, 10))

        self.react_cut_donations_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mode_box, text="🚫 Вырезать чтение донатов во время паузы (по ключевым словам)", variable=self.react_cut_donations_var).pack(anchor=tk.W, pady=2)

        p_row = ttk.Frame(mode_box)
        p_row.pack(fill=tk.X, pady=2)
        ttk.Label(p_row, text="Макс. длительность паузы стримера:").pack(side=tk.LEFT)
        self.react_max_pause_spin = ttk.Spinbox(p_row, from_=3, to=60, width=5)
        self.react_max_pause_spin.set(10)
        self.react_max_pause_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(p_row, text="сек (все что дольше — обрезается, оставляя начало и конец реакции)").pack(side=tk.LEFT)

        self.btn_run_react = ttk.Button(self.tab_react, text="▶ СМОНТИРОВАТЬ ВСЕ ФРАГМЕНТЫ (С ЧАТОМ И БАННЕРОМ)", command=self.start_react_pipeline)
        self.btn_run_react.pack(fill=tk.X, pady=6, ipady=5)

    def add_react_fragment(self, start_time="00:00:00", end_time=""):
        row_frame = ttk.Frame(self.react_frags_container)
        row_frame.pack(fill=tk.X, pady=2)

        lbl = ttk.Label(row_frame, text=f"Фрагмент #{len(self.react_fragment_rows)+1}:", width=14, font=("Arial", 9, "bold"))
        lbl.pack(side=tk.LEFT)

        ttk.Label(row_frame, text="Начало:").pack(side=tk.LEFT)
        s_entry = ttk.Entry(row_frame, width=9)
        s_entry.insert(0, start_time or "00:00:00")
        s_entry.pack(side=tk.LEFT, padx=(3, 15))

        ttk.Label(row_frame, text="Конец:").pack(side=tk.LEFT)
        e_entry = ttk.Entry(row_frame, width=9)
        if end_time: e_entry.insert(0, end_time)
        e_entry.pack(side=tk.LEFT, padx=(3, 10))

        row_dict = {
            "frame": row_frame,
            "label": lbl,
            "start_entry": s_entry,
            "end_entry": e_entry
        }

        del_btn = ttk.Button(row_frame, text="❌", width=3, command=lambda: self.remove_react_fragment(row_dict))
        del_btn.pack(side=tk.LEFT)
        row_dict["del_btn"] = del_btn

        self.react_fragment_rows.append(row_dict)
        self.update_react_fragment_indices()

    def remove_react_fragment(self, row_dict):
        if row_dict in self.react_fragment_rows:
            row_dict["frame"].destroy()
            self.react_fragment_rows.remove(row_dict)
            self.update_react_fragment_indices()

    def update_react_fragment_indices(self):
        for idx, r in enumerate(self.react_fragment_rows):
            r["label"].config(text=f"Фрагмент #{idx+1}:")

    def browse_react_input(self):
        fn = filedialog.askopenfilename(title="Выберите видео реакции", filetypes=[("Видео", "*.mp4 *.mkv *.avi *.ts *.flv *.mov"), ("Все", "*.*")])
        if fn:
            self.react_input_entry.delete(0, tk.END); self.react_input_entry.insert(0, os.path.normpath(fn))
            d, b = os.path.split(fn); name, _ = os.path.splitext(b)
            self.react_output_entry.delete(0, tk.END); self.react_output_entry.insert(0, os.path.normpath(os.path.join(d, f"{name}_CLEAN_REACT.mp4")))
            self.save_settings()

    def browse_react_chat_vid(self):
        fn = filedialog.askopenfilename(title="Выберите видео чата", filetypes=[("Видео", "*.mp4 *.mov *.mkv *.webm"), ("Все", "*.*")])
        if fn:
            self.react_chat_vid_entry.delete(0, tk.END); self.react_chat_vid_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def browse_react_output(self):
        fn = filedialog.asksaveasfilename(title="Сохранить как", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if fn:
            self.react_output_entry.delete(0, tk.END); self.react_output_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def browse_react_overlay_file(self):
        fn = filedialog.askopenfilename(title="Выберите PNG", filetypes=[("PNG", "*.png"), ("Все", "*.*")])
        if fn:
            self.react_overlay_file_entry.delete(0, tk.END); self.react_overlay_file_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def start_react_pipeline(self):
        self.save_settings()
        vid = self.react_input_entry.get().strip()
        chat_vid = self.react_chat_vid_entry.get().strip()
        out = self.react_output_entry.get().strip()
        sw_file = self.swears_file_entry.get().strip()
        png_file = self.react_overlay_file_entry.get().strip()

        if not vid or not os.path.exists(vid): messagebox.showerror("Ошибка", "Исходное видео не найдено!"); return
        if not out: messagebox.showerror("Ошибка", "Укажите путь сохранения!"); return

        parsed_fragments = []
        for r in self.react_fragment_rows:
            st = parse_time_str(r['start_entry'].get().strip()) or 0.0
            et = parse_time_str(r['end_entry'].get().strip())
            parsed_fragments.append((st, et))

        if not parsed_fragments:
            messagebox.showerror("Ошибка", "Добавьте хотя бы один фрагмент!"); return

        max_pause_sec = float(self.react_max_pause_spin.get() or 10.0)
        cut_donations = self.react_cut_donations_var.get()
        mode_text = self.react_mode_combo.get()
        do_chat = self.react_use_chat_var.get() and chat_vid and os.path.exists(chat_vid)
        do_overlay = self.react_use_overlay_var.get() and png_file and os.path.exists(png_file)

        self.set_busy(True)

        def worker():
            temp_dir = os.path.join(TEMP_DIR, "react_cut_clips")
            temp_list = os.path.join(temp_dir, "list.txt")
            temp_concat = os.path.join(TEMP_DIR, "react_concat.mkv")

            temp_chat_dir = os.path.join(TEMP_DIR, "react_chat_clips")
            temp_chat_list = os.path.join(temp_chat_dir, "list_chat.txt")
            temp_concat_chat = os.path.join(TEMP_DIR, "react_concat_chat.mkv")

            clean_audio_wav = None

            try:
                os.makedirs(temp_dir, exist_ok=True)
                if do_chat: os.makedirs(temp_chat_dir, exist_ok=True)

                self.log("=" * 60)
                self.log(f"СТАРТ МОНТАЖА РЕАКЦИИ / КЕЙСОВ: {len(parsed_fragments)} фрагментов...")

                cap = cv2.VideoCapture(vid)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_sec = total_frames / fps

                normalized_fragments = []
                for s_val, e_val in parsed_fragments:
                    s_norm = max(0.0, min(total_sec, s_val))
                    e_norm = min(total_sec, e_val if e_val is not None else total_sec)
                    if e_norm > s_norm:
                        normalized_fragments.append((s_norm, e_norm))

                if not normalized_fragments:
                    cap.release()
                    raise RuntimeError("Время конца отрезков должно быть больше времени начала!")

                all_final_segments = []

                for frag_idx, (seg_start, seg_end) in enumerate(normalized_fragments):
                    seg_duration = seg_end - seg_start
                    self.log(f"--- ОБРАБОТКА ФРАГМЕНТА #{frag_idx+1} ({int(seg_start//3600)}ч {int((seg_start%3600)//60):02d}м -> {int(seg_end//3600)}ч {int((seg_end%3600)//60):02d}м, длина: {int(seg_duration//60)} мин) ---")

                    sample_interval = 1.0
                    sample_times = np.arange(seg_start, seg_end, sample_interval)
                    motion_scores = np.zeros(len(sample_times))
                    prev_crop = None

                    for idx, t in enumerate(sample_times):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
                        ret, frame = cap.read()
                        if not ret or frame is None: break
                        h, w = frame.shape[:2]
                        crop = cv2.cvtColor(frame[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)], cv2.COLOR_BGR2GRAY)
                        crop_small = cv2.resize(crop, (160, 90))

                        if prev_crop is not None:
                            diff = cv2.absdiff(crop_small, prev_crop)
                            motion_scores[idx] = np.mean(diff)
                        prev_crop = crop_small

                    donation_ranges = []
                    if cut_donations and WHISPER_AVAILABLE:
                        self.log(f"Поиск донатов во фрагменте #{frag_idx+1}...")
                        temp_wav = os.path.join(TEMP_DIR, f"temp_react_whisper_{frag_idx}.wav")
                        subprocess.run([
                            self.ffmpeg_bin, "-y", "-ss", str(seg_start), "-to", str(seg_end),
                            "-i", vid, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav
                        ], capture_output=True)

                        model = WhisperModel("base", device="auto", compute_type="int8", cpu_threads=4)
                        segments, _ = model.transcribe(temp_wav, language="ru", word_timestamps=True, vad_filter=True, beam_size=1)

                        for seg in segments:
                            if not seg.words: continue
                            for w_item in seg.words:
                                clean_w = re.sub(r"[^\w]", "", w_item.word).lower()
                                if clean_w in DONATION_TRIGGERS:
                                    abs_st = seg_start + w_item.start - 2.0
                                    abs_en = seg_start + w_item.end + 8.0
                                    donation_ranges.append([max(seg_start, abs_st), min(seg_end, abs_en)])

                        if os.path.exists(temp_wav): os.remove(temp_wav)

                    donation_ranges.sort(key=lambda x: x[0])
                    merged_donations = []
                    for d_r in donation_ranges:
                        if not merged_donations: merged_donations.append(d_r)
                        elif d_r[0] <= merged_donations[-1][1] + 3.0:
                            merged_donations[-1][1] = max(merged_donations[-1][1], d_r[1])
                        else: merged_donations.append(d_r)

                    is_react_video = "Реакция" in mode_text
                    motion_threshold = 2.2

                    keep_segments = []
                    pause_start = None

                    for sec_idx, t in enumerate(sample_times):
                        is_playing = motion_scores[sec_idx] > motion_threshold
                        is_in_donation = any(d[0] <= t <= d[1] for d in merged_donations)

                        if is_playing or not is_react_video:
                            if pause_start is not None:
                                pause_dur = t - pause_start
                                if pause_dur > max_pause_sec or is_in_donation:
                                    keep_segments.append([pause_start, pause_start + 4.5])
                                    keep_segments.append([max(pause_start + 4.5, t - 2.0), t])
                                else:
                                    keep_segments.append([pause_start, t])
                                pause_start = None
                            keep_segments.append([t, t + sample_interval])
                        else:
                            if pause_start is None:
                                pause_start = t

                    if pause_start is not None:
                        keep_segments.append([pause_start, min(seg_end, pause_start + 4.5)])

                    keep_segments.sort(key=lambda x: x[0])
                    frag_final = []
                    for seg in keep_segments:
                        if not frag_final: frag_final.append(seg)
                        elif seg[0] <= frag_final[-1][1] + 0.5:
                            frag_final[-1][1] = max(frag_final[-1][1], seg[1])
                        else: frag_final.append(seg)

                    all_final_segments.extend(frag_final)

                cap.release()

                self.log(f"[РЕЗУЛЬТАТ] Сформировано {len(all_final_segments)} непрерывных сцен.")

                clip_names = []
                chat_clip_names = []
                slice_args = [
                    "-vf", "setpts=PTS-STARTPTS", "-af", "asetpts=PTS-STARTPTS,aresample=44100:first_pts=0",
                ] + video_enc_args(self.ffmpeg_bin, "ultrafast", 22, 60) + [
                    "-c:a", "pcm_s16le", "-ar", "44100"
                ]

                for idx, (st, en) in enumerate(all_final_segments):
                    fn = f"react_{idx:04d}.mkv"
                    cmd = [self.ffmpeg_bin, "-y", "-ss", str(st), "-to", str(en), "-i", vid] + slice_args + [os.path.join(temp_dir, fn)]
                    subprocess.run(cmd, capture_output=True)
                    clip_names.append(fn)

                    if do_chat:
                        cfn = f"chat_{idx:04d}.mkv"
                        cmd_c = [self.ffmpeg_bin, "-y", "-ss", str(st), "-to", str(en), "-i", chat_vid,
                                 "-vf", "setpts=PTS-STARTPTS"] + video_enc_args(self.ffmpeg_bin, "ultrafast", 22, 60) + [
                                 "-an", os.path.join(temp_chat_dir, cfn)]
                        subprocess.run(cmd_c, capture_output=True)
                        chat_clip_names.append(cfn)

                with open(temp_list, "w", encoding="utf-8") as f:
                    for n in clip_names: f.write(f"file '{n}'\n")

                subprocess.run([self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", os.path.abspath(temp_concat)], cwd=temp_dir, capture_output=True)

                if do_chat:
                    with open(temp_chat_list, "w", encoding="utf-8") as f:
                        for cn in chat_clip_names: f.write(f"file '{cn}'\n")
                    subprocess.run([self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", "list_chat.txt", "-c", "copy", os.path.abspath(temp_concat_chat)], cwd=temp_chat_dir, capture_output=True)

                if self.censor_after_cut_var.get() and sw_file and os.path.exists(sw_file):
                    clean_audio_wav, _ = self.mute_audio_data(temp_concat, sw_file)

                inputs = ["-i", temp_concat]
                filter_chains = []
                cur_v = "0:v"

                if do_chat and os.path.exists(temp_concat_chat):
                    chat_idx = len(inputs) // 2
                    inputs.extend(["-i", temp_concat_chat])
                    try:
                        cx = int(self.react_chat_x_entry.get()); cy = int(self.react_chat_y_entry.get())
                        cw = int(self.react_chat_w_entry.get()); ch_h = int(self.react_chat_h_entry.get())
                    except:
                        cx, cy, cw, ch_h = 35, 240, 340, 550

                    filter_chains.append(f"[{chat_idx}:v]scale={cw}:{ch_h}[chat_scaled]")
                    filter_chains.append(f"[{cur_v}][chat_scaled]overlay={cx}:{cy}[v_chat]")
                    cur_v = "v_chat"

                if do_overlay:
                    try:
                        bx = int(self.react_x_entry.get()), int(self.react_y_entry.get())
                        bw, bh = int(self.react_w_entry.get()), int(self.react_h_entry.get())
                    except:
                        bx, by, bw, bh = 1580, 30, 290, 70

                    real_png = remove_fake_checkerboard_transparency(png_file, self.log)
                    logo_idx = len(inputs) // 2
                    inputs.extend(["-i", real_png])

                    target_w = int(bw * 1.5); target_h = int(bh * 1.5)
                    cx_pt = bx + bw // 2; cy_pt = by + bh // 2
                    pos_x = max(0, min(1920 - target_w, cx_pt - target_w // 2))
                    pos_y = max(0, min(1080 - target_h, cy_pt - target_h // 2))

                    filter_chains.append(f"[{logo_idx}:v]scale={target_w}:{target_h}[logo]")
                    filter_chains.append(f"[{cur_v}][logo]overlay={pos_x}:{pos_y}[v_out]")
                    cur_v = "v_out"

                enc = video_enc_args(self.ffmpeg_bin, "faster", 20, 60) + ["-c:a", "aac", "-b:a", "192k", "-shortest"]
                if filter_chains:
                    audio_src = f"{len(inputs)//2}:a:0" if clean_audio_wav else "0:a:0"
                    if clean_audio_wav: inputs.extend(["-i", clean_audio_wav])
                    cmd_final = [self.ffmpeg_bin, "-y"] + inputs + ["-filter_complex", ";".join(filter_chains), "-map", f"[{cur_v}]", "-map", audio_src] + enc + [out]
                else:
                    cmd_final = [self.ffmpeg_bin, "-y", "-i", temp_concat]
                    if clean_audio_wav: cmd_final.extend(["-i", clean_audio_wav, "-map", "0:v:0", "-map", "1:a:0"])
                    cmd_final.extend(enc + [out] if clean_audio_wav else ["-c", "copy", out])

                subprocess.run(cmd_final, capture_output=True)
                self.log(f"[УСПЕХ] Готово! Сохранено в: {out}")
                self.root.after(0, lambda: messagebox.showinfo("Готово!", f"Все фрагменты смонтированы и склеены!\nФайл: {os.path.basename(out)}"))
            except Exception as e:
                self.log(f"[ОШИБКА] {e}")
                self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
            finally:
                if os.path.exists(temp_dir): shutil.rmtree(temp_dir, ignore_errors=True)
                if os.path.exists(temp_chat_dir): shutil.rmtree(temp_chat_dir, ignore_errors=True)
                if os.path.exists(temp_concat):
                    try: os.remove(temp_concat)
                    except: pass
                if os.path.exists(temp_concat_chat):
                    try: os.remove(temp_concat_chat)
                    except: pass
                self.root.after(0, self.set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    # ---------------- ВКЛАДКА 3: ТОП КЛИПЫ ----------------
    def build_clips_tab(self):
        c_set_box = ttk.LabelFrame(self.tab_clips, text=" 1. Настройки поиска клипов ", padding="8"); c_set_box.pack(fill=tk.X, pady=4)
        row1 = ttk.Frame(c_set_box); row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="Канал стримера:").pack(side=tk.LEFT)
        self.clips_channel_entry = ttk.Entry(row1, width=15); self.clips_channel_entry.insert(0, "daxak"); self.clips_channel_entry.pack(side=tk.LEFT, padx=(5, 20))
        self.clips_period_combo = ttk.Combobox(row1, values=["За месяц (30 дней)", "За неделю (7 дней)", "За все время"], state="readonly", width=22)
        self.clips_period_combo.set("За месяц (30 дней)"); self.clips_period_combo.pack(side=tk.LEFT, padx=(5, 20))
        self.clips_count_spin = ttk.Spinbox(row1, from_=3, to=40, width=5); self.clips_count_spin.set(15); self.clips_count_spin.pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(c_set_box); row2.pack(fill=tk.X, pady=3)
        ttk.Label(row2, text="Сохранить ролик как:").pack(side=tk.LEFT)
        self.clips_out_entry = ttk.Entry(row2, width=45); self.clips_out_entry.insert(0, os.path.abspath("DAXAK_TOP_CLIPS_MONTH.mp4")); self.clips_out_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="Обзор...", command=self.browse_clips_out).pack(side=tk.LEFT)

        c_opt_box = ttk.LabelFrame(self.tab_clips, text=" 2. Стиль, плашки и оформление клипов ", padding="8"); c_opt_box.pack(fill=tk.X, pady=4)
        self.clips_titles_var = tk.BooleanVar(value=True); ttk.Checkbutton(c_opt_box, text="🏷 Название клипа в ПРАВОМ ВЕРХНЕМ углу", variable=self.clips_titles_var).pack(anchor=tk.W)
        self.clips_fade_var = tk.BooleanVar(value=True); ttk.Checkbutton(c_opt_box, text="✨ Плавный переход между клипами (Smooth Fade)", variable=self.clips_fade_var).pack(anchor=tk.W)

        opt_r1 = ttk.Frame(c_opt_box); opt_r1.pack(fill=tk.X, pady=4)
        self.clips_censor_var = tk.BooleanVar(value=True); ttk.Checkbutton(opt_r1, text="🔇 Заглушать маты (Whisper)", variable=self.clips_censor_var).pack(side=tk.LEFT)
        self.clips_swears_entry = ttk.Entry(opt_r1, width=38); self.clips_swears_entry.pack(side=tk.LEFT, padx=8)
        ttk.Button(opt_r1, text="Обзор словаря...", command=self.browse_swears_file).pack(side=tk.LEFT)

        opt_r2 = ttk.Frame(c_opt_box); opt_r2.pack(fill=tk.X, pady=2)
        self.clips_overlay_var = tk.BooleanVar(value=True); ttk.Checkbutton(opt_r2, text="🖼 Закрыть FONBET (1.5x)", variable=self.clips_overlay_var).pack(side=tk.LEFT)
        self.clips_png_entry = ttk.Entry(opt_r2, width=38); self.clips_png_entry.pack(side=tk.LEFT, padx=8)
        ttk.Button(opt_r2, text="Обзор PNG...", command=self.browse_overlay_file).pack(side=tk.LEFT)

        self.btn_run_clips = ttk.Button(self.tab_clips, text="🔥 СКАЧАТЬ КЛИПЫ И СОБРАТЬ ЛУЧШИЕ МОМЕНТЫ", command=self.start_clips_pipeline)
        self.btn_run_clips.pack(fill=tk.X, pady=10, ipady=6)

    def browse_clips_out(self):
        fn = filedialog.asksaveasfilename(title="Сохранить клипы", defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
        if fn:
            self.clips_out_entry.delete(0, tk.END); self.clips_out_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def start_clips_pipeline(self):
        self.save_settings()
        ch = self.clips_channel_entry.get().strip() or "daxak"
        out = self.clips_out_entry.get().strip()
        period = self.clips_period_combo.get()
        range_code = "7d" if "недел" in period else ("all" if "все время" in period else "30d")
        limit = int(self.clips_count_spin.get())
        if not out: messagebox.showerror("Ошибка", "Укажите имя для сохранения!"); return

        do_titles = self.clips_titles_var.get()
        do_fade = self.clips_fade_var.get()
        do_censor = self.clips_censor_var.get()
        sw_file = self.clips_swears_entry.get().strip()
        do_overlay = self.clips_overlay_var.get()
        png = self.clips_png_entry.get().strip()

        if do_overlay and (not png or not os.path.exists(png)):
            messagebox.showerror("Ошибка", "PNG для закрытия баннера не найден!"); return
        if do_censor and WHISPER_AVAILABLE and (not sw_file or not os.path.exists(sw_file)):
            messagebox.showerror("Ошибка", "Словарь матов не найден!"); return

        out_dir = os.path.dirname(os.path.abspath(out))
        if out_dir: os.makedirs(out_dir, exist_ok=True)

        self.set_busy(True)
        def worker():
            temp_dir = os.path.join(TEMP_DIR, "dota_studio_clips")
            try:
                clip_items = download_top_twitch_clips_ytdlp(ch, range_code, limit, temp_dir, self.log)
                if not clip_items: raise RuntimeError("Не удалось скачать клипы.")

                self.log(f"[КЛИПЫ] Скачано {len(clip_items)}. Накладываю титры и собираю сборник...")
                build_clips_compilation(clip_items, out, self.ffmpeg_bin,
                                        titles=do_titles, fade=do_fade,
                                        overlay_png=png if do_overlay else None, log=self.log)

                if do_censor and WHISPER_AVAILABLE:
                    self.log("[КЛИПЫ] Заглушка матов...")
                    clean_wav, muted = self.mute_audio_data(out, sw_file)
                    if clean_wav and os.path.exists(clean_wav):
                        tmp_out = out + ".muted.mp4"
                        r = subprocess.run([self.ffmpeg_bin, "-y", "-i", out, "-i", clean_wav,
                                            "-map", "0:v:0", "-map", "1:a:0",
                                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                                            "-shortest", tmp_out], capture_output=True, text=True, errors="replace")
                        if r.returncode == 0:
                            os.replace(tmp_out, out)
                        try: os.remove(clean_wav)
                        except: pass
                        self.log(f"[КЛИПЫ] Заглушено матов: {muted}")

                self.log(f"[УСПЕХ] Сборник готов: {out}")
                self.root.after(0, lambda: messagebox.showinfo("Готово!", "Клипы собраны!"))
            except Exception as e:
                self.log(f"[ОШИБКА] {e}")
            finally:
                if os.path.exists(temp_dir): shutil.rmtree(temp_dir, ignore_errors=True)
                self.root.after(0, self.set_busy, False)
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- ВКЛАДКА 4: SHORTS (9:16) ----------------
    def build_shorts_tab(self):
        s_box = ttk.LabelFrame(self.tab_shorts, text=" 1. Выбор видео ", padding="8"); s_box.pack(fill=tk.X, pady=4)
        ttk.Label(s_box, text="Исходное видео:").grid(row=0, column=0, sticky=tk.W)
        self.shorts_input_entry = ttk.Entry(s_box, width=58); self.shorts_input_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(s_box, text="Обзор...", command=self.browse_shorts_input).grid(row=0, column=2)
        ttk.Label(s_box, text="Папка для Shorts:").grid(row=1, column=0, sticky=tk.W)
        self.shorts_out_dir_entry = ttk.Entry(s_box, width=58); self.shorts_out_dir_entry.insert(0, os.path.abspath("Shorts_Result")); self.shorts_out_dir_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(s_box, text="Выбрать...", command=self.browse_shorts_out_dir).grid(row=1, column=2)
        s_box.columnconfigure(1, weight=1)

        cam_box = ttk.LabelFrame(self.tab_shorts, text=" 2. Позиция вебки Дахака (сверху 1080x600) ", padding="8"); cam_box.pack(fill=tk.X, pady=4)
        c_row1 = ttk.Frame(cam_box); c_row1.pack(fill=tk.X, pady=3)
        ttk.Button(c_row1, text="🔍 Выбрать вебку на кадре (мышкой)", command=self.open_shorts_cam_selector).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(c_row1, text="X:").pack(side=tk.LEFT); self.shorts_cam_x = ttk.Entry(c_row1, width=5); self.shorts_cam_x.insert(0, "1520"); self.shorts_cam_x.pack(side=tk.LEFT, padx=2)
        ttk.Label(c_row1, text="Y:").pack(side=tk.LEFT); self.shorts_cam_y = ttk.Entry(c_row1, width=5); self.shorts_cam_y.insert(0, "720"); self.shorts_cam_y.pack(side=tk.LEFT, padx=2)
        ttk.Label(c_row1, text="W:").pack(side=tk.LEFT); self.shorts_cam_w = ttk.Entry(c_row1, width=5); self.shorts_cam_w.insert(0, "400"); self.shorts_cam_w.pack(side=tk.LEFT, padx=2)
        ttk.Label(c_row1, text="H:").pack(side=tk.LEFT); self.shorts_cam_h = ttk.Entry(c_row1, width=5); self.shorts_cam_h.insert(0, "300"); self.shorts_cam_h.pack(side=tk.LEFT, padx=2)

        param_box = ttk.LabelFrame(self.tab_shorts, text=" 3. Настройки нарезки ", padding="8"); param_box.pack(fill=tk.X, pady=4)
        self.shorts_count_spin = ttk.Spinbox(param_box, from_=1, to=15, width=5); self.shorts_count_spin.set(5); self.shorts_count_spin.pack(side=tk.LEFT, padx=5)
        self.shorts_dur_lbl = ttk.Label(param_box, text="Длина: 30 сек", font=("Arial", 9, "bold")); self.shorts_dur_lbl.pack(side=tk.LEFT, padx=10)
        self.shorts_dur_slider = ttk.Scale(param_box, from_=15, to=55, orient=tk.HORIZONTAL, command=self.on_shorts_slider); self.shorts_dur_slider.set(30); self.shorts_dur_slider.pack(fill=tk.X, pady=3)

        p_row2 = ttk.Frame(param_box); p_row2.pack(fill=tk.X, pady=3)
        self.shorts_censor_var = tk.BooleanVar(value=True); ttk.Checkbutton(p_row2, text="🔇 Заглушать маты", variable=self.shorts_censor_var).pack(side=tk.LEFT)
        self.shorts_swears_entry = ttk.Entry(p_row2, width=38); self.shorts_swears_entry.pack(side=tk.LEFT, padx=8)
        ttk.Button(p_row2, text="Обзор словаря...", command=self.browse_swears_file).pack(side=tk.LEFT)

        ad_box = ttk.LabelFrame(self.tab_shorts, text=" 4. Реклама DotaUp (вставка в начало) ", padding="8"); ad_box.pack(fill=tk.X, pady=4)
        self.shorts_ad_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ad_box, text="Вставлять рекламный ролик в начало каждого Shorts", variable=self.shorts_ad_var).pack(anchor=tk.W)
        ad_row = ttk.Frame(ad_box); ad_row.pack(fill=tk.X, pady=3)
        ttk.Label(ad_row, text="Ролик:").pack(side=tk.LEFT)
        self.shorts_ad_entry = ttk.Entry(ad_row, width=48); self.shorts_ad_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(ad_row, text="Обзор...", command=self.browse_shorts_ad).pack(side=tk.LEFT)
        ttk.Label(ad_box, text="Ролик играет целиком и без изменений: шортс замирает, звук молчит, потом продолжается.", font=("Arial", 8, "italic"), foreground="#00aa00").pack(anchor=tk.W)

        self.btn_run_shorts = ttk.Button(self.tab_shorts, text="📱 НАРЕЗАТЬ ВЕРТИКАЛЬНЫЕ SHORTS (9:16)", command=self.start_shorts_pipeline)
        self.btn_run_shorts.pack(fill=tk.X, pady=10, ipady=6)

    def browse_shorts_input(self):
        fn = filedialog.askopenfilename(title="Выберите видео для Shorts", filetypes=[("Видео", "*.mp4 *.mkv *.avi *.mov"), ("Все", "*.*")])
        if fn:
            self.shorts_input_entry.delete(0, tk.END); self.shorts_input_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def browse_shorts_out_dir(self):
        dn = filedialog.askdirectory(title="Выберите папку для Shorts")
        if dn:
            self.shorts_out_dir_entry.delete(0, tk.END); self.shorts_out_dir_entry.insert(0, os.path.normpath(dn))
            self.save_settings()

    def browse_shorts_ad(self):
        fn = filedialog.askopenfilename(title="Выберите рекламный ролик DotaUp", filetypes=[("Видео", "*.mp4 *.mov *.mkv *.webm"), ("Все", "*.*")])
        if fn:
            self.shorts_ad_entry.delete(0, tk.END); self.shorts_ad_entry.insert(0, os.path.normpath(fn))
            self.shorts_ad_var.set(True)
            self.save_settings()

    def open_shorts_cam_selector(self):
        video_p = self.shorts_input_entry.get().strip() or self.cut_input_entry.get().strip()
        if not video_p or not os.path.exists(video_p): messagebox.showerror("Ошибка", "Сначала выберите видео!"); return
        if not CV2_AVAILABLE: messagebox.showerror("Ошибка", "cv2 не установлен!"); return

        cap = cv2.VideoCapture(video_p); total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_f * 0.35)); ret, frame = cap.read(); cap.release()
        if not ret or frame is None: messagebox.showerror("Ошибка", "Не удалось прочитать кадр!"); return

        win_name = "Выделите ВЕБКУ Дахака мышкой и нажмите ENTER"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL); cv2.resizeWindow(win_name, 1280, 720)
        user_roi = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True); cv2.destroyAllWindows()

        ux, uy, uw, uh = user_roi
        if uw > 30 and uh > 30:
            self.shorts_cam_x.delete(0, tk.END); self.shorts_cam_x.insert(0, str(ux))
            self.shorts_cam_y.delete(0, tk.END); self.shorts_cam_y.insert(0, str(uy))
            self.shorts_cam_w.delete(0, tk.END); self.shorts_cam_w.insert(0, str(uw))
            self.shorts_cam_h.delete(0, tk.END); self.shorts_cam_h.insert(0, str(uh))
            self.save_settings()
            messagebox.showinfo("Готово!", f"Вебка зафиксирована:\nX={ux}, Y={uy}\nРазмер: {uw}x{uh} px")

    def start_shorts_pipeline(self):
        vid = self.shorts_input_entry.get().strip(); out_dir = self.shorts_out_dir_entry.get().strip()
        if not vid or not os.path.exists(vid): messagebox.showerror("Ошибка", "Выберите видео!"); return
        ad_file = self.shorts_ad_entry.get().strip() if self.shorts_ad_var.get() else ""
        if ad_file and not os.path.exists(ad_file): messagebox.showerror("Ошибка", "Файл рекламы не найден!"); return
        try:
            cam_x, cam_y = int(self.shorts_cam_x.get()), int(self.shorts_cam_y.get())
            cam_w, cam_h = int(self.shorts_cam_w.get()), int(self.shorts_cam_h.get())
        except: messagebox.showerror("Ошибка", "Координаты должны быть числами!"); return

        count = int(self.shorts_count_spin.get()); short_dur = int(self.shorts_dur_slider.get())
        self.set_busy(True)

        def worker():
            temp_audio = os.path.join(TEMP_DIR, "temp_shorts_audio.wav")
            try:
                subprocess.run([self.ffmpeg_bin, "-y", "-i", vid, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio], capture_output=True)
                rate, data = wavfile.read(temp_audio); data = data.astype(np.float32)
                window_size = 16000; num_windows = len(data) // window_size
                rms_values = np.array([np.sqrt(np.mean(data[i*window_size:(i+1)*window_size]**2)) for i in range(num_windows)])

                pks, _ = find_peaks(rms_values, distance=short_dur + 5)
                chosen_peaks = sorted(sorted(pks, key=lambda p: rms_values[p], reverse=True)[:count])

                os.makedirs(out_dir, exist_ok=True)
                stack = (
                    f"[0:v]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},scale=1080:600:force_original_aspect_ratio=increase,crop=1080:600[cam];"
                    f"[0:v]crop=1300:1080:310:0,scale=1080:1320:force_original_aspect_ratio=increase,crop=1080:1320[game];"
                    f"[cam][game]vstack"
                )
                if ad_file:
                    ad_cov, ad_green, ad_dur = ad_probe(self.ffmpeg_bin, ad_file)
                    chroma = ad_dur is not None and ad_green is not None and ad_green > 40.0
                    if chroma:
                        self.log("[РЕКЛАМА] Хромакей: зелёный фон вырезается, ролик ложится поверх шорта.")
                    elif ad_cov is not None and ad_cov < 30.0:
                        self.log(f"[РЕКЛАМА] ВНИМАНИЕ: ролик займёт {ad_cov:.0f}% кадра, DotaUp платит только от 30%")
                    elif ad_cov:
                        self.log(f"[РЕКЛАМА] Ролик займёт {ad_cov:.0f}% кадра, вставка перед пиком.")

                for idx, p in enumerate(chosen_peaks):
                    start_s = max(0, p - int(short_dur * 0.35)); end_s = start_s + short_dur
                    out_name = os.path.join(out_dir, f"short_{idx+1:02d}.mp4")
                    if chroma:
                        # шортс замирает на время ролика, поэтому пик окажется через 0.5 с после его конца
                        ad_at = max(0.5, (p - start_s) - 0.5)
                        filter_complex = shorts_ad_overlay_filter(stack, ad_at, ad_dur)
                        self.log(f"[РЕКЛАМА] {os.path.basename(out_name)}: ролик на {int(ad_at//60)}:{int(ad_at%60):02d}, шортс на паузе, пик после.")
                    elif ad_file:
                        ad_at = max(1.0, (p - start_s) - 1.0)
                        filter_complex = shorts_ad_splice_filter(stack, ad_at)
                        self.log(f"[РЕКЛАМА] {os.path.basename(out_name)}: ролик на {int(ad_at//60)}:{int(ad_at%60):02d}, пик сразу после.")
                    else:
                        filter_complex = stack + "[v_out]"
                    if ad_file:
                        cmd = [self.ffmpeg_bin, "-y", "-ss", str(start_s), "-to", str(end_s), "-i", vid, "-i", ad_file,
                               "-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]"] + \
                              video_enc_args(self.ffmpeg_bin, "ultrafast", 22, 60) + \
                              ["-c:a", "aac", "-b:a", "192k", out_name]
                    else:
                        cmd = [self.ffmpeg_bin, "-y", "-ss", str(start_s), "-to", str(end_s), "-i", vid,
                               "-filter_complex", filter_complex, "-map", "[v_out]", "-map", "0:a:0"] + \
                              video_enc_args(self.ffmpeg_bin, "ultrafast", 22, 60) + \
                              ["-c:a", "aac", "-b:a", "192k", out_name]
                    res = ffmpeg_run(cmd, self.ffmpeg_bin, self.log)
                    if res.returncode != 0:
                        self.log(f"[ОШИБКА] {os.path.basename(out_name)}: {res.stderr.strip()[-300:]}")

                self.root.after(0, lambda: messagebox.showinfo("Готово!", f"Shorts сохранены в:\n{out_dir}"))
            except Exception as e:
                self.log(f"[ОШИБКА] {e}")
            finally:
                if os.path.exists(temp_audio):
                    try: os.remove(temp_audio)
                    except: pass
                self.root.after(0, self.set_busy, False)
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- ВКЛАДКА 5: ТОЛЬКО ЦЕНЗОР ----------------
    def build_censor_tab(self):
        c_box = ttk.LabelFrame(self.tab_censor, text=" Видеофайл ", padding="10"); c_box.pack(fill=tk.X, pady=4)
        ttk.Label(c_box, text="Видео:").grid(row=0, column=0, sticky=tk.W)
        self.censor_input_entry = ttk.Entry(c_box, width=60); self.censor_input_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(c_box, text="Обзор...", command=self.browse_censor_input).grid(row=0, column=2)
        ttk.Label(c_box, text="Куда сохранить:").grid(row=1, column=0, sticky=tk.W)
        self.censor_output_entry = ttk.Entry(c_box, width=60); self.censor_output_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(c_box, text="Обзор...", command=self.browse_censor_output).grid(row=1, column=2)
        c_box.columnconfigure(1, weight=1)

        dict_c_box = ttk.LabelFrame(self.tab_censor, text=" Словарь матов ", padding="10"); dict_c_box.pack(fill=tk.X, pady=4)
        self.censor_swears_entry = ttk.Entry(dict_c_box, width=60); self.censor_swears_entry.grid(row=0, column=0, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(dict_c_box, text="Обзор...", command=self.browse_censor_swears).grid(row=0, column=1)
        dict_c_box.columnconfigure(0, weight=1)

        self.btn_run_censor_only = ttk.Button(self.tab_censor, text="🔇 ЗАГЛУШИТЬ МАТЫ В ЭТОМ ВИДЕО", command=self.start_censor_only)
        self.btn_run_censor_only.pack(fill=tk.X, pady=10, ipady=4)

    def browse_censor_input(self):
        fn = filedialog.askopenfilename(title="Выберите видео", filetypes=[("Видео", "*.mp4 *.mkv *.avi *.mov"), ("Все", "*.*")])
        if fn:
            self.censor_input_entry.delete(0, tk.END); self.censor_input_entry.insert(0, os.path.normpath(fn))
            d, b = os.path.split(fn); name, _ = os.path.splitext(b)
            self.censor_output_entry.delete(0, tk.END); self.censor_output_entry.insert(0, os.path.normpath(os.path.join(d, f"{name}_MUTED.mp4")))
            self.save_settings()

    def browse_censor_output(self):
        fn = filedialog.asksaveasfilename(title="Сохранить результат", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if fn:
            self.censor_output_entry.delete(0, tk.END); self.censor_output_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def start_censor_only(self):
        self.save_settings()
        inp = self.censor_input_entry.get().strip(); out = self.censor_output_entry.get().strip()
        sw_file = self.censor_swears_entry.get().strip()
        if not inp or not out or not sw_file: messagebox.showerror("Ошибка", "Заполните все поля!"); return

        self.set_busy(True)
        def worker():
            try:
                clean_wav, muted = self.mute_audio_data(inp, sw_file)
                if clean_wav:
                    cmd_mux = [self.ffmpeg_bin, "-y", "-i", inp, "-i", clean_wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0", "-shortest", out]
                    subprocess.run(cmd_mux, capture_output=True)
                    if os.path.exists(clean_wav): os.remove(clean_wav)
                else:
                    shutil.copyfile(inp, out)
                self.log(f"[УСПЕХ] Заглушено матов: {muted}. Файл: {out}")
                self.root.after(0, lambda: messagebox.showinfo("Готово!", f"Готово!\nЗаглушено слов: {muted}"))
            except Exception as e:
                self.log(f"[ОШИБКА] {e}")
            finally:
                self.root.after(0, self.set_busy, False)
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- ВКЛАДКА 6: ТОЛЬКО БАННЕР ----------------
    def build_overlay_tab(self):
        o_box = ttk.LabelFrame(self.tab_overlay, text=" 1. Файлы ", padding="8"); o_box.pack(fill=tk.X, pady=4)
        ttk.Label(o_box, text="Исходное видео:").grid(row=0, column=0, sticky=tk.W)
        self.single_vid_entry = ttk.Entry(o_box, width=58); self.single_vid_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(o_box, text="Обзор...", command=self.browse_single_vid).grid(row=0, column=2)
        ttk.Label(o_box, text="Куда сохранить:").grid(row=1, column=0, sticky=tk.W)
        self.single_out_entry = ttk.Entry(o_box, width=58); self.single_out_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(o_box, text="Обзор...", command=self.browse_single_out).grid(row=1, column=2)
        ttk.Label(o_box, text="Картинка (PNG):").grid(row=2, column=0, sticky=tk.W)
        self.single_overlay_file_entry = ttk.Entry(o_box, width=58); self.single_overlay_file_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(o_box, text="Обзор...", command=self.browse_single_overlay_file).grid(row=2, column=2)
        o_box.columnconfigure(1, weight=1)

        pos_box = ttk.LabelFrame(self.tab_overlay, text=" 2. Позиция и размер баннера (картинка увеличится в 1.5x) ", padding="8"); pos_box.pack(fill=tk.X, pady=4)
        p_row = ttk.Frame(pos_box); p_row.pack(fill=tk.X, pady=3)
        ttk.Button(p_row, text="🔍 Указать баннер мышкой", command=lambda: self.open_interactive_detector(self.single_vid_entry.get(), is_single_tab=True)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(p_row, text="X:").pack(side=tk.LEFT); self.single_x_entry = ttk.Entry(p_row, width=5); self.single_x_entry.insert(0, "1580"); self.single_x_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(p_row, text="Y:").pack(side=tk.LEFT); self.single_y_entry = ttk.Entry(p_row, width=5); self.single_y_entry.insert(0, "30"); self.single_y_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(p_row, text="W:").pack(side=tk.LEFT); self.single_w_entry = ttk.Entry(p_row, width=5); self.single_w_entry.insert(0, "290"); self.single_w_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(p_row, text="H:").pack(side=tk.LEFT); self.single_h_entry = ttk.Entry(p_row, width=5); self.single_h_entry.insert(0, "70"); self.single_h_entry.pack(side=tk.LEFT, padx=2)

        self.btn_run_single_overlay = ttk.Button(self.tab_overlay, text="▶ ЗАКРЫТЬ БАННЕР КАРТИНКОЙ (1.5x)", command=self.start_single_overlay)
        self.btn_run_single_overlay.pack(fill=tk.X, pady=10, ipady=4)

    def browse_single_vid(self):
        fn = filedialog.askopenfilename(title="Выберите видео", filetypes=[("Видео", "*.mp4 *.mkv *.avi *.mov"), ("Все", "*.*")])
        if fn:
            self.single_vid_entry.delete(0, tk.END); self.single_vid_entry.insert(0, os.path.normpath(fn))
            d, b = os.path.split(fn); name, _ = os.path.splitext(b)
            self.single_out_entry.delete(0, tk.END); self.single_out_entry.insert(0, os.path.normpath(os.path.join(d, f"{name}_OVERLAY.mp4")))
            self.save_settings()

    def browse_single_out(self):
        fn = filedialog.asksaveasfilename(title="Сохранить результат", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if fn:
            self.single_out_entry.delete(0, tk.END); self.single_out_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def browse_single_overlay_file(self):
        self.browse_overlay_file()

    def start_single_overlay(self):
        self.save_settings()
        vid = self.single_vid_entry.get().strip(); out = self.single_out_entry.get().strip()
        png = self.single_overlay_file_entry.get().strip()
        if not vid or not out or not png: messagebox.showerror("Ошибка", "Заполните все поля!"); return

        try:
            bx, by = int(self.single_x_entry.get()), int(self.single_y_entry.get())
            bw, bh = int(self.single_w_entry.get()), int(self.single_h_entry.get())
        except: messagebox.showerror("Ошибка", "Координаты должны быть числами!"); return

        self.set_busy(True)
        def worker():
            try:
                real_png = remove_fake_checkerboard_transparency(png, self.log)
                target_w = int(bw * 1.5); target_h = int(bh * 1.5)
                cx_pt = bx + bw // 2; cy_pt = by + bh // 2
                pos_x = max(0, min(1920 - target_w, cx_pt - target_w // 2))
                pos_y = max(0, min(1080 - target_h, cy_pt - target_h // 2))

                filter_complex_str = f"[1:v]scale={target_w}:{target_h}[logo];[0:v][logo]overlay={pos_x}:{pos_y}[v_out]"
                cmd = [self.ffmpeg_bin, "-y", "-i", vid, "-i", real_png, "-filter_complex", filter_complex_str,
                       "-map", "[v_out]", "-map", "0:a?"] + video_enc_args(self.ffmpeg_bin, "faster", 20, None) + [
                       "-c:a", "copy", out]
                subprocess.run(cmd, capture_output=True)
                self.log(f"[УСПЕХ] Готово: {out}")
                self.root.after(0, lambda: messagebox.showinfo("Готово!", "Баннер закрыт!"))
            except Exception as e:
                self.log(f"[ОШИБКА] {e}")
            finally:
                self.root.after(0, self.set_busy, False)
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- ВКЛАДКА 7: ВСТАВКА РЕКЛАМЫ ----------------
    def build_ad_tab(self):
        f_box = ttk.LabelFrame(self.tab_ads, text=" 1. Файлы ", padding="8"); f_box.pack(fill=tk.X, pady=4)

        ttk.Label(f_box, text="Исходное видео:").grid(row=0, column=0, sticky=tk.W)
        self.ad_src_entry = ttk.Entry(f_box, width=58); self.ad_src_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(f_box, text="Обзор...", command=self.browse_ad_src).grid(row=0, column=2)

        ttk.Label(f_box, text="Рекламный ролик:").grid(row=1, column=0, sticky=tk.W)
        self.ad_file_entry = ttk.Entry(f_box, width=58); self.ad_file_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(f_box, text="Обзор...", command=self.browse_ad_file).grid(row=1, column=2)

        ttk.Label(f_box, text="Куда сохранить:").grid(row=2, column=0, sticky=tk.W)
        self.ad_out_entry = ttk.Entry(f_box, width=58); self.ad_out_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(f_box, text="Обзор...", command=self.browse_ad_out).grid(row=2, column=2)
        f_box.columnconfigure(1, weight=1)

        t_box = ttk.LabelFrame(self.tab_ads, text=" 2. Тайминг вставки ", padding="8"); t_box.pack(fill=tk.X, pady=4)
        t_row = ttk.Frame(t_box); t_row.pack(fill=tk.X)
        ttk.Label(t_row, text="Вставить на:").pack(side=tk.LEFT)
        self.ad_time_entry = ttk.Entry(t_row, width=12); self.ad_time_entry.insert(0, "01:00"); self.ad_time_entry.pack(side=tk.LEFT, padx=6)
        ttk.Label(t_row, text="(мм:сс или чч:мм:сс)").pack(side=tk.LEFT)
        ttk.Label(t_box, text="Ролик встанет на указанной секунде и растянет видео на свою длину. Всё одним проходом.",
                  font=("Arial", 8, "italic"), foreground="#00aa00").pack(anchor=tk.W, pady=(4, 0))

        self.btn_run_ad = ttk.Button(self.tab_ads, text="📢 ВСТАВИТЬ РЕКЛАМУ", command=self.start_ad_insert)
        self.btn_run_ad.pack(fill=tk.X, pady=10, ipady=6)

    def browse_ad_src(self):
        fn = filedialog.askopenfilename(title="Выберите видео", filetypes=[("Видео", "*.mp4 *.mkv *.mov *.avi"), ("Все", "*.*")])
        if fn:
            self.ad_src_entry.delete(0, tk.END); self.ad_src_entry.insert(0, os.path.normpath(fn))
            d, b = os.path.split(fn); name, _ = os.path.splitext(b)
            self.ad_out_entry.delete(0, tk.END); self.ad_out_entry.insert(0, os.path.normpath(os.path.join(d, f"{name}_WITH_AD.mp4")))
            self.save_settings()

    def browse_ad_file(self):
        fn = filedialog.askopenfilename(title="Выберите рекламный ролик", filetypes=[("Видео", "*.mp4 *.mov *.mkv"), ("Все", "*.*")])
        if fn:
            self.ad_file_entry.delete(0, tk.END); self.ad_file_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def browse_ad_out(self):
        fn = filedialog.asksaveasfilename(title="Сохранить как", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if fn:
            self.ad_out_entry.delete(0, tk.END); self.ad_out_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def browse_ad_cut_file(self):
        fn = filedialog.askopenfilename(title="Выберите рекламный ролик", filetypes=[("Видео", "*.mp4 *.mov *.mkv"), ("Все", "*.*")])
        if fn:
            self.ad_cut_entry.delete(0, tk.END); self.ad_cut_entry.insert(0, os.path.normpath(fn))
            self.ad_cut_var.set(True)
            self.save_settings()

    def start_ad_insert(self):
        self.save_settings()
        src = self.ad_src_entry.get().strip()
        ad = self.ad_file_entry.get().strip()
        out = self.ad_out_entry.get().strip()
        if not src or not os.path.exists(src): messagebox.showerror("Ошибка", "Выберите исходное видео!"); return
        if not ad or not os.path.exists(ad): messagebox.showerror("Ошибка", "Выберите рекламный ролик!"); return
        if not out: messagebox.showerror("Ошибка", "Укажите куда сохранить!"); return
        at = parse_time_str(self.ad_time_entry.get())
        if at is None: messagebox.showerror("Ошибка", "Тайминг в формате 01:00 или 00:01:00"); return

        out_dir = os.path.dirname(os.path.abspath(out))
        if out_dir: os.makedirs(out_dir, exist_ok=True)

        self.set_busy(True)
        def worker():
            try:
                self.log(f"[РЕКЛАМА] Вставка на {format_time_hms(at)}...")
                insert_ad_at(src, ad, at, out, self.ffmpeg_bin, log_callback=self.log)
                self.log(f"[РЕКЛАМА] Готово: {out}")
                self.root.after(0, lambda: messagebox.showinfo("Готово!", f"Реклама вставлена:\n{out}"))
            except Exception as e:
                self.log(f"[ОШИБКА] {e}")
                self.root.after(0, lambda msg=str(e): messagebox.showerror("Ошибка", msg))
            finally:
                self.root.after(0, self.set_busy, False)
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- ВКЛАДКА 8: ОПИСАНИЕ ЧЕРЕЗ DEEPSEEK ----------------
    def build_desc_tab(self):
        d_box = ttk.LabelFrame(self.tab_desc, text=" 1. Готовый ролик ", padding="8"); d_box.pack(fill=tk.X, pady=4)

        ttk.Label(d_box, text="Видео:").grid(row=0, column=0, sticky=tk.W)
        self.desc_video_entry = ttk.Entry(d_box, width=58); self.desc_video_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(d_box, text="Обзор...", command=self.browse_desc_video).grid(row=0, column=2)

        ttk.Label(d_box, text="Куда сохранить:").grid(row=1, column=0, sticky=tk.W)
        self.desc_out_entry = ttk.Entry(d_box, width=58); self.desc_out_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(d_box, text="Обзор...", command=self.browse_desc_out).grid(row=1, column=2)
        d_box.columnconfigure(1, weight=1)

        k_box = ttk.LabelFrame(self.tab_desc, text=" 2. DeepSeek ", padding="8"); k_box.pack(fill=tk.X, pady=4)
        k_row = ttk.Frame(k_box); k_row.pack(fill=tk.X, pady=2)
        ttk.Label(k_row, text="API-ключ:").pack(side=tk.LEFT)
        self.deepseek_key_entry = ttk.Entry(k_row, width=46, show="*"); self.deepseek_key_entry.pack(side=tk.LEFT, padx=5)
        self.deepseek_show_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(k_row, text="показать", variable=self.deepseek_show_var,
                        command=lambda: self.deepseek_key_entry.config(show="" if self.deepseek_show_var.get() else "*")).pack(side=tk.LEFT)
        ttk.Label(k_box, text="Ключ берётся из поля или из переменной DEEPSEEK_API_KEY. Хранится в studio_config.json локально.",
                  font=("Arial", 8, "italic"), foreground="#00aa00").pack(anchor=tk.W, pady=(4, 0))

        self.btn_run_desc = ttk.Button(self.tab_desc, text="🤖 СГЕНЕРИРОВАТЬ ЗАГОЛОВОК, ОПИСАНИЕ И ГЛАВЫ",
                                       command=self.start_desc_pipeline)
        self.btn_run_desc.pack(fill=tk.X, pady=10, ipady=6)

    def browse_desc_video(self):
        fn = filedialog.askopenfilename(title="Выберите готовый ролик", filetypes=[("Видео", "*.mp4 *.mkv *.mov"), ("Все", "*.*")])
        if fn:
            self.desc_video_entry.delete(0, tk.END); self.desc_video_entry.insert(0, os.path.normpath(fn))
            d, b = os.path.split(fn); name, _ = os.path.splitext(b)
            self.desc_out_entry.delete(0, tk.END); self.desc_out_entry.insert(0, os.path.normpath(os.path.join(d, f"{name}_описание.txt")))
            self.save_settings()

    def browse_desc_out(self):
        fn = filedialog.asksaveasfilename(title="Сохранить описание", defaultextension=".txt", filetypes=[("Текст", "*.txt")])
        if fn:
            self.desc_out_entry.delete(0, tk.END); self.desc_out_entry.insert(0, os.path.normpath(fn))
            self.save_settings()

    def start_desc_pipeline(self):
        self.save_settings()
        vid = self.desc_video_entry.get().strip()
        out = self.desc_out_entry.get().strip()
        if not vid or not os.path.exists(vid): messagebox.showerror("Ошибка", "Выберите готовый ролик!"); return
        if not out: messagebox.showerror("Ошибка", "Укажите куда сохранить описание!"); return
        if not WHISPER_AVAILABLE:
            messagebox.showerror("Ошибка", "faster-whisper не установлен — расшифровать нечем"); return
        key = (self.deepseek_key_entry.get().strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip())
        if not key:
            messagebox.showerror("Ошибка", "Впиши API-ключ DeepSeek или задай DEEPSEEK_API_KEY"); return

        out_dir = os.path.dirname(os.path.abspath(out))
        if out_dir: os.makedirs(out_dir, exist_ok=True)

        self.set_busy(True)
        def worker():
            try:
                answer = make_description(vid, out, key, self.log)
                self.log("=" * 60)
                for line in answer.splitlines():
                    self.log(line)
                self.log("=" * 60)
                self.root.after(0, lambda: messagebox.showinfo("Готово!", f"Описание сохранено:\n{out}"))
                try: os.startfile(out)
                except Exception: pass
            except Exception as e:
                self.log(f"[ОШИБКА] {e}")
                self.root.after(0, lambda msg=str(e): messagebox.showerror("Ошибка", msg))
            finally:
                self.root.after(0, self.set_busy, False)
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- УМНЫЙ ЦЕНЗОР WHISPER ----------------
    def mute_audio_data(self, video_path, swear_file_path):
        if not WHISPER_AVAILABLE:
            self.log("[ЦЕНЗОР] faster-whisper не установлен. Пропуск цензуры.")
            return None, 0

        swear_dict = load_user_swears(swear_file_path)
        if not swear_dict:
            self.log("[ЦЕНЗОР] Словарь матов пуст или не найден.")
            return None, 0

        temp_wav = os.path.join(TEMP_DIR, f"censor_in_{int(time.time()*1000)}.wav")
        out_clean_wav = os.path.join(TEMP_DIR, f"censor_out_{int(time.time()*1000)}.wav")

        try:
            self.log("[ЦЕНЗОР] Экспорт аудиодорожки для анализа речи...")
            cmd = [self.ffmpeg_bin, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", temp_wav]
            subprocess.run(cmd, capture_output=True, check=True)

            rate, audio_data = wavfile.read(temp_wav)
            total_duration_sec = len(audio_data) / float(rate)

            has_cuda = False
            try:
                import torch
                has_cuda = torch.cuda.is_available()
            except: pass

            device = "cuda" if has_cuda else "cpu"
            compute_type = "float16" if has_cuda else "int8"

            self.log(f"[ЦЕНЗОР] Запуск Whisper (модель: base, устройство: {device.upper()})...")
            model = WhisperModel("base", device=device, compute_type=compute_type, cpu_threads=4)

            self.log(f"[ЦЕНЗОР] Распознавание речи ({int(total_duration_sec//60)} мин)...")
            segments, info = model.transcribe(temp_wav, language="ru", word_timestamps=True, vad_filter=True, beam_size=1)

            clean_audio = audio_data.copy()
            muted_count = 0
            last_logged_time = 0

            fade_len = int(0.005 * rate)
            fade_out = np.linspace(1, 0, fade_len)
            fade_in = np.linspace(0, 1, fade_len)

            for segment in segments:
                if segment.end - last_logged_time >= 30.0:
                    last_logged_time = segment.end
                    pct = min(99.0, (segment.end / total_duration_sec) * 100.0)
                    cur_m, cur_s = int(segment.end // 60), int(segment.end % 60)
                    tot_m, tot_s = int(total_duration_sec // 60), int(total_duration_sec % 60)
                    self.log(f"[ЦЕНЗОР ПРОГРЕСС] {pct:.1f}% ({cur_m}:{cur_s:02d} / {tot_m}:{tot_s:02d}) | Заглушено: {muted_count}")

                if not segment.words: continue

                for word_info in segment.words:
                    word_clean = word_info.word.strip()
                    if is_bad_word(word_clean, swear_dict):
                        muted_count += 1
                        s_time = max(0.0, word_info.start - 0.05)
                        e_time = min(total_duration_sec, word_info.end + 0.05)

                        s_idx = int(s_time * rate)
                        e_idx = int(e_time * rate)

                        if e_idx - s_idx > fade_len * 2:
                            if len(clean_audio.shape) == 2:
                                clean_audio[s_idx : s_idx + fade_len, :] = (clean_audio[s_idx : s_idx + fade_len, :].T * fade_out).T
                                clean_audio[s_idx + fade_len : e_idx - fade_len, :] = 0
                                clean_audio[e_idx - fade_len : e_idx, :] = (clean_audio[e_idx - fade_len : e_idx, :].T * fade_in).T
                            else:
                                clean_audio[s_idx : s_idx + fade_len] = clean_audio[s_idx : s_idx + fade_len] * fade_out
                                clean_audio[s_idx + fade_len : e_idx - fade_len] = 0
                                clean_audio[e_idx - fade_len : e_idx] = clean_audio[e_idx - fade_len : e_idx] * fade_in

            self.log(f"[ЦЕНЗОР УСПЕХ] Анализ завершен! Заглушено матов: {muted_count}")
            wavfile.write(out_clean_wav, rate, clean_audio)
            return out_clean_wav, muted_count

        except Exception as e:
            self.log(f"[ЦЕНЗОР ОШИБКА] {str(e)}")
            return None, 0
        finally:
            if os.path.exists(temp_wav):
                try: os.remove(temp_wav)
                except: pass

    # ---------------- ОБЩИЕ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ----------------
    def browse_overlay_file(self):
        fn = filedialog.askopenfilename(title="Выберите PNG", filetypes=[("PNG", "*.png"), ("Все", "*.*")])
        if fn:
            entries = [
                getattr(self, 'overlay_file_entry', None),
                getattr(self, 'single_overlay_file_entry', None),
                getattr(self, 'clips_png_entry', None)
            ]
            for e in entries:
                if e is not None:
                    e.delete(0, tk.END)
                    e.insert(0, os.path.normpath(fn))
            self.save_settings()

    def browse_swears_file(self):
        fn = filedialog.askopenfilename(title="Выберите словарь", filetypes=[("Text", "*.txt"), ("Все", "*.*")])
        if fn:
            entries = [
                getattr(self, 'swears_file_entry', None),
                getattr(self, 'censor_swears_entry', None),
                getattr(self, 'clips_swears_entry', None),
                getattr(self, 'shorts_swears_entry', None)
            ]
            for e in entries:
                if e is not None:
                    e.delete(0, tk.END)
                    e.insert(0, os.path.normpath(fn))
            self.save_settings()

    def browse_censor_swears(self):
        self.browse_swears_file()

    def on_cut_slider(self, val):
        m = int(float(val))
        txt = f"Длина ролика: {m} минут ({m//60}ч {m%60}м)" if m >= 60 else f"Длина ролика: {m} минут"
        self.target_label.config(text=txt)

    def set_cut_duration(self, mins):
        self.cut_slider.set(mins); self.on_cut_slider(mins)

    def on_shorts_slider(self, val):
        self.shorts_dur_lbl.config(text=f"Длина: {int(float(val))} сек")

    def toggle_mode_view(self):
        if self.series_mode_var.get() == "series":
            self.single_frame.pack_forget(); self.series_frame.pack(fill=tk.X, pady=4)
        else:
            self.series_frame.pack_forget(); self.single_frame.pack(fill=tk.X, pady=4)

    def get_parsed_maps(self):
        if self.series_mode_var.get() == "single":
            s = parse_time_str(self.start_time_entry.get()) or 0.0
            e = parse_time_str(self.end_time_entry.get())
            return [(1, s, e)]

        raw_pairs = [
            (self.m1_s.get(), self.m1_e.get()),
            (self.m2_s.get(), self.m2_e.get()),
            (self.m3_s.get(), self.m3_e.get()),
            (self.m4_s.get(), self.m4_e.get()),
            (self.m5_s.get(), self.m5_e.get())
        ]
        maps = []
        for idx, (rs, re_t) in enumerate(raw_pairs):
            s = parse_time_str(rs); e = parse_time_str(re_t)
            if s is not None and e is not None and e > s: maps.append((idx + 1, s, e))
        return maps

    def open_interactive_chat_selector(self, is_react=False):
        video_p = (self.react_input_entry.get() if is_react else self.cut_input_entry.get()).strip()
        if not video_p or not os.path.exists(video_p): messagebox.showerror("Ошибка", "Сначала выберите видео!"); return
        if not CV2_AVAILABLE: messagebox.showerror("Ошибка", "cv2 не установлен!"); return

        cap = cv2.VideoCapture(video_p); total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_f * 0.35)); ret, frame = cap.read(); cap.release()
        if not ret or frame is None: messagebox.showerror("Ошибка", "Не удалось прочитать кадр!"); return

        try:
            if is_react:
                cur_x = int(self.react_chat_x_entry.get()); cur_y = int(self.react_chat_y_entry.get())
                cur_w = int(self.react_chat_w_entry.get()); ch_h = int(self.react_chat_h_entry.get())
            else:
                cur_x = int(self.cut_chat_x_entry.get()); cur_y = int(self.cut_chat_y_entry.get())
                cur_w = int(self.cut_chat_w_entry.get()); ch_h = int(self.cut_chat_h_entry.get())
        except:
            cur_x, cur_y, cur_w, ch_h = 35, 240, 340, 550

        preview_frame = frame.copy()
        cv2.rectangle(preview_frame, (cur_x, cur_y), (cur_x + cur_w, cur_y + ch_h), (255, 200, 0), 2)
        cv2.putText(preview_frame, "TWITCH CHAT [ENTER - OK, либо выделите мышкой]", (max(10, cur_x), max(25, cur_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        win_name = "Выбор положения чата Twitch (ENTER - зафиксировать)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL); cv2.resizeWindow(win_name, 1280, 720)
        user_roi = cv2.selectROI(win_name, preview_frame, fromCenter=False, showCrosshair=True); cv2.destroyAllWindows()

        ux, uy, uw, uh = user_roi
        final_x, final_y, final_w, final_h = (ux, uy, uw, uh) if (uw > 40 and uh > 40) else (cur_x, cur_y, cur_w, ch_h)

        if is_react:
            self.react_chat_x_entry.delete(0, tk.END); self.react_chat_x_entry.insert(0, str(final_x))
            self.react_chat_y_entry.delete(0, tk.END); self.react_chat_y_entry.insert(0, str(final_y))
            self.react_chat_w_entry.delete(0, tk.END); self.react_chat_w_entry.insert(0, str(final_w))
            self.react_chat_h_entry.delete(0, tk.END); self.react_chat_h_entry.insert(0, str(final_h))
        else:
            self.cut_chat_x_entry.delete(0, tk.END); self.cut_chat_x_entry.insert(0, str(final_x))
            self.cut_chat_y_entry.delete(0, tk.END); self.cut_chat_y_entry.insert(0, str(final_y))
            self.cut_chat_w_entry.delete(0, tk.END); self.cut_chat_w_entry.insert(0, str(final_w))
            self.cut_chat_h_entry.delete(0, tk.END); self.cut_chat_h_entry.insert(0, str(final_h))

        self.save_settings()
        messagebox.showinfo("Готово!", f"Чат зафиксирован:\nX={final_x}, Y={final_y}\nРазмер: {final_w}x{final_h} px")

    def open_interactive_detector(self, video_path, is_single_tab=False, is_react_tab=False):
        if not video_path or not os.path.exists(video_path): messagebox.showerror("Ошибка", "Сначала выберите видео!"); return
        if not CV2_AVAILABLE: messagebox.showerror("Ошибка", "cv2 не установлен!"); return

        cap = cv2.VideoCapture(video_path); total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_f * 0.35)); ret, frame = cap.read(); cap.release()
        if not ret or frame is None: messagebox.showerror("Ошибка", "Не удалось прочитать кадр!"); return

        try:
            if is_react_tab:
                cur_x = int(self.react_x_entry.get()); cur_y = int(self.react_y_entry.get())
                cur_w = int(self.react_w_entry.get()); cur_h = int(self.react_h_entry.get())
            elif is_single_tab:
                cur_x = int(self.single_x_entry.get()); cur_y = int(self.single_y_entry.get())
                cur_w = int(self.single_w_entry.get()); cur_h = int(self.single_h_entry.get())
            else:
                cur_x = int(self.cut_x_entry.get()); cur_y = int(self.cut_y_entry.get())
                cur_w = int(self.cut_w_entry.get()); cur_h = int(self.cut_h_entry.get())
            box = (cur_x, cur_y, cur_w, cur_h)
        except:
            box = detect_fonbet_box_single_frame(frame) or (1580, 30, 290, 70)

        preview_frame = frame.copy(); bx, by, bw, bh = box
        cv2.rectangle(preview_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(preview_frame, "ТЕКУЩИЙ БАННЕР [Выделите мышкой нужную область и нажмите ENTER]", (max(10, bx - 120), max(25, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        win_name = "Выбор положения баннера FONBET (ENTER - зафиксировать)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL); cv2.resizeWindow(win_name, 1280, 720)
        user_roi = cv2.selectROI(win_name, preview_frame, fromCenter=False, showCrosshair=True); cv2.destroyAllWindows()

        ux, uy, uw, uh = user_roi
        final_box = (ux, uy, uw, uh) if (uw > 20 and uh > 15) else box
        fx, fy, fw, fh = final_box

        if is_react_tab:
            self.react_x_entry.delete(0, tk.END); self.react_x_entry.insert(0, str(fx))
            self.react_y_entry.delete(0, tk.END); self.react_y_entry.insert(0, str(fy))
            self.react_w_entry.delete(0, tk.END); self.react_w_entry.insert(0, str(fw))
            self.react_h_entry.delete(0, tk.END); self.react_h_entry.insert(0, str(fh))
        elif is_single_tab:
            self.single_x_entry.delete(0, tk.END); self.single_x_entry.insert(0, str(fx))
            self.single_y_entry.delete(0, tk.END); self.single_y_entry.insert(0, str(fy))
            self.single_w_entry.delete(0, tk.END); self.single_w_entry.insert(0, str(fw))
            self.single_h_entry.delete(0, tk.END); self.single_h_entry.insert(0, str(fh))
        else:
            self.cut_x_entry.delete(0, tk.END); self.cut_x_entry.insert(0, str(fx))
            self.cut_y_entry.delete(0, tk.END); self.cut_y_entry.insert(0, str(fy))
            self.cut_w_entry.delete(0, tk.END); self.cut_w_entry.insert(0, str(fw))
            self.cut_h_entry.delete(0, tk.END); self.cut_h_entry.insert(0, str(fh))

        self.save_settings()
        self.log(f"[КАЛИБРОВКА БАННЕРА] Сохранены координаты: X={fx}, Y={fy}, W={fw}, H={fh} (Размер оверлея 1.5x: {int(fw*1.5)}x{int(fh*1.5)} px)")
        messagebox.showinfo("Готово!", f"Позиция баннера сохранена:\nX={fx}, Y={fy}, W={fw}, H={fh}\nКартинка закроет его с увеличением 1.5x ({int(fw*1.5)}x{int(fh*1.5)} px)")

    def set_busy(self, busy=True):
        if busy:
            self.btn_run_cutter.config(state=tk.DISABLED); self.btn_run_react.config(state=tk.DISABLED)
            self.btn_run_censor_only.config(state=tk.DISABLED); self.btn_run_single_overlay.config(state=tk.DISABLED)
            self.btn_run_clips.config(state=tk.DISABLED); self.btn_run_shorts.config(state=tk.DISABLED)
            if hasattr(self, "btn_run_ad"): self.btn_run_ad.config(state=tk.DISABLED)
            if hasattr(self, "btn_run_desc"): self.btn_run_desc.config(state=tk.DISABLED)
            self.progress_bar.start(10)
        else:
            self.btn_run_cutter.config(state=tk.NORMAL); self.btn_run_react.config(state=tk.NORMAL)
            self.btn_run_censor_only.config(state=tk.NORMAL); self.btn_run_single_overlay.config(state=tk.NORMAL)
            if hasattr(self, "btn_run_ad"): self.btn_run_ad.config(state=tk.NORMAL)
            if hasattr(self, "btn_run_desc"): self.btn_run_desc.config(state=tk.NORMAL)
            self.btn_run_clips.config(state=tk.NORMAL); self.btn_run_shorts.config(state=tk.NORMAL)
            self.progress_bar.stop()

    def log(self, message):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.root.after(0, self._append_log, timestamp + message + "\n")

    def _append_log(self, text):
        self.log_text.insert(tk.END, text); self.log_text.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedDotaStudioApp(root)
    root.mainloop()