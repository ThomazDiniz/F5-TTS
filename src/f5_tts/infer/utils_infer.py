# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys

os.environ["PYTOCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/")

import hashlib
import re
import shutil
import subprocess
import tempfile
import wave
from importlib.resources import files

import librosa
import soundfile as sf
import matplotlib

# HuggingFace ASR: path local em str -> open+ffmpeg_read(bytes). Se o ffmpeg devolver 0 amostras,
# levanta o mesmo ValueError falso "Soundfile is either not in the correct format or is malformed"
# (vem de transformers.pipelines.audio_utils.ffmpeg_read — NÃO do soundfile).
# Ver: https://github.com/Vaibhavs10/insanely-fast-whisper/issues/90 (muitas vezes ffmpeg/libs quebrados).
# Evitamos: só passar dict {raw, sampling_rate} carregado com soundfile/numpy.

matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import snapshot_download, hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)

_ref_audio_cache = {}

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None
_asr_ffmpeg_env_logged = False


def _log_ffmpeg_environment_and_transformers() -> None:
    """Diagnóstico: o erro 'Soundfile is malformed' no ASR do HF vem muitas vezes de ffmpeg a devolver 0 bytes."""
    global _asr_ffmpeg_env_logged
    if _asr_ffmpeg_env_logged:
        return
    _asr_ffmpeg_env_logged = True
    try:
        import transformers

        _asr_transcribe_log(
            f"env: transformers={getattr(transformers, '__version__', '?')} "
            f"torch={torch.__version__} torchaudio={torchaudio.__version__}"
        )
    except Exception as e:  # noqa: BLE001
        _asr_transcribe_log(f"env: nao foi possivel ler versoes: {e}")
    exe = shutil.which("ffmpeg")
    _asr_transcribe_log(f"env: ffmpeg which={exe!r} (necessario se o HF pipeline passar ficheiro por str; com dict numpy normalmente basta; se o ffmpeg no sistema estiver partido, ver github.com/Vaibhavs10/insanely-fast-whisper/issues/90)")
    if not exe:
        _asr_transcribe_log("env: AVISO — ffmpeg nao esta no PATH. No Docker a imagem ja inclui; no Windows, instala ffmpeg (winget/choco) ou conda-forge: ffmpeg")
        return
    try:
        p = subprocess.run(
            [exe, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        first = (p.stdout or "").splitlines()[:1]
        _asr_transcribe_log(f"env: ffmpeg -version rc={p.returncode} {first!r}")
    except Exception as e:  # noqa: BLE001
        _asr_transcribe_log(f"env: ffmpeg -version FALHOU: {e}")
    # Teste rápido: decodificar 0,1s de sinal (falha se ffmpeg/libs estiverem partidos, como no issue #90)
    try:
        r2 = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=16000:duration=0.1",
                "-f",
                "f32le",
                "-ac",
                "1",
                "pipe:1",
            ],
            capture_output=True,
            timeout=15,
        )
        n = int(len(r2.stdout) // 4)  # float32
        if r2.returncode != 0 or n == 0:
            _asr_transcribe_log(
                f"env: teste lavfi->pipe FALHOU (rc={r2.returncode} samples~{n} stderr={(r2.stderr or b'').decode(errors='replace')[:400]!r}) — o teu ffmpeg pode estar partido; reinstala."
            )
        else:
            _asr_transcribe_log(f"env: teste ffmpeg lavfi ok (~{n} amostras float32)")
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001
        _asr_transcribe_log(
            f"env: teste lavfi nao executado (normal em Windows sem lavfi, etc.): {e}"
        )


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 6
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    _log_ffmpeg_environment_and_transformers()
    print("[ASR] Loading Whisper model (first time may take a minute)...", flush=True)
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
        ignore_warning=True,
    )
    print("[ASR] Whisper ready.", flush=True)


# transcribe


def _asr_transcribe_log(msg: str) -> None:
    line = f"[F5TTS-ASR] {msg}"
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)


def _looks_like_hf_ffmpeg_read_error(err_text: str) -> bool:
    """A mensagem 'Soundfile' vem muitas vezes de `transformers.pipelines.audio_utils.ffmpeg_read`, não do ficheiro."""
    t = (err_text or "").lower()
    return (
        "not in the correct format" in t
        or "malformed" in t
        or "soundfile" in t
        or "valid audio file extension" in t
        or (("full address" in t) and ("download" in t))
    )


def _resample_mono_f32_inmemory(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Reamostra só com torchaudio (sem librosa; evita efeitos secundários de stack de áudio no path ASR)."""
    if int(orig_sr) == int(target_sr):
        return np.asarray(x, dtype=np.float32)
    t = torch.from_numpy(np.asarray(x, dtype=np.float32).copy()).unsqueeze(0)
    t = torchaudio.functional.resample(t, int(orig_sr), int(target_sr))
    return t.squeeze(0).numpy()


def _load_wav_stdlib_float_mono_native(p: str) -> tuple[np.ndarray, int]:
    """RIFF WAVE PCM 8/16/32-bit, mono float32, **sample rate do ficheiro** (sem reamostrar)."""
    with wave.open(p, "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if nch < 1 or nch > 2:
        raise ValueError(f"canais nao suportado: {nch}")
    if sw == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    elif sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"PCM sample width nao suportado: {sw}")
    if nch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32), int(sr)


def _load_wav_stdlib_float_mono(p: str) -> tuple[np.ndarray, int]:
    """
    RIFF WAVE com PCM (8/16/32-bit) — só stdlib, sem soundfile (útil no Windows com DLL/codec estranhos).
    Deprecado para leitura genérica: preferir _load_wav_stdlib_float_mono_native + resample.
    """
    x, sr = _load_wav_stdlib_float_mono_native(p)
    if int(sr) != 16000:
        _asr_transcribe_log(f"LOAD stdlib: resample {int(sr)} -> 16000")
        x = _resample_mono_f32_inmemory(x, int(sr), 16000)
        sr = 16000
    return x.astype(np.float32), int(sr)


def _load_mono_torchaudio_native(p: str) -> tuple[np.ndarray, int]:
    """torchaudio.read, mono float32, **taxa nativa** do ficheiro."""
    wav, sr = torchaudio.load(p)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    x = wav.cpu().numpy().astype(np.float32)
    return x, int(sr)


def _load_mono_16k_torchaudio_path(p: str) -> tuple[np.ndarray, int]:
    x, sr = _load_mono_torchaudio_native(p)
    if int(sr) != 16000:
        t = torch.from_numpy(x).unsqueeze(0)
        t = torchaudio.functional.resample(t, int(sr), 16000)
        x = t.squeeze(0).numpy()
        sr = 16000
    return x, int(sr)


def _read_wav_mono_native(path: str) -> tuple[np.ndarray, int]:
    """
    Lê ficheiro de áudio: soundfile -> stdlib wave -> torchaudio, mono float32, **taxa nativa**.
    Não chama librosa.load. Usado por ASR (16 kHz) e por finetune (ex.: 24 kHz).
    """
    import traceback

    p = os.path.abspath(os.path.normpath(path))
    _asr_transcribe_log(f"READ native: path={p!r}")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    if os.path.getsize(p) < 100:
        raise ValueError("WAV muito pequeno")
    # 1) soundfile
    try:
        info = sf.info(p)
        _asr_transcribe_log(
            f"READ: sf.info frames={info.frames} sr={info.samplerate} ch={getattr(info, 'channels', '?')}"
        )
        data, sr = sf.read(p, dtype="float32", always_2d=False)
        if hasattr(data, "ndim") and data.ndim == 2:
            data = np.mean(data, axis=1, dtype=np.float32)
        data = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if data.size < 8:
            raise ValueError("muito curto")
        n = int(data.shape[0])
        _asr_transcribe_log(f"READ: soundfile OK len={n} sr={sr}")
        return data, int(sr)
    except Exception as e:
        _asr_transcribe_log(f"READ: soundfile FAIL {type(e).__name__}: {e} — fallback")
        traceback.print_exc()
    # 2) stdlib
    try:
        y, srr = _load_wav_stdlib_float_mono_native(p)
        if len(y) < 8:
            raise ValueError("muito curto")
        _asr_transcribe_log(f"READ: stdlib OK len={len(y)} sr={srr}")
        return y, srr
    except Exception as e2:
        _asr_transcribe_log(f"READ: stdlib FAIL {type(e2).__name__}: {e2}")
        traceback.print_exc()
    # 3) torchaudio
    try:
        y, srr = _load_mono_torchaudio_native(p)
        if len(y) < 8:
            raise ValueError("muito curto")
        _asr_transcribe_log(f"READ: torchaudio OK len={len(y)} sr={srr}")
        return y, srr
    except Exception as e3:
        _asr_transcribe_log(f"READ: torchaudio FAIL {type(e3).__name__}: {e3}")
        traceback.print_exc()
    raise RuntimeError(
        "Nao foi possivel ler o ficheiro (soundfile, stdlib e torchaudio falharam). "
        "Re-exporta a PCM 16 bit ou reinstala soundfile/torchaudio."
    )


def load_wav_path_mono_f32(path: str, target_sample_rate: int) -> np.ndarray:
    """
    Carga única p/ pipeline Gradio (slicing a 24 k): **não** usar librosa.load no path
    (no Windows o mesmo ValueError falso acontece no passo 1, antes de segment_*.wav / ASR).
    """
    p = os.path.abspath(os.path.normpath(path))
    y, sr = _read_wav_mono_native(p)
    if int(sr) == int(target_sample_rate):
        return np.ascontiguousarray(y, dtype=np.float32)
    print(
        f"[F5TTS-LOAD] load_wav_path_mono_f32: resample {int(sr)} -> {int(target_sample_rate)} {p!r}",
        flush=True,
    )
    y = _resample_mono_f32_inmemory(y, int(sr), int(target_sample_rate))
    return np.ascontiguousarray(y, dtype=np.float32)


def _load_mono_16k_for_asr(path: str) -> tuple[np.ndarray, int]:
    """
    Carrega àudio para o Whisper a 16 kHz. NUNCA chamar librosa.load(ficheiro).
    """
    p = os.path.abspath(os.path.normpath(path))
    _asr_transcribe_log(f"LOAD ASR: path={p!r}")
    y, sr = _read_wav_mono_native(p)
    if len(y) < 32:
        raise ValueError(f"Audio muito curto: {len(y)} amostras")
    if int(sr) != 16000:
        _asr_transcribe_log(f"LOAD ASR: resample {int(sr)} -> 16000")
        y = _resample_mono_f32_inmemory(y, int(sr), 16000)
        sr = 16000
    return y, int(sr)


def transcribe(ref_audio, language=None):
    import traceback

    global asr_pipe
    if asr_pipe is None:
        _asr_transcribe_log("transcribe: a inicializar pipeline Whisper (pode demorar na 1.ª vez)…")
        initialize_asr_pipeline(device=device)

    # --- Paths locais: NUNCA passar str ao pipeline (evita open+ffmpeg_read e o ValueError falso "soundfile"). ---
    if isinstance(ref_audio, (str, os.PathLike)):
        raw_s = str(ref_audio)
        p = os.path.abspath(os.path.normpath(raw_s))
        _asr_transcribe_log(
            f"transcribe: entrada str/Path | raw={raw_s!r} | abspath={p!r} | isfile={os.path.isfile(p)}"
        )
        if raw_s.startswith("http://") or raw_s.startswith("https://"):
            # URL: o HF pipeline trata; não forçar dict
            ref_audio = raw_s
            _asr_transcribe_log("transcribe: URL — deixar pipeline tratar o URL")
        elif os.path.isfile(p):
            _asr_transcribe_log(
                "transcribe: ficheiro local — _load_mono_16k (soundfile → stdlib.wave → torchaudio; NÃO librosa.load no path)"
            )
            y, sr = _load_mono_16k_for_asr(p)
            y = np.ascontiguousarray(np.asarray(y, dtype=np.float32), dtype=np.float32)
            ref_audio = {"raw": y, "sampling_rate": int(sr)}
            _asr_transcribe_log(
                f"transcribe: dict pronto | raw len={len(y)} sr={sr} dtype={y.dtype}"
            )
        else:
            # Se isto acontecer, passar str ao pipeline quase certamente dá o erro falso "soundfile"
            msg = f"transcribe: ficheiro inexistente — não vou passar str ao ASR: {p!r}"
            _asr_transcribe_log(msg)
            raise FileNotFoundError(p)

    if isinstance(ref_audio, dict) and "raw" in ref_audio:
        r = ref_audio["raw"]
        if hasattr(r, "shape"):
            r = np.ascontiguousarray(np.asarray(r, dtype=np.float32), dtype=np.float32)
        ref_audio = {**ref_audio, "raw": r}

    gen_kwargs = {"task": "transcribe"}
    if language:
        lang = str(language).strip().lower()
        _lang_map = {
            "portuguese": "pt",
            "pt-br": "pt",
            "pt_br": "pt",
            "brazilian portuguese": "pt",
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "japanese": "ja",
            "korean": "ko",
            "chinese": "zh",
        }
        gen_kwargs["language"] = _lang_map.get(lang, lang)
    if isinstance(ref_audio, str) and not (
        ref_audio.startswith("http://") or ref_audio.startswith("https://")
    ):
        _asr_transcribe_log("transcribe: [BUG] ainda str local antes do pipeline — isto não devia acontecer")
        raise TypeError("ref_audio ainda path local em str; lógica acima deveria converter para dict")

    def _pipe_call(inp):
        return asr_pipe(
            inp,
            chunk_length_s=0,
            batch_size=1,
            generate_kwargs=gen_kwargs,
            return_timestamps=False,
        )["text"].strip()

    def _bypass_pipeline_generate(d: dict) -> str:
        """
        Não passa por AutomaticSpeechRecognitionPipeline.__call__ (dataloader/ffmpeg_read).
        Preferir `processor` (Whisper) alinhado com o modelo; depois `feature_extractor` + `model.generate`.
        Útil quando o pipeline levanta o falso "Soundfile" (ffmpeg_read).
        """
        y = np.ascontiguousarray(
            np.asarray(
                d.get("raw", d.get("array")),
                dtype=np.float32,
            ).reshape(-1)
        )
        sr = int(d["sampling_rate"])
        p = asr_pipe
        m = p.model
        proc = getattr(p, "processor", None)
        fe = p.feature_extractor
        with torch.inference_mode():
            input_features = None
            if proc is not None and callable(proc):
                try:
                    out = proc(
                        y,
                        sampling_rate=sr,
                        return_tensors="pt",
                    )
                    if hasattr(out, "get"):
                        input_features = out["input_features"]
                    elif hasattr(out, "input_features"):
                        input_features = out.input_features
                except Exception:  # noqa: BLE001
                    input_features = None
            if input_features is None and fe is not None:
                proc_out = fe(
                    y,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding="longest",
                    return_attention_mask=True,
                )
                if isinstance(proc_out, dict):
                    input_features = proc_out["input_features"]
                else:
                    input_features = getattr(proc_out, "input_features", None) or proc_out[0]
            if input_features is None:
                raise RuntimeError("bypass: sem input_features a partir de processor ou feature_extractor")
            if input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)
            input_features = input_features.to(device=m.device, dtype=torch.float32)

        lang = gen_kwargs.get("language")
        tsk = gen_kwargs.get("task", "transcribe")
        tok = getattr(p, "tokenizer", None)
        if tok is None and proc is not None:
            tok = getattr(proc, "tokenizer", None)
        gkw: dict = {"max_new_tokens": 444}
        pids = None
        if proc is not None and hasattr(proc, "get_decoder_prompt_ids"):
            try:
                pids = proc.get_decoder_prompt_ids(
                    language=lang,
                    task=tsk or "transcribe",
                )
            except Exception as e_pid:  # noqa: BLE001
                _asr_transcribe_log(
                    f"bypass: get_decoder_prompt_ids: {e_pid!r} — tento tokenizer ou task/language no generate"
                )
        if pids is None and proc is not None:
            toka = getattr(proc, "tokenizer", None)
            if toka is not None and hasattr(toka, "get_decoder_prompt_ids"):
                try:
                    pids = toka.get_decoder_prompt_ids(
                        language=lang,
                        task=tsk or "transcribe",
                    )
                except Exception:  # noqa: BLE001
                    pids = None
        if pids is not None:
            gkw["forced_decoder_ids"] = pids
        elif tsk is not None or lang is not None:
            if tsk is not None:
                gkw["task"] = tsk
            if lang is not None:
                gkw["language"] = lang
        with torch.inference_mode():
            try:
                out_ids = m.generate(input_features, **gkw)
            except (TypeError, ValueError) as e_gen:
                gkw2 = {k: v for k, v in gkw.items() if k not in ("task", "language", "forced_decoder_ids")}
                gkw2.setdefault("max_new_tokens", 444)
                _asr_transcribe_log(
                    f"bypass: model.generate c/kwargs {list(gkw)} falhou ({e_gen!r}) — retentar só {list(gkw2)}"
                )
                out_ids = m.generate(input_features, **gkw2)
        if tok is None:
            raise RuntimeError("Whisper: pipeline sem tokenizer para decode direto")
        return tok.decode(out_ids[0], skip_special_tokens=True).strip()

    try:
        _asr_transcribe_log(
            f"transcribe: chamar asr_pipe | type={type(ref_audio)} | "
            f"keys={list(ref_audio) if isinstance(ref_audio, dict) else 'n/a'}"
        )
        try:
            text = _pipe_call(ref_audio)
        except Exception as e0:
            # Algumas versões HF aceitam só "array" (datasets) em vez de "raw"
            if (
                isinstance(ref_audio, dict)
                and "raw" in ref_audio
                and "sampling_rate" in ref_audio
            ):
                _asr_transcribe_log(
                    f"transcribe: asr_pipe falhou ({e0!r}) — retentar com chave 'array' em vez de 'raw'"
                )
                alt = {
                    "array": ref_audio["raw"],
                    "sampling_rate": ref_audio["sampling_rate"],
                }
                text = _pipe_call(alt)
            else:
                raise
        _asr_transcribe_log(f"transcribe: asr_pipe OK | chars={len(text)}")
        return text
    except Exception as e:
        err = f"{e}"
        _asr_transcribe_log(f"transcribe: asr_pipe FATAL {type(e).__name__}: {e}")
        # Bypass total: a mensagem falsa "Soundfile" vem de audio_utils.ffmpeg_read dentro do __call__ do pipeline
        if (
            isinstance(ref_audio, dict)
            and "sampling_rate" in ref_audio
            and ("raw" in ref_audio or "array" in ref_audio)
            and _looks_like_hf_ffmpeg_read_error(err)
        ):
            _asr_transcribe_log("transcribe: fallback generate direto (feature_extractor + model) — evita o pipeline()")
            try:
                d = {
                    "raw": ref_audio.get("raw", ref_audio.get("array")),
                    "sampling_rate": int(ref_audio["sampling_rate"]),
                }
                text = _bypass_pipeline_generate(d)
                _asr_transcribe_log(
                    f"transcribe: bypass OK | chars={len(text)} (se vir isto, o pipeline() estava a falhar no host)"
                )
                return text
            except Exception as e2:  # noqa: BLE001
                _asr_transcribe_log(
                    f"transcribe: bypass FATAL {type(e2).__name__}: {e2!r} — a seguir re-lanço isto (não o falso 'Soundfile')"
                )
                traceback.print_exc()
                raise e2 from e
        if _looks_like_hf_ffmpeg_read_error(err):
            _asr_transcribe_log(
                "DICA: mensagem 'Soundfile' muitas vezes é de ffmpeg_read (HF), não do ficheiro. "
                "Foi tentado generate direto; se ainda falhar, vê o traceback. "
                "Reinicia a app; confirma [F5TTS-BOOT] e que não importas f5_tts antigo do site-packages."
            )
        traceback.print_exc()
        raise


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 6
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(ref_audio_orig, ref_text, clip_short=True, show_info=print, device=device):
    show_info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                    show_info("Audio is over 15s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 15000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                        show_info("Audio is over 15s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 15000:
                aseg = aseg[:15000]
                show_info("Audio is over 15s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    # Compute a hash of the reference audio file
    with open(ref_audio, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    if not ref_text.strip():
        global _ref_audio_cache
        if audio_hash in _ref_audio_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_audio_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_audio_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text)
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return infer_batch_process(
        (audio, sr),
        ref_text,
        gen_text_batches,
        model_obj,
        vocoder,
        mel_spec_type=mel_spec_type,
        progress=progress,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
        device=device,
    )


# infer batches


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    for i, gen_text in enumerate(progress.tqdm(gen_text_batches)):
        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated_mel_spec)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated_mel_spec)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            generated_waves.append(generated_wave)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )

            final_wave = new_wave

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)

    return final_wave, target_sample_rate, combined_spectrogram


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
