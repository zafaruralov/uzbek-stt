import sys
import time
import html as html_module

sys.path.insert(0, "/app/stt")

from stt_service_new import UzbekSTTService
import gradio as gr

import os

MODEL = os.environ.get("STT_MODEL", "RubaiLab/kotib_call_base_stt_15")
print(f"Loading Uzbek STT model: {MODEL}")
stt = UzbekSTTService(model_name=MODEL, offline_mode=True)
print("Model ready!")


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def format_time_precise(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:05.2f}"


def transcribe(audio_path):
    if audio_path is None:
        return (
            '<div class="placeholder">Audio fayl yuklang yoki mikrofon orqali yozing.</div>',
            "",
            "",
        )

    start = time.time()
    result = stt.transcribe_with_timestamps(audio_path)
    elapsed = time.time() - start

    if not result["success"]:
        err = html_module.escape(result.get("error", "Noma'lum xatolik"))
        return (
            f'<div class="error">Xatolik: {err}</div>',
            "",
            "",
        )

    segments = result.get("segments", [])
    plain_text = result.get("text", "")

    # Build HTML transcript
    if segments:
        spans = []
        for seg in segments:
            t = html_module.escape(seg["text"])
            s = f'{seg["start"]:.2f}'
            e = f'{seg["end"]:.2f}'
            ts_label = format_time(seg["start"])
            spans.append(
                f'<span class="seg" data-start="{s}" data-end="{e}">'
                f'<span class="ts">[{ts_label}]</span> {t}</span>'
            )
        dur = f'{result["duration"]:.2f}'
        transcript_html = (
            '<div id="now-playing"></div>'
            f'<div id="transcript" data-duration="{dur}">{"".join(spans)}</div>'
        )
    else:
        transcript_html = (
            f'<div id="transcript"><span class="seg">'
            f'{html_module.escape(plain_text)}</span></div>'
        )

    stats = (
        f"Davomiyligi: **{result['duration']:.1f}s** | "
        f"Bo'laklar: **{result['chunks']}** | "
        f"Segmentlar: **{len(segments)}** | "
        f"Ishlov vaqti: **{elapsed:.1f}s**"
    )

    return transcript_html, plain_text, stats


# --------------- CSS ---------------
custom_css = """
/* ---- Now-playing bar ---- */
#now-playing {
    display: none;                 /* hidden until audio plays */
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    margin-bottom: 8px;
    border-radius: 8px;
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: #fff;
    font-size: 0.95em;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}
#now-playing.visible { display: flex; }
#now-playing .np-time { font-family: monospace; font-size: 1.1em; }
#now-playing .np-label { opacity: 0.8; font-size: 0.85em; }

/* ---- Transcript container ---- */
#transcript {
    max-height: 450px;
    overflow-y: auto;
    scroll-behavior: smooth;
    padding: 16px;
    line-height: 2.1;
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    background: var(--background-fill-primary);
}

/* ---- Segment spans ---- */
.seg {
    display: inline;
    padding: 4px 7px;
    margin: 2px 1px;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s, box-shadow 0.15s;
}
.seg:hover {
    background: rgba(59,130,246,0.12);
}
.seg.active {
    background: #3b82f6;
    color: #fff;
    font-weight: 600;
    box-shadow: 0 1px 4px rgba(59,130,246,0.35);
}

/* ---- Timestamp badge ---- */
.seg .ts {
    font-family: monospace;
    font-size: 0.78em;
    font-weight: 600;
    color: #6b7280;
    background: rgba(107,114,128,0.10);
    padding: 1px 5px;
    border-radius: 4px;
    margin-right: 3px;
}
.seg:hover .ts {
    color: #3b82f6;
    background: rgba(59,130,246,0.10);
}
.seg.active .ts {
    color: #dbeafe;
    background: rgba(255,255,255,0.18);
}

/* ---- Misc ---- */
.placeholder {
    padding: 40px 20px;
    text-align: center;
    color: var(--body-text-color-subdued);
    font-size: 1.05em;
}
.error {
    padding: 16px;
    color: #dc2626;
    font-weight: 600;
}
#stats-row {
    font-size: 0.88em;
    opacity: 0.7;
    margin-top: 4px;
}

/* ---- Model badge ---- */
#model-badge-wrap {
    text-align: right;
    display: flex;
    align-items: center;
    justify-content: flex-end;
}
#model-badge {
    display: inline-block;
    font-family: monospace;
    font-size: 0.82em;
    font-weight: 600;
    color: #6b7280;
    background: rgba(107,114,128,0.10);
    border: 1px solid rgba(107,114,128,0.20);
    padding: 4px 10px;
    border-radius: 6px;
    white-space: nowrap;
    text-decoration: none;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
}
a#model-badge:hover {
    color: #3b82f6;
    border-color: rgba(59,130,246,0.35);
    background: rgba(59,130,246,0.08);
}

/* ---- Fix audio player overflow in narrow column ---- */
#audio-col {
    overflow: hidden;
}
#audio-col div,
#audio-col audio {
    max-width: 100% !important;
    box-sizing: border-box !important;
}
#audio-col .wrap, #audio-col .waveform-container,
#audio-col [data-testid] {
    overflow: hidden !important;
}
"""

# --------------- JS ---------------
custom_js = """
(function() {
    let _audio = null;
    let _wrapper = null;

    /* ---- Traverse shadow roots to find elements ---- */
    function deepQuery(root, selector) {
        if (!root) return null;
        const el = root.querySelector(selector);
        if (el) return el;
        for (const node of root.querySelectorAll('*')) {
            if (node.shadowRoot) {
                const found = deepQuery(node.shadowRoot, selector);
                if (found) return found;
            }
        }
        return null;
    }

    function getAudio() {
        const a = deepQuery(document, 'audio');
        if (a && a !== _audio) attachAudio(a);
        return _audio;
    }

    /* Find WaveSurfer .wrapper inside #audio-col shadow roots */
    function getWrapper() {
        if (_wrapper && _wrapper.isConnected) return _wrapper;
        const col = document.getElementById('audio-col');
        if (!col) return null;
        _wrapper = deepQuery(col, '.wrapper');
        if (_wrapper) console.log('[STT] found waveform .wrapper', _wrapper);
        return _wrapper;
    }

    function fmtTime(sec) {
        if (!sec || isNaN(sec)) return '0:00.00';
        const m = Math.floor(sec / 60);
        const s = sec - m * 60;
        return m + ':' + ('0' + s.toFixed(2)).slice(-5);
    }

    /* ---- Sync transcript highlight + now-playing bar ---- */
    function onTimeUpdate() {
        if (!_audio) return;
        const t = _audio.currentTime;
        const dur = _audio.duration || 0;

        const np = document.getElementById('now-playing');
        if (np) {
            np.classList.add('visible');
            np.innerHTML =
                '<span class="np-label">Now playing</span>' +
                '<span class="np-time">' + fmtTime(t) + ' / ' + fmtTime(dur) + '</span>';
        }

        document.querySelectorAll('.seg').forEach(el => {
            const s = parseFloat(el.dataset.start);
            const e = parseFloat(el.dataset.end);
            const isActive = !isNaN(s) && !isNaN(e) && t >= s && t < e;
            el.classList.toggle('active', isActive);
            if (isActive) {
                const container = document.getElementById('transcript');
                if (container) {
                    const elTop = el.offsetTop - container.offsetTop;
                    const elBot = elTop + el.offsetHeight;
                    const viewTop = container.scrollTop;
                    const viewBot = viewTop + container.clientHeight;
                    if (elTop < viewTop || elBot > viewBot) {
                        container.scrollTop = elTop - container.clientHeight / 3;
                    }
                }
            }
        });
    }

    function onPause() {
        if (!_audio) return;
        const np = document.getElementById('now-playing');
        if (np && _audio.paused && !_audio.seeking) {
            np.innerHTML =
                '<span class="np-label">Paused</span>' +
                '<span class="np-time">' + fmtTime(_audio.currentTime) +
                ' / ' + fmtTime(_audio.duration || 0) + '</span>';
        }
    }

    function attachAudio(audio) {
        if (_audio === audio) return;
        if (_audio) {
            _audio.removeEventListener('timeupdate', onTimeUpdate);
            _audio.removeEventListener('pause', onPause);
        }
        _audio = audio;
        _audio.addEventListener('timeupdate', onTimeUpdate);
        _audio.addEventListener('pause', onPause);
        console.log('[STT] attached to <audio>', audio);
    }

    /* ---- Read actual duration from WaveSurfer's time display ---- */
    let _wsDuration = 0;

    function parseTimeStr(str) {
        /* "1:23" -> 83, "0:05" -> 5, "10:05" -> 605 */
        const parts = str.split(':').map(Number);
        if (parts.length === 2) return parts[0] * 60 + parts[1];
        if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
        return 0;
    }

    function scanPlayerDuration() {
        /* Scan #audio-col for text matching m:ss time pattern — the largest
           value is the total duration as displayed by WaveSurfer. */
        const col = document.getElementById('audio-col');
        if (!col) return 0;
        const walker = document.createTreeWalker(col, NodeFilter.SHOW_TEXT);
        let best = 0;
        while (walker.nextNode()) {
            const txt = walker.currentNode.textContent.trim();
            const m = txt.match(/^(\d{1,2}:\d{2})$/);
            if (m) {
                const sec = parseTimeStr(m[1]);
                if (sec > best) best = sec;
            }
        }
        /* Also scan shadow roots */
        col.querySelectorAll('*').forEach(el => {
            if (!el.shadowRoot) return;
            const sw = document.createTreeWalker(el.shadowRoot, NodeFilter.SHOW_TEXT);
            while (sw.nextNode()) {
                const txt = sw.currentNode.textContent.trim();
                const m = txt.match(/^(\d{1,2}:\d{2})$/);
                if (m) {
                    const sec = parseTimeStr(m[1]);
                    if (sec > best) best = sec;
                }
            }
        });
        if (best > 0 && best !== _wsDuration) {
            _wsDuration = best;
            console.log('[STT] detected player duration:', _wsDuration, 's');
        }
        return best;
    }

    function getDuration() {
        /* 1) from WaveSurfer player UI (most accurate) */
        const ws = scanPlayerDuration();
        if (ws > 0) return ws;
        /* 2) from transcript data attribute (fallback) */
        const tc = document.getElementById('transcript');
        if (tc && tc.dataset.duration) {
            const d = parseFloat(tc.dataset.duration);
            if (d > 0) return d;
        }
        return 0;
    }

    /* ---- Seek by simulating a click on the WaveSurfer waveform ---- */
    function seekTo(seconds) {
        const duration = getDuration();
        if (!duration) {
            console.warn('[STT] cannot seek: no duration');
            return;
        }
        const wrapper = getWrapper();
        if (wrapper) {
            const progress = Math.max(0, Math.min(1, seconds / duration));
            const rect = wrapper.getBoundingClientRect();
            const clientX = rect.left + progress * rect.width;
            const clientY = rect.top + rect.height / 2;
            console.log('[STT] seeking', seconds.toFixed(1) + 's, duration=' +
                duration + 's, progress=' + progress.toFixed(4));
            wrapper.dispatchEvent(new PointerEvent('click', {
                clientX: clientX,
                clientY: clientY,
                bubbles: true,
                composed: true,
            }));
            return;
        }
        console.warn('[STT] no waveform wrapper found');
    }

    /* ---- MutationObserver: re-scan whenever DOM changes ---- */
    const obs = new MutationObserver(() => {
        getAudio();
        getWrapper();
    });
    obs.observe(document.body, { childList: true, subtree: true });
    getAudio();

    /* ---- Click-to-seek on transcript segments ---- */
    document.addEventListener('click', (e) => {
        const seg = e.target.closest('.seg');
        if (!seg || seg.dataset.start === undefined) return;
        const time = parseFloat(seg.dataset.start);
        console.log('[STT] segment clicked, seeking to', time);
        seekTo(time);
    });
})();
"""

# --------------- Model display name ---------------
# STT_MODEL_HF = HuggingFace model ID for display/link (e.g. "aisha-org/Whisper-Uzbek")
# Falls back to STT_MODEL if it looks like a HF ID (org/name), otherwise shows raw path.
_model_hf = os.environ.get("STT_MODEL_HF", "")
if not _model_hf:
    # Auto-detect: if MODEL looks like "org/name" (no leading /), treat as HF ID
    if "/" in MODEL and not MODEL.startswith("/"):
        _model_hf = MODEL.rstrip("/")

if _model_hf:
    _hf_url = f"https://huggingface.co/{_model_hf}"
    _badge_html = f'<a id="model-badge" href="{_hf_url}" target="_blank">{_model_hf}</a>'
else:
    _badge_html = f'<span id="model-badge">{MODEL}</span>'

# --------------- Layout ---------------
with gr.Blocks(title="Uzbek Speech-to-Text", css=custom_css, js=custom_js) as demo:
    with gr.Row():
        gr.Markdown("# Uzbek Speech-to-Text")
        gr.Markdown(
            _badge_html,
            elem_id="model-badge-wrap",
        )
    gr.Markdown(
        "Audio faylni yuklang yoki mikrofon orqali yozing — "
        "transkripsiya avtomatik boshlanadi. "
        "Segmentni bosing — audio o'sha joydan o'ynaydi."
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=280, elem_id="audio-col"):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio",
            )
            transcribe_btn = gr.Button(
                "Qayta transkripsiya",
                variant="secondary",
                size="sm",
                interactive=False,
            )

        with gr.Column(scale=2):
            transcript_html = gr.HTML(
                value='<div class="placeholder">Audio fayl yuklang yoki mikrofon orqali yozing.</div>',
                label="Natija",
            )
            stats_md = gr.Markdown("", elem_id="stats-row")
            with gr.Accordion("Oddiy matn nusxasi", open=False):
                plain_text = gr.Textbox(
                    label="Matn",
                    lines=4,
                    interactive=False,
                )

    # Auto-transcribe when audio is uploaded / recorded
    audio_input.change(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[transcript_html, plain_text, stats_md],
    ).then(
        fn=lambda: gr.update(interactive=True),
        outputs=[transcribe_btn],
    )

    # Manual re-transcribe (enabled only after first transcription)
    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[transcript_html, plain_text, stats_md],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
