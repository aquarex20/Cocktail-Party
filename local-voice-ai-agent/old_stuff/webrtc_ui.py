"""
WebRTC voice chat UI: hidden block + JS to start stream and record from the Start session button.
"""

import gradio as gr
from fastrtc import ReplyOnPause, WebRTC

from voice_config import SYSTEM_DEFAULT_LABEL


def add_voice_chat_block(echo_fn, start_session_fn, start_btn, input_dropdown, session_status):
    """
    Add the WebRTC column and wire Start session to pre-configure mic and click start/record.
    Call this inside your gr.Blocks() context, after start_btn, input_dropdown, session_status exist.
    """
    with gr.Column(visible=True):
        audio = WebRTC(
            mode="send-receive",
            modality="audio",
            elem_id="webrtc_voice",
        )
        audio.stream(fn=ReplyOnPause(echo_fn), inputs=[audio], outputs=[audio])

    _sys_default = SYSTEM_DEFAULT_LABEL.replace('"', '\\"')
    _js_start_webrtc_and_mic = f"""
    (selectedInputName) => {{
        const SYS_DEFAULT = "{_sys_default}";
        function clickWebRTCButtons() {{
            const root = document.querySelector('[id^="webrtc_voice"]') || document.getElementById('webrtc_voice');
            if (!root) return;
            const startBtn = root.querySelector('button[aria-label="start stream"]') || root.querySelector('button');
            if (startBtn) startBtn.click();
            setTimeout(() => {{
                const recordBtn = document.evaluate(
                    "//*[starts-with(@id, \\"webrtc_voice\\")]/div[2]/div[2]/button[1]",
                    document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null
                ).singleNodeValue;
                if (recordBtn) recordBtn.click();
            }}, 600);
        }}
        function doStart() {{
            clickWebRTCButtons();
            return [selectedInputName != null ? selectedInputName : ""];
        }}
        if (!selectedInputName || selectedInputName === SYS_DEFAULT)
            return doStart();
        return navigator.mediaDevices.enumerateDevices()
            .then(devices => {{
                const audioInputs = devices.filter(d => d.kind === "audioinput");
                const match = audioInputs.find(d => d.label && (
                    d.label === selectedInputName ||
                    d.label.includes(selectedInputName) ||
                    selectedInputName.includes(d.label)
                ));
                if (match && match.deviceId) {{
                    const orig = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                    navigator.mediaDevices.getUserMedia = function(constraints) {{
                        if (constraints && constraints.audio) {{
                            const c = constraints.audio === true ? {{}} : (typeof constraints.audio === "object" ? {{ ...constraints.audio }} : {{}});
                            c.deviceId = {{ exact: match.deviceId }};
                            constraints = {{ ...constraints, audio: c }};
                        }}
                        return orig(constraints);
                    }};
                }}
                return doStart();
            }})
            .catch(() => doStart());
    }}
    """
    start_btn.click(
        fn=start_session_fn,
        inputs=[input_dropdown],
        outputs=[session_status],
        js=_js_start_webrtc_and_mic,
    )
