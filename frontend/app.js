
(() => {
  const toggleBtn = document.getElementById("toggleBtn");
  const voiceSelect = document.getElementById("voiceSelect");
  const emailInput = document.getElementById("emailInput");
  const statusPill = document.getElementById("statusPill");
  const chatLog = document.getElementById("chatLog");
  const eventsLog = document.getElementById("eventsLog");
  const errorBanner = document.getElementById("errorBanner");

  let ws = null;
  let audioCtx = null;
  let mediaStream = null;
  let workletNode = null;
  let muteNode = null;
  let running = false;

  let nextTime = 0;
  let activeSources = [];
  let currentChunk = null;
  let flushedGeneration = 0;
  // Keep a small playback cushion so network / TTS packet jitter does not
  // create gaps between PCM buffers (which sound like clipped syllables).
  const PLAYBACK_BUFFER_SEC = 0.2;

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.style.display = "block";
  }

  function clearError() {
    errorBanner.textContent = "";
    errorBanner.style.display = "none";
  }

  function setState(state) {
    statusPill.textContent = state;
    statusPill.className = "pill " + state;
  }

  function appendChat(role, text) {
    const div = document.createElement("div");
    div.className = "msg " + role;
    div.textContent =
      (role === "user" ? "You: " : "Agent: ") + text;

    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function appendEvent(ev, content, metadata) {
    const div = document.createElement("div");
    div.className = "event-item";

    div.textContent =
      `[${ev}] ${content || ""} ${
        metadata ? JSON.stringify(metadata) : ""
      }`;

    eventsLog.appendChild(div);
    eventsLog.scrollTop = eventsLog.scrollHeight;
  }

  function flushPlayback() {
    flushedGeneration += 1;

    for (const src of activeSources) {
      try {
        src.stop();
      } catch (_) {}
    }

    activeSources = [];
    currentChunk = null;
    nextTime = 0;
  }

  function handleBinaryAudio(arrayBuffer) {
    if (!audioCtx || !currentChunk) {
      console.warn("Received audio without an active TTS chunk", {
        audioContextState: audioCtx?.state,
        currentChunk,
        bytes: arrayBuffer.byteLength,
      });
      return;
    }

    if (audioCtx.state !== "running") {
      console.warn(
        "AudioContext is not running:",
        audioCtx.state
      );
      return;
    }

    const int16 = new Int16Array(arrayBuffer);

    if (int16.length === 0) return;

    const float32 = new Float32Array(int16.length);

    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768.0;
    }

    const buf = audioCtx.createBuffer(
      1,
      float32.length,
      currentChunk.rate
    );

    buf.getChannelData(0).set(float32);

    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);

    const now = audioCtx.currentTime;
    const startAt = nextTime > now
      ? nextTime
      : now + PLAYBACK_BUFFER_SEC;

    nextTime = startAt + buf.duration;

    activeSources.push(src);
    currentChunk.lastSource = src;
    currentChunk.hasAudio = true;

    src.onended = () => {
      const idx = activeSources.indexOf(src);

      if (idx !== -1) {
        activeSources.splice(idx, 1);
      }
    };

    src.start(startAt);
  }

  async function startSession() {
    clearError();

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });

      audioCtx = new AudioContext();

      await audioCtx.resume();

      console.log(
        "AudioContext state:",
        audioCtx.state,
        "Sample rate:",
        audioCtx.sampleRate
      );

      await audioCtx.audioWorklet.addModule(
        "/capture-worklet.js"
      );

      const source =
        audioCtx.createMediaStreamSource(mediaStream);

      workletNode = new AudioWorkletNode(
        audioCtx,
        "capture-worklet"
      );

      muteNode = audioCtx.createGain();
      muteNode.gain.value = 0;

      source.connect(workletNode);
      workletNode.connect(muteNode);
      muteNode.connect(audioCtx.destination);

      const proto =
        location.protocol === "https:" ? "wss:" : "ws:";

      ws = new WebSocket(
        `${proto}//${location.host}/ws/voice`
      );

      ws.binaryType = "arraybuffer";

      workletNode.port.onmessage = (e) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(e.data);
        }
      };

      ws.onopen = () => {
        console.log("Voice WebSocket connected");

        const emailVal = emailInput.value.trim();

        ws.send(
          JSON.stringify({
            type: "start",
            voice: voiceSelect.value,
            email: emailVal || null,
          })
        );
      };

      ws.onmessage = (event) => {
        if (typeof event.data !== "string") {
          handleBinaryAudio(event.data);
          return;
        }

        const msg = JSON.parse(event.data);

        switch (msg.type) {
          case "ready":
            console.log("Voice session ready:", msg);
            break;

          case "state":
            setState(msg.state);
            break;

          case "transcript":
            appendChat("user", msg.text);
            break;

          case "agent_text":
            appendChat("agent", msg.text);
            break;

          case "agent_event":
            appendEvent(
              msg.event,
              msg.content,
              msg.metadata
            );
            break;

          case "tts_chunk_start":
            console.log("TTS started:", msg);

            currentChunk = {
              id: msg.chunk_id,
              rate: msg.sample_rate,
              lastSource: null,
              hasAudio: false,
            };
            break;

          case "tts_chunk_end":
            console.log("TTS ended:", msg.chunk_id);

            if (
              currentChunk &&
              currentChunk.id === msg.chunk_id
            ) {
              currentChunk = null;
            }
            break;

          case "barge_in":
            console.log("Barge-in detected");
            flushPlayback();
            break;

          case "error":
            console.error("Server error:", msg);
            showError(
              `${msg.code}: ${msg.message}`
            );
            break;

          default:
            console.log("Unknown server message:", msg);
        }
      };

      ws.onerror = (event) => {
        console.error("WebSocket error:", event);
        showError("WebSocket error.");
      };

      ws.onclose = (event) => {
        console.log(
          "WebSocket closed:",
          event.code,
          event.reason
        );

        if (running) {
          showError("Connection closed.");
          stopSession(false);
        }
      };

      running = true;

      toggleBtn.textContent = "Stop";
      toggleBtn.className = "";

    } catch (err) {
      console.error("Failed to start voice session:", err);

      showError(err.message || String(err));

      stopSession(false);
    }
  }

  function stopSession(sendStop = true) {
    running = false;

    toggleBtn.textContent = "Start";
    toggleBtn.className = "";

    setState("idle");
    flushPlayback();

    if (ws) {
      if (
        sendStop &&
        ws.readyState === WebSocket.OPEN
      ) {
        try {
          ws.send(
            JSON.stringify({ type: "stop" })
          );
        } catch (_) {}
      }

      try {
        ws.close();
      } catch (_) {}

      ws = null;
    }

    if (workletNode) {
      try {
        workletNode.disconnect();
      } catch (_) {}

      workletNode = null;
    }

    if (muteNode) {
      try {
        muteNode.disconnect();
      } catch (_) {}

      muteNode = null;
    }

    if (mediaStream) {
      for (const track of mediaStream.getTracks()) {
        track.stop();
      }

      mediaStream = null;
    }

    if (audioCtx) {
      try {
        audioCtx.close();
      } catch (_) {}

      audioCtx = null;
    }
  }

  toggleBtn.addEventListener("click", () => {
    if (running) {
      stopSession(true);
    } else {
      startSession();
    }
  });

  voiceSelect.addEventListener("change", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "voice",
          voice: voiceSelect.value,
        })
      );
    }
  });
})();
