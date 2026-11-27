class CaptureWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._targetRate = 16000;
    this._pos = 0;
    this._prevSample = 0;
    this._hasPrev = false;
    this._out = new Int16Array(512);
    this._outIdx = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }
    const channel = input[0];
    const ratio = sampleRate / this._targetRate;

    if (!this._hasPrev) {
      this._prevSample = channel[0];
      this._hasPrev = true;
    }

    while (this._pos < channel.length) {
      const idx = Math.floor(this._pos);
      const frac = this._pos - idx;
      const s0 = idx === 0 ? this._prevSample : channel[idx - 1];
      const s1 = channel[idx];
      const val = s0 + (s1 - s0) * frac;

      const clamped = Math.max(-1, Math.min(1, val));
      this._out[this._outIdx++] = clamped < 0 ? clamped * 32768 : clamped * 32767;

      if (this._outIdx === 512) {
        const buf = this._out.buffer.slice(0);
        this.port.postMessage(buf, [buf]);
        this._outIdx = 0;
      }
      this._pos += ratio;
    }

    this._pos -= channel.length;
    this._prevSample = channel[channel.length - 1];
    return true;
  }
}

registerProcessor("capture-worklet", CaptureWorkletProcessor);
