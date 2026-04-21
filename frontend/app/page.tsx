"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const WS_URL = "ws://localhost:8000/ws";
const FRAME_INTERVAL = 500;
const HOLD_REQUIRED = 3;

type Result = {
  prediction: string;
  confidence: number;
  landmarks: { x: number; y: number }[];
  hand_detected: boolean;
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const holdCountRef = useRef(0);
  const holdCandidateRef = useRef("");
  const lastCommittedRef = useRef("");

  const [connected, setConnected] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [sentence, setSentence] = useState("");
  const [lastLetter, setLastLetter] = useState("");
  const [copied, setCopied] = useState(false);
  const [streaming, setStreaming] = useState(false);

  const commitPrediction = (prediction: string) => {
    setSentence((s) => {
      if (prediction === "space") return s + " ";
      if (prediction === "del") return s.slice(0, -1);
      if (prediction === "nothing") return s;
      return s + prediction;
    });
    setLastLetter(prediction);
    lastCommittedRef.current = prediction;
  };

  const resetHoldState = (resetCommitted = false) => {
    holdCountRef.current = 0;
    holdCandidateRef.current = "";
    if (resetCommitted) {
      lastCommittedRef.current = "";
      setLastLetter("");
    }
  };

  const connectWS = useCallback(() => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => setConnected(true);

    ws.onclose = () => {
      setConnected(false);
      setTimeout(connectWS, 2000);
    };

    ws.onmessage = (e) => {
      const data: Result = JSON.parse(e.data);
      setResult(data);

      if (!data.hand_detected) {
        resetHoldState(true);
      } else if (!data.prediction || data.confidence <= 70) {
        resetHoldState(false);
      } else if (data.prediction === "nothing") {
        resetHoldState(true);
      } else {
        if (data.prediction !== holdCandidateRef.current) {
          holdCandidateRef.current = data.prediction;
          holdCountRef.current = 1;
        } else {
          holdCountRef.current += 1;

          if (holdCountRef.current >= HOLD_REQUIRED) {
            if (data.prediction !== lastCommittedRef.current) {
              commitPrediction(data.prediction);
            }
            holdCountRef.current = 0;
          }
        }
      }

      if (overlayRef.current && videoRef.current) {
        const ctx = overlayRef.current.getContext("2d")!;
        ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);

        if (data.landmarks.length > 0) {
          const w = overlayRef.current.width;
          const h = overlayRef.current.height;

          ctx.fillStyle = "#00ff88";
          ctx.shadowColor = "#00ff88";
          ctx.shadowBlur = 8;

          data.landmarks.forEach((lm) => {
            ctx.beginPath();
            ctx.arc(lm.x * w, lm.y * h, 5, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connectWS();

    return () => {
      wsRef.current?.close();
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [connectWS]);

  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play();
      setStreaming(true);
    }

    intervalRef.current = setInterval(() => {
      if (!canvasRef.current || !videoRef.current || !wsRef.current || wsRef.current.readyState !== 1) return;

      const ctx = canvasRef.current.getContext("2d")!;
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;

      if (overlayRef.current) {
        overlayRef.current.width = videoRef.current.videoWidth;
        overlayRef.current.height = videoRef.current.videoHeight;
      }

      ctx.drawImage(videoRef.current, 0, 0);

      canvasRef.current.toBlob((blob) => {
        if (!blob) return;

        const reader = new FileReader();
        reader.onload = () => {
          const b64 = (reader.result as string).split(",")[1];
          wsRef.current?.send(JSON.stringify({ frame: b64 }));
        };
        reader.readAsDataURL(blob);
      }, "image/jpeg", 0.7);
    }, FRAME_INTERVAL);
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(sentence);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const clearSentence = () => {
    setSentence("");
    resetHoldState(true);
  };

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white font-mono">
      <header className="border-b border-white/10 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-[#00ff88] animate-pulse" />
          <span className="text-lg font-bold tracking-widest uppercase">ASL Translator</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <div className={`w-2 h-2 rounded-full ${connected ? "bg-[#00ff88]" : "bg-red-500"}`} />
          <span className="text-white/50">{connected ? "Backend connected" : "Connecting..."}</span>
        </div>
      </header>

      <div className="flex flex-col lg:flex-row h-[calc(100vh-65px)]">
        <div className="flex-1 relative bg-black flex items-center justify-center overflow-hidden">
          <video ref={videoRef} className="w-full h-full object-cover" muted playsInline />
          <canvas ref={overlayRef} className="absolute inset-0 w-full h-full" />
          <canvas ref={canvasRef} className="hidden" />

          {!streaming && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-6 bg-black/80">
              <div className="text-6xl">✋</div>
              <p className="text-white/50 text-sm tracking-widest uppercase">Camera not started</p>
              <button
                onClick={startCamera}
                className="px-8 py-3 bg-[#00ff88] text-black font-bold tracking-widest uppercase text-sm hover:bg-white transition-colors"
              >
                Start Camera
              </button>
            </div>
          )}

          {streaming && result && !result.hand_detected && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/70 border border-white/20 px-4 py-2 text-sm text-white/50 tracking-widest">
              NO HAND DETECTED
            </div>
          )}
        </div>

        <div className="w-full lg:w-[380px] border-l border-white/10 flex flex-col">
          <div className="border-b border-white/10 p-6 flex flex-col gap-2">
            <span className="text-xs text-white/30 tracking-widest uppercase">Detected Sign</span>
            <div className="flex items-end gap-4">
              <span className="text-8xl font-bold text-[#00ff88]">
                {result?.prediction?.toUpperCase() || "—"}
              </span>
            </div>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#00ff88] transition-all duration-300"
                  style={{ width: `${result?.confidence ?? 0}%` }}
                />
              </div>
              <span className="text-xs text-white/40 w-12 text-right">
                {result?.confidence?.toFixed(0) ?? 0}%
              </span>
            </div>
          </div>

          <div className="flex-1 p-6 flex flex-col gap-4">
            <span className="text-xs text-white/30 tracking-widest uppercase">Sentence</span>
            <div className="flex-1 bg-white/5 border border-white/10 p-4 text-lg leading-relaxed break-words min-h-[120px]">
              {sentence || <span className="text-white/20">Start signing...</span>}
              <span className="animate-pulse">|</span>
            </div>

            <div className="flex gap-2">
              <button
                onClick={copyToClipboard}
                disabled={!sentence}
                className="flex-1 py-2 border border-white/20 text-sm tracking-widest uppercase hover:bg-white/10 transition-colors disabled:opacity-30"
              >
                {copied ? "Copied!" : "Copy"}
              </button>
              <button
                onClick={clearSentence}
                disabled={!sentence}
                className="flex-1 py-2 border border-red-500/40 text-red-400 text-sm tracking-widest uppercase hover:bg-red-500/10 transition-colors disabled:opacity-30"
              >
                Clear
              </button>
            </div>
          </div>

          <div className="border-t border-white/10 p-6">
            <span className="text-xs text-white/30 tracking-widest uppercase block mb-3">How to use</span>
            <ul className="text-xs text-white/40 space-y-1">
              <li>→ Hold each sign steady for about 3 detections</li>
              <li>→ Move to neutral or remove hand to repeat the same letter</li>
              <li>→ Sign "space" to add a space</li>
              <li>→ Sign "del" to delete last letter</li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  );
}