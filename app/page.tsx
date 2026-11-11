"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Upload,
  Activity,
  Signal,
  Wifi,
  WifiOff,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Send,
  Copy,
  Check,
  FileText,
  RefreshCw,
  Info,
  Layers,
  TrendingDown,
  Globe,
  Database,
  Trash2,
  Volume2
} from "lucide-react";
import { compressInference, InferenceResult } from "@/lib/securePayload";
import { runInference } from "@/lib/inference";
import { useTranslation } from "@/lib/i18n";
import {
  QueuedPayload,
  queuePayload,
  getQueuedPayloads,
  clearQueue,
  startQueueSync
} from "@/lib/offlineQueue";

// Helper to generate a random Patient ID
const generatePatientId = () => {
  return `PID-${Math.floor(10000 + Math.random() * 90000)}`;
};

/**
 * Generates a synthetic Chest X-ray image using HTML5 Canvas.
 * This ensures the application works offline and comes preloaded with test samples
 * that are fully processed by our inference preprocessor.
 */
function drawSampleXray(type: "normal" | "pneumonia"): string {
  if (typeof window === "undefined") return "";

  const canvas = document.createElement("canvas");
  canvas.width = 400;
  canvas.height = 400;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  // 1. Dark background representing raw lung cavity
  ctx.fillStyle = "#0c0d12";
  ctx.fillRect(0, 0, 400, 400);

  // 2. Thoracic spine column (midline vertical bone density)
  const spineGrad = ctx.createLinearGradient(200, 0, 200, 400);
  spineGrad.addColorStop(0, "#12141a");
  spineGrad.addColorStop(0.5, "#303644");
  spineGrad.addColorStop(1, "#12141a");
  ctx.fillStyle = spineGrad;
  ctx.fillRect(192, 10, 16, 380);

  // 3. Clavicles (collar bones)
  ctx.strokeStyle = "#434b5e";
  ctx.lineWidth = 10;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(80, 50);
  ctx.quadraticCurveTo(150, 75, 190, 75);
  ctx.moveTo(320, 50);
  ctx.quadraticCurveTo(250, 75, 210, 75);
  ctx.stroke();

  // 4. Lungs lobes (dark air cavities)
  ctx.fillStyle = "#07080a";
  // Left Lung
  ctx.beginPath();
  ctx.moveTo(90, 80);
  ctx.bezierCurveTo(140, 70, 180, 100, 180, 290);
  ctx.bezierCurveTo(180, 320, 115, 320, 85, 290);
  ctx.bezierCurveTo(55, 230, 55, 120, 90, 80);
  ctx.fill();

  // Right Lung
  ctx.beginPath();
  ctx.moveTo(310, 80);
  ctx.bezierCurveTo(260, 70, 220, 100, 220, 290);
  ctx.bezierCurveTo(220, 320, 285, 320, 315, 290);
  ctx.bezierCurveTo(345, 230, 345, 120, 310, 80);
  ctx.fill();

  // 5. Rib cage outlines (semi-translucent arching bones)
  ctx.strokeStyle = "rgba(67, 75, 94, 0.4)";
  ctx.lineWidth = 5;
  for (let i = 0; i < 7; i++) {
    const y = 95 + i * 35;
    // Left Ribs
    ctx.beginPath();
    ctx.arc(100, y, 90, Math.PI * 1.15, Math.PI * 1.85);
    ctx.stroke();
    // Right Ribs
    ctx.beginPath();
    ctx.arc(300, y, 90, Math.PI * 1.15, Math.PI * 1.85);
    ctx.stroke();
  }

  // 6. Cardiac silhouette (heart shadow in lower-left-center chest)
  ctx.fillStyle = "#1b1e26";
  ctx.beginPath();
  ctx.moveTo(175, 210);
  ctx.bezierCurveTo(175, 280, 130, 310, 205, 310);
  ctx.bezierCurveTo(235, 310, 230, 260, 200, 210);
  ctx.closePath();
  ctx.fill();

  // 7. Diaphragm dome structure (bottom boundary)
  ctx.fillStyle = "#1e222c";
  ctx.beginPath();
  ctx.moveTo(30, 400);
  ctx.quadraticCurveTo(110, 310, 195, 320);
  ctx.quadraticCurveTo(280, 310, 370, 400);
  ctx.closePath();
  ctx.fill();

  // 8. If Pneumonia, inject focal consolidated opacities (white infiltrate clouds)
  if (type === "pneumonia") {
    // Primary consolidation in right lower lung field
    const pneuGrad1 = ctx.createRadialGradient(280, 240, 5, 280, 240, 50);
    pneuGrad1.addColorStop(0, "rgba(240, 245, 255, 0.8)");
    pneuGrad1.addColorStop(0.3, "rgba(220, 230, 250, 0.5)");
    pneuGrad1.addColorStop(0.7, "rgba(200, 210, 240, 0.2)");
    pneuGrad1.addColorStop(1, "rgba(200, 210, 240, 0)");
    ctx.fillStyle = pneuGrad1;
    ctx.beginPath();
    ctx.arc(280, 240, 50, 0, Math.PI * 2);
    ctx.fill();

    // Secondary smaller consolidation in left middle lung field
    const pneuGrad2 = ctx.createRadialGradient(110, 200, 2, 110, 200, 30);
    pneuGrad2.addColorStop(0, "rgba(240, 245, 255, 0.55)");
    pneuGrad2.addColorStop(0.4, "rgba(220, 230, 250, 0.3)");
    pneuGrad2.addColorStop(1, "rgba(200, 210, 240, 0)");
    ctx.fillStyle = pneuGrad2;
    ctx.beginPath();
    ctx.arc(110, 200, 30, 0, Math.PI * 2);
    ctx.fill();
  }

  return canvas.toDataURL("image/png");
}

export default function EdgeTriageDashboard() {
  const { locale, setLocale, t } = useTranslation();

  // Patient Details
  const [patientId, setPatientId] = useState("");
  
  // Image states
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  
  // Inference states
  const [isInferenceRunning, setIsInferenceRunning] = useState(false);
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null);
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);
  const [modelType, setModelType] = useState<"ONNX-WebAssembly" | "Client-Simulation">("Client-Simulation");

  // Compression states
  const [compressedPacket, setCompressedPacket] = useState<string>("");
  const [isCopied, setIsCopied] = useState(false);

  // Network Simulation states
  const [isOnline, setIsOnline] = useState(true); // Toggle to simulate GSM cell signal availability
  const [isTransmitting, setIsTransmitting] = useState(false);
  const [transmissionLogs, setTransmissionLogs] = useState<string[]>([]);
  const [transmissionSuccess, setTransmissionSuccess] = useState<boolean | null>(null);
  const [serverMessageId, setServerMessageId] = useState<string | null>(null);

  // Offline Queue State
  const [offlineQueue, setOfflineQueue] = useState<QueuedPayload[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Initialize with dummy patient ID on mount and load initial Offline Queue
  useEffect(() => {
    setPatientId(generatePatientId());
    getQueuedPayloads().then(setOfflineQueue).catch(console.error);
  }, []);

  // Auto-scroll transmission logs
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [transmissionLogs]);

  // Background Sync Queue Polling
  useEffect(() => {
    const checkConnection = () => {
      return isOnline && (typeof navigator !== "undefined" ? navigator.onLine : true);
    };

    const cleanup = startQueueSync(
      async (packet) => {
        try {
          const response = await fetch("/api/transmit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ packet }),
          });
          const data = await response.json();
          if (response.ok) {
            return { success: true, logs: data.logs || [] };
          }
          return { success: false, logs: data.logs || [] };
        } catch (err) {
          return { success: false, logs: [(err as Error).toString()] };
        }
      },
      checkConnection,
      (updatedQueue) => {
        setOfflineQueue(updatedQueue);
      }
    );

    return cleanup;
  }, [isOnline]);

  // Hook into Web Speech API to speak clinical result
  const speakTriageResult = (result: InferenceResult) => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    // Interrupt any ongoing speech synthesis
    window.speechSynthesis.cancel();

    const textToSpeak = result.label === "Pneumonia"
      ? t("ttsAlertPneumonia")
      : t("ttsAlertNormal");

    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    const voiceLang = locale === "en" ? "en-US" : "ta-IN";
    utterance.lang = voiceLang;

    // Adjust rate slightly for medical announcements in clinical settings
    utterance.rate = 0.95;

    // Try finding specific voice matching the language code
    const voices = window.speechSynthesis.getVoices();
    const voice = voices.find(v => v.lang.startsWith(voiceLang) || v.lang.includes(voiceLang.replace("-", "_")));
    if (voice) {
      utterance.voice = voice;
    }

    window.speechSynthesis.speak(utterance);
  };

  // Handle local file uploads
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setSelectedImage(event.target.result as string);
          setInferenceResult(null);
          setInferenceTime(null);
          setCompressedPacket("");
          setTransmissionSuccess(null);
          setServerMessageId(null);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  // Drag and Drop support
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setSelectedImage(event.target.result as string);
          setInferenceResult(null);
          setInferenceTime(null);
          setCompressedPacket("");
          setTransmissionSuccess(null);
          setServerMessageId(null);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  // Load pre-generated local samples
  const loadSample = (type: "normal" | "pneumonia") => {
    const dataUrl = drawSampleXray(type);
    setSelectedImage(dataUrl);
    setInferenceResult(null);
    setInferenceTime(null);
    setCompressedPacket("");
    setTransmissionSuccess(null);
    setServerMessageId(null);
  };

  // Run local ML inference
  const triggerInference = async () => {
    if (!selectedImage) return;

    setIsInferenceRunning(true);
    const start = Date.now();

    try {
      // Determine if ONNX runtime is available in public directory
      let detectedModel: typeof modelType = "Client-Simulation";
      try {
        const res = await fetch("/model_quantized.onnx", { method: "HEAD" });
        if (res.ok) {
          detectedModel = "ONNX-WebAssembly";
        }
      } catch (err) {
        console.log("Unable to query ONNX model, running simulation fallback", err);
      }
      setModelType(detectedModel);

      // Run inference (onnx runtime WebAssembly or simulated high-fidelity pixel analyzer)
      const result = await runInference(selectedImage, patientId);
      
      const duration = Date.now() - start;
      setInferenceTime(duration);
      setInferenceResult(result);

      // Compress output to secure hashed SMS packet
      const packet = await compressInference(result);
      setCompressedPacket(packet);

      // Speak diagnostic result (accessibility TTS feedback)
      speakTriageResult(result);
    } catch (err) {
      console.error("Inference execution failed:", err);
    } finally {
      setIsInferenceRunning(false);
    }
  };

  // Transmit data to simulated SMS Gateway
  const handleTransmit = async () => {
    if (!compressedPacket) return;

    setIsTransmitting(true);
    setTransmissionSuccess(null);
    setServerMessageId(null);
    setTransmissionLogs([
      `Initializing SMS Transmitter...`,
      `Secure SHA-256 Hashed Payload: ${compressedPacket}`,
      `Payload size: ${compressedPacket.length} bytes.`
    ]);

    // Check online status (both simulation and browser offline status)
    const isCurrentlyOnline = isOnline && (typeof navigator !== "undefined" ? navigator.onLine : true);

    if (!isCurrentlyOnline) {
      // Queue offline behavior in IndexedDB
      await new Promise((resolve) => setTimeout(resolve, 800));
      
      setTransmissionLogs((prev) => [
        ...prev,
        "Carrier Connection Status: NO SIGNAL.",
        "Error: Unable to connect to GSM cell tower.",
        "System: Intercepted packet and queued locally in IndexedDB for store-and-forward transmission."
      ]);

      try {
        await queuePayload(compressedPacket, patientId);
        const updated = await getQueuedPayloads();
        setOfflineQueue(updated);
      } catch (err) {
        console.error("Failed to write to IndexedDB Offline Queue:", err);
      }

      setTransmissionSuccess(false);
      setIsTransmitting(false);
      return;
    }

    setTransmissionLogs((prev) => [
      ...prev,
      "Carrier Status: Connected (2G EDGE).",
      "Dialing local uplink station...",
      "Sending packet via HTTP SMS Gateway route `/api/transmit`..."
    ]);

    try {
      const response = await fetch("/api/transmit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ packet: compressedPacket }),
      });

      const data = await response.json();

      if (response.ok) {
        setTransmissionLogs((prev) => [
          ...prev,
          ...data.logs,
          `Gateway Status: DELIVERED. Message ID: ${data.gatewayMessageId}`,
          `Total Transmission Latency: ${data.totalDurationMs}ms (including retries)`
        ]);
        setTransmissionSuccess(true);
        setServerMessageId(data.gatewayMessageId);
      } else {
        setTransmissionLogs((prev) => [
          ...prev,
          ...(data.logs || []),
          `Carrier Error: ${data.error || "Uplink handshake failed."}`,
          `Link Status: Transmission failed after ${data.networkAttempts || 0} attempts.`,
          "System: Packet queued locally in IndexedDB for store-and-forward transmission."
        ]);

        // Queue on gateway failure
        try {
          await queuePayload(compressedPacket, patientId);
          const updated = await getQueuedPayloads();
          setOfflineQueue(updated);
        } catch (err) {
          console.error("Failed to queue payload:", err);
        }

        setTransmissionSuccess(false);
      }
    } catch (err) {
      setTransmissionLogs((prev) => [
        ...prev,
        "Network Link Error: Failed to fetch API route. Check server connectivity.",
        (err as Error).toString(),
        "System: Packet queued locally in IndexedDB for store-and-forward transmission."
      ]);

      // Queue on network failure
      try {
        await queuePayload(compressedPacket, patientId);
        const updated = await getQueuedPayloads();
        setOfflineQueue(updated);
      } catch (err) {
        console.error("Failed to queue payload:", err);
      }

      setTransmissionSuccess(false);
    } finally {
      setIsTransmitting(false);
    }
  };

  // Copy SMS packet to clipboard
  const copyToClipboard = () => {
    if (compressedPacket) {
      navigator.clipboard.writeText(compressedPacket);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    }
  };

  // Calculate sizes for comparison
  const originalJsonSize = inferenceResult
    ? JSON.stringify({
        patientId: inferenceResult.patientId,
        label: inferenceResult.label,
        confidence: inferenceResult.confidence,
        boundingBox: inferenceResult.boundingBox,
      }).length
    : 0;

  const compressionRatio = originalJsonSize > 0 
    ? Math.round(((originalJsonSize - compressedPacket.length) / originalJsonSize) * 100)
    : 0;

  return (
    <main className="flex-1 bg-slate-950 text-slate-100 font-sans p-6 md:p-12 flex flex-col gap-8">
      {/* Title Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-900 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs px-2.5 py-0.5 rounded-full font-mono flex items-center gap-1.5 shadow-[0_0_15px_rgba(16,185,129,0.05)]">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              {t("stableVersion")}
            </span>
            <span className="text-slate-600 text-xs">|</span>
            <span className="text-slate-500 text-xs font-mono">{t("edgeAiArchitecture")}</span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white md:text-4xl bg-clip-text bg-gradient-to-r from-white via-slate-200 to-slate-400">
            {t("title")}
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            {t("subtitle")}
          </p>
        </div>

        {/* Global Network/Model Status Indicators & Language Toggle */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Client-Side Multilingual Toggle Switch */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-0.5 flex items-center shadow-md">
            <button
              onClick={() => setLocale("en")}
              className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all duration-200 flex items-center gap-1 ${
                locale === "en"
                  ? "bg-emerald-500 text-slate-950 shadow-[0_2px_8px_rgba(16,185,129,0.3)]"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Globe className="w-3 h-3" />
              EN
            </button>
            <button
              onClick={() => setLocale("ta")}
              className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all duration-200 flex items-center gap-1 ${
                locale === "ta"
                  ? "bg-emerald-500 text-slate-950 shadow-[0_2px_8px_rgba(16,185,129,0.3)]"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Globe className="w-3 h-3" />
              தமிழ்
            </button>
          </div>

          {/* Signal Indicator Toggle */}
          <button
            onClick={() => setIsOnline(!isOnline)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all duration-200 ${
              isOnline
                ? "bg-slate-900/60 border-slate-800 text-slate-300 hover:bg-slate-800"
                : "bg-red-500/10 border-red-500/20 text-red-400 hover:bg-red-500/20"
            }`}
            title="Click to toggle GSM cellular tower connection"
          >
            {isOnline ? (
              <>
                <Wifi className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400 font-semibold">{t("gsmLinkOnline")}</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-red-400" />
                <span className="text-red-400 font-semibold">{t("gsmLinkOffline")}</span>
              </>
            )}
          </button>

          {/* Model Status Badge */}
          <div className="bg-slate-900/60 border border-slate-800 px-3 py-1.5 rounded-lg flex items-center gap-2 text-xs text-slate-300 font-medium">
            <Activity className="w-3.5 h-3.5 text-indigo-400" />
            <span>{t("engine")} </span>
            <span className="text-indigo-400 font-semibold">{modelType}</span>
          </div>
        </div>
      </div>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8 items-start">
        
        {/* COLUMN 1: Upload & Patient Config (4 cols) */}
        <div className="xl:col-span-4 flex flex-col gap-6">
          <div className="bg-slate-900/40 backdrop-blur-md border border-slate-900 rounded-xl p-5 shadow-xl flex flex-col gap-4">
            <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <Layers className="w-4 h-4 text-emerald-400" />
              <h2 className="font-bold text-slate-200 text-sm tracking-wide uppercase">
                {t("clinicSetup")}
              </h2>
            </div>

            {/* Patient ID Input */}
            <div className="flex flex-col gap-1.5">
              <label htmlFor="patient-id" className="text-xs font-semibold text-slate-400">
                {t("patientIdLabel")}
              </label>
              <div className="flex gap-2">
                <input
                  id="patient-id"
                  type="text"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value.toUpperCase())}
                  placeholder={t("patientIdPlaceholder")}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 w-full focus:outline-none focus:border-emerald-500 font-mono tracking-wide"
                />
                <button
                  onClick={() => setPatientId(generatePatientId())}
                  className="bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 px-3 py-2 rounded-lg text-xs flex items-center justify-center transition"
                  title={t("generateNewId")}
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Drag & Drop File Upload */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-400">
                {t("xrayUploadLabel")}
              </label>
              <div
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-8 text-center flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${
                  selectedImage
                    ? "border-emerald-500/50 bg-emerald-500/[0.02]"
                    : "border-slate-800 hover:border-slate-700 bg-slate-950 hover:bg-slate-900/30"
                }`}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept="image/*"
                  className="hidden"
                />
                <Upload className={`w-8 h-8 mb-3 transition-colors ${selectedImage ? "text-emerald-400" : "text-slate-500"}`} />
                <p className="text-xs text-slate-300 font-medium mb-1">
                  {t("dragDropText")}
                </p>
                <p className="text-[10px] text-slate-500">
                  {t("fileLimitText")}
                </p>
              </div>
            </div>

            {/* Preset Samples */}
            <div className="flex flex-col gap-2 pt-2">
              <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                {t("demoXraysLabel")}
              </span>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => loadSample("normal")}
                  className="bg-slate-950 hover:bg-slate-900 border border-slate-850 hover:border-slate-700 text-slate-300 text-xs py-2 px-3 rounded-lg font-medium transition flex items-center justify-center gap-1.5"
                >
                  <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                  {t("normalChest")}
                </button>
                <button
                  onClick={() => loadSample("pneumonia")}
                  className="bg-slate-950 hover:bg-slate-900 border border-slate-850 hover:border-slate-700 text-slate-300 text-xs py-2 px-3 rounded-lg font-medium transition flex items-center justify-center gap-1.5"
                >
                  <span className="w-2 h-2 rounded-full bg-red-500"></span>
                  {t("pneumoniaChest")}
                </button>
              </div>
            </div>
          </div>

          {/* Infrastructure Notes */}
          <div className="bg-slate-900/20 border border-slate-900/60 rounded-xl p-4 text-xs text-slate-400 flex gap-3 leading-relaxed">
            <Info className="w-5 h-5 text-indigo-400 shrink-0" />
            <div className="flex flex-col gap-1">
              <span className="font-semibold text-slate-300">{t("rationaleTitle")}</span>
              <p>{t("rationaleBody")}</p>
            </div>
          </div>
        </div>

        {/* COLUMN 2: Medical Viewer & Diagnostic Output (4 cols) */}
        <div className="xl:col-span-4 flex flex-col gap-6">
          <div className="bg-slate-900/40 backdrop-blur-md border border-slate-900 rounded-xl p-5 shadow-xl flex flex-col gap-5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-emerald-400" />
                <h2 className="font-bold text-slate-200 text-sm tracking-wide uppercase">
                  {t("viewportTitle")}
                </h2>
              </div>
            </div>

            {/* Medical Viewer Area */}
            <div className="relative aspect-square w-full rounded-xl bg-black border border-slate-900 overflow-hidden flex items-center justify-center shadow-inner group">
              {selectedImage ? (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={selectedImage}
                    alt="Patient Chest X-Ray"
                    className="w-full h-full object-cover transition duration-300 group-hover:scale-[1.02]"
                  />
                  {/* Anomaly Bounding Box Overlay */}
                  {inferenceResult?.label === "Pneumonia" && inferenceResult.boundingBox && (
                    <div
                      className="absolute border-2 border-red-500 bg-red-500/20 shadow-[0_0_15px_rgba(239,68,68,0.4)] animate-pulse rounded-sm transition-all duration-300"
                      style={{
                        top: `${inferenceResult.boundingBox.ymin}%`,
                        left: `${inferenceResult.boundingBox.xmin}%`,
                        width: `${inferenceResult.boundingBox.xmax - inferenceResult.boundingBox.xmin}%`,
                        height: `${inferenceResult.boundingBox.ymax - inferenceResult.boundingBox.ymin}%`,
                      }}
                    >
                      <span className="absolute -top-6 -left-0.5 bg-red-500 text-white font-mono text-[9px] font-bold px-1.5 py-0.5 rounded-t-sm uppercase tracking-wider flex items-center gap-1 shadow-md">
                        <AlertTriangle className="w-2.5 h-2.5" />
                        {t("anomalyInfiltrate")} {Math.round(inferenceResult.confidence * 100)}%
                      </span>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-slate-600 text-xs flex flex-col items-center gap-2 p-6">
                  <Layers className="w-10 h-10 text-slate-800" />
                  <span>{t("noImageLoaded")}</span>
                </div>
              )}
            </div>

            {/* Run Button */}
            {selectedImage && !inferenceResult && (
              <button
                onClick={triggerInference}
                disabled={isInferenceRunning}
                className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800 disabled:text-indigo-300 text-white font-bold py-3 rounded-lg text-xs flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/20 transition cursor-pointer"
              >
                {isInferenceRunning ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>{t("runningTransformer")}</span>
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4" />
                    <span>{t("runClassifier")}</span>
                  </>
                )}
              </button>
            )}

            {/* Diagnosis Result Output */}
            {inferenceResult && (
              <div className="flex flex-col gap-4">
                {/* Metrics Badges */}
                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className="bg-slate-950 border border-slate-850 rounded-xl p-3 flex flex-col items-center justify-center relative">
                    <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">
                      {t("diagnosisClassification")}
                    </span>
                    <span
                      className={`text-base font-extrabold mt-1 tracking-wide uppercase ${
                        inferenceResult.label === "Pneumonia" ? "text-rose-400" : "text-emerald-400"
                      }`}
                    >
                      {inferenceResult.label === "Pneumonia" ? t("labelPneumonia") : t("labelNormal")}
                    </span>
                  </div>
                  <div className="bg-slate-950 border border-slate-850 rounded-xl p-3 flex flex-col items-center justify-center">
                    <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">
                      {t("confidenceScore")}
                    </span>
                    <span className="text-base font-extrabold mt-1 text-slate-200">
                      {Math.round(inferenceResult.confidence * 1000) / 10}%
                    </span>
                  </div>
                </div>

                <div className="bg-slate-950 border border-slate-850 rounded-xl p-3 text-xs flex justify-between items-center font-mono">
                  <div className="flex items-center gap-2">
                    <span className="text-slate-500">{t("inferenceLatency")}</span>
                    <span className="text-indigo-400 font-bold">{inferenceTime} ms</span>
                  </div>
                  {/* Manual TTS Announcement Button */}
                  <button
                    onClick={() => speakTriageResult(inferenceResult)}
                    className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:bg-slate-800 text-indigo-400 hover:text-indigo-300 transition"
                    title="Play Text-to-Speech audio announcement"
                  >
                    <Volume2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Reset Trigger */}
                <button
                  onClick={triggerInference}
                  className="w-full bg-slate-900 border border-slate-800 hover:bg-slate-850 text-slate-300 py-2.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  {t("reevaluateScan")}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* COLUMN 3: SMS Compression & GSM Routing & Offline Queue (4 cols) */}
        <div className="xl:col-span-4 flex flex-col gap-6">
          
          {/* Compression Card */}
          <div className="bg-slate-900/40 backdrop-blur-md border border-slate-900 rounded-xl p-5 shadow-xl flex flex-col gap-4">
            <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <TrendingDown className="w-4 h-4 text-emerald-400" />
              <h2 className="font-bold text-slate-200 text-sm tracking-wide uppercase">
                {t("compressionTitle")}
              </h2>
            </div>

            {inferenceResult && compressedPacket ? (
              <div className="flex flex-col gap-4">
                {/* Raw SMS Output */}
                <div className="flex flex-col gap-1.5">
                  <span className="text-xs font-semibold text-slate-400">
                    {t("compressedGsmPacket")}
                  </span>
                  <div className="relative">
                    <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 pr-12 text-xs font-mono text-emerald-400 break-all select-all leading-normal tracking-wide">
                      {compressedPacket}
                    </div>
                    <button
                      onClick={copyToClipboard}
                      className="absolute right-2 top-2 p-1.5 rounded-md bg-slate-900 border border-slate-850 hover:bg-slate-800 hover:text-white transition text-slate-400"
                      title={t("copyPacket")}
                    >
                      {isCopied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                  </div>
                </div>

                {/* Compression Metrics */}
                <div className="grid grid-cols-3 gap-2 bg-slate-950/70 border border-slate-850 rounded-xl p-3.5 text-center">
                  <div className="flex flex-col">
                    <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">{t("jsonSize")}</span>
                    <span className="text-sm font-bold text-slate-400 font-mono mt-0.5">{originalJsonSize} B</span>
                  </div>
                  <div className="flex flex-col border-x border-slate-900">
                    <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">{t("smsSize")}</span>
                    <span className="text-sm font-bold text-emerald-400 font-mono mt-0.5">{compressedPacket.length} B</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">{t("reduction")}</span>
                    <span className="text-sm font-bold text-indigo-400 font-mono mt-0.5">-{compressionRatio}%</span>
                  </div>
                </div>

                {/* Packet Byte Bar */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>{t("smsLimitProgress")}</span>
                    <span className={compressedPacket.length > 140 ? "text-red-400 font-bold" : "text-emerald-400 font-bold"}>
                      {compressedPacket.length} / 140 Bytes
                    </span>
                  </div>
                  <div className="w-full bg-slate-950 border border-slate-900 h-2.5 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 rounded-full ${
                        compressedPacket.length > 140 ? "bg-red-500" : "bg-gradient-to-r from-emerald-500 to-teal-400"
                      }`}
                      style={{ width: `${Math.min(100, (compressedPacket.length / 140) * 100)}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-slate-600 text-xs py-4 text-center">
                {t("compressionAwaiting")}
              </div>
            )}
          </div>

          {/* Network Transmission Card */}
          <div className="bg-slate-900/40 backdrop-blur-md border border-slate-900 rounded-xl p-5 shadow-xl flex flex-col gap-4">
            <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <Signal className="w-4 h-4 text-emerald-400" />
              <h2 className="font-bold text-slate-200 text-sm tracking-wide uppercase">
                {t("cellularTransmissionTitle")}
              </h2>
            </div>

            {compressedPacket ? (
              <div className="flex flex-col gap-4">
                <button
                  onClick={handleTransmit}
                  disabled={isTransmitting}
                  className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:bg-emerald-800 disabled:text-emerald-300 text-white font-bold py-3 rounded-lg text-xs flex items-center justify-center gap-2 shadow-lg shadow-emerald-600/20 transition cursor-pointer"
                >
                  {isTransmitting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>{t("transmittingButton")}</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      <span>{t("transmitButton")}</span>
                    </>
                  )}
                </button>

                {/* Transmission Status Badge */}
                {transmissionSuccess !== null && (
                  <div
                    className={`flex items-center gap-2.5 p-3 rounded-xl border text-xs leading-relaxed ${
                      transmissionSuccess
                        ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                        : "bg-red-500/10 border-red-500/20 text-red-400"
                    }`}
                  >
                    {transmissionSuccess ? (
                      <>
                        <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
                        <div>
                          <p className="font-bold">{t("transmissionConfirmed")}</p>
                          <p className="text-[10px] text-emerald-500 font-mono mt-0.5">Uplink verified. ID: {serverMessageId}</p>
                        </div>
                      </>
                    ) : (
                      <>
                        <AlertTriangle className="w-4 h-4 shrink-0 text-red-400" />
                        <div>
                          <p className="font-bold">{t("transmissionFailed")}</p>
                          <p className="text-[10px] text-red-500 mt-0.5">
                            {!(isOnline && (typeof navigator !== "undefined" ? navigator.onLine : true))
                              ? t("transmissionQueued")
                              : t("transmissionTimeout")}
                          </p>
                        </div>
                      </>
                    )}
                  </div>
                )}

                {/* Simulated GSM Console logs */}
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                    {t("terminalLogs")}
                  </span>
                  <div
                    ref={logContainerRef}
                    className="bg-slate-950 border border-slate-900 rounded-lg p-3 h-32 overflow-y-auto text-[10px] font-mono text-slate-400 flex flex-col gap-1 select-text scrollbar-thin"
                  >
                    {transmissionLogs.length > 0 ? (
                      transmissionLogs.map((log, index) => (
                        <div key={index} className="leading-normal flex gap-1.5">
                          <span className="text-slate-600 shrink-0">&gt;</span>
                          <span
                            className={
                              log.includes("succeeded") || log.includes("DELIVERED") || log.includes("succeed")
                                ? "text-emerald-400"
                                : log.includes("failed") || log.includes("Error:")
                                ? "text-red-400"
                                : log.includes("Waiting") || log.includes("Attempt") || log.includes("Queued")
                                ? "text-indigo-400"
                                : "text-slate-400"
                            }
                          >
                            {log}
                          </span>
                        </div>
                      ))
                    ) : (
                      <span className="text-slate-700 italic">{t("terminalAwaiting")}</span>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-slate-600 text-xs py-4 text-center">
                {t("transmissionAwaiting")}
              </div>
            )}
          </div>

          {/* Store & Forward IndexedDB Offline Queue Panel */}
          <div className="bg-slate-900/40 backdrop-blur-md border border-slate-900 rounded-xl p-5 shadow-xl flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-emerald-400" />
                <h2 className="font-bold text-slate-200 text-sm tracking-wide uppercase">
                  {t("offlineQueueTitle")}
                </h2>
              </div>
              {offlineQueue.length > 0 && (
                <button
                  onClick={async () => {
                    await clearQueue();
                    setOfflineQueue([]);
                  }}
                  className="text-slate-500 hover:text-rose-400 transition p-1 rounded hover:bg-slate-950/50"
                  title="Clear Queue"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {offlineQueue.length > 0 ? (
              <div className="flex flex-col gap-3">
                <div className="flex justify-between items-center text-xs text-slate-400">
                  <span>{t("pendingItems")}:</span>
                  <span className="bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded-full font-mono font-bold text-[10px]">
                    {offlineQueue.length}
                  </span>
                </div>
                <div className="flex flex-col gap-2 max-h-48 overflow-y-auto scrollbar-thin">
                  {offlineQueue.map((item) => (
                    <div
                      key={item.id}
                      className="bg-slate-950/85 border border-slate-900 rounded-lg p-2.5 flex items-center justify-between text-xs transition duration-200 hover:border-slate-800"
                    >
                      <div className="flex flex-col gap-0.5">
                        <span className="font-mono text-slate-200 font-semibold">{item.patientId}</span>
                        <span className="text-[9px] text-slate-500 font-mono">
                          {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        {item.status === "transmitting" ? (
                          <span className="text-[9px] text-indigo-400 font-medium animate-pulse flex items-center gap-1 bg-indigo-500/10 px-2 py-0.5 rounded">
                            <RefreshCw className="w-2.5 h-2.5 animate-spin" />
                            {t("sending")}
                          </span>
                        ) : item.status === "failed" ? (
                          <span className="text-[9px] text-rose-400 font-medium flex items-center gap-1 bg-rose-500/10 px-2 py-0.5 rounded" title={`Failed: ${item.attempts} attempts`}>
                            <AlertTriangle className="w-2.5 h-2.5 text-rose-400" />
                            {item.attempts} retries
                          </span>
                        ) : (
                          <span className="text-[9px] text-slate-400 font-medium bg-slate-800 px-2 py-0.5 rounded">
                            {t("queued")}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-slate-600 text-xs py-4 text-center">
                {t("emptyQueue")}
              </div>
            )}
          </div>

        </div>

      </div>
    </main>
  );
}
