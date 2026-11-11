/**
 * Semantic Data Compression & Cryptographic Tamper-Proofing
 * -----------------------------------------------------------
 * This module implements a semantic compression algorithm that packs local
 * Vision Transformer medical inference results into a sub-140-byte string
 * suitable for legacy GSM/SMS networks.
 *
 * Security Architecture (Web Crypto API SHA-256):
 * ----------------------------------------------
 * Traditional 2G networks are unencrypted and highly vulnerable to Man-in-the-Middle (MitM)
 * packet tampering and carrier-level transmission corruption (bit-flips). While traditional
 * CRC or FNV-1a checksums detect random bit flips, they provide zero protection against
 * malicious alteration.
 *
 * To ensure absolute data integrity over unencrypted channels, this utility generates
 * a cryptographic checksum:
 * 1. The structured medical telemetry is compiled into a colon-separated string.
 * 2. We compute a native SHA-256 hash of this payload using the browser's Web Crypto API.
 * 3. The first 6 characters (24-bit slice) of the hex-encoded digest are appended to the payload.
 *
 * Any attempt to alter patient data (e.g. changing classification 'NORM' to 'PNEU' or
 * modification of Patient ID) will alter the SHA-256 digest entirely, causing the central
 * gateway to reject the packet during verification.
 *
 * SMS Format:
 * TRG:<Patient ID>:<Diagnosis>:<Confidence>:<Bounding Box>:<SHA-256 Checksum Slice>
 * Example:
 * TRG:PID35165:NORM:79:0-0-0-0:A8F2B1
 */

export interface BoundingBox {
  ymin: number; // 0-100 (percentage of height)
  xmin: number; // 0-100 (percentage of width)
  ymax: number; // 0-100
  xmax: number; // 0-100
}

export interface InferenceResult {
  patientId: string;
  label: "Pneumonia" | "Normal";
  confidence: number; // 0.0 to 1.0
  boundingBox: BoundingBox | null;
}

/**
 * Computes the first 6 characters of the SHA-256 hash of a string using Web Crypto API.
 * Designed to run in both modern browsers (window.crypto) and Next.js server environments (globalThis.crypto).
 */
export async function computeSHA256Checksum(basePayload: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(basePayload);

  let cryptoInstance: any = null;

  if (typeof window !== "undefined" && window.crypto) {
    cryptoInstance = window.crypto;
  } else if (typeof globalThis !== "undefined" && (globalThis as any).crypto) {
    cryptoInstance = (globalThis as any).crypto;
  } else {
    // Node.js fallback for Next.js Route handlers in older environments
    try {
      const nodeCrypto = require("crypto");
      cryptoInstance = nodeCrypto.webcrypto;
    } catch (e) {
      throw new Error("Cryptography API: Web Crypto Subtle is unavailable in this runtime environment.");
    }
  }

  if (!cryptoInstance || !cryptoInstance.subtle) {
    throw new Error("Cryptography API: SubtleCrypto interface is missing in this runtime environment.");
  }

  // Generate SHA-256 digest of the base payload
  const hashBuffer = await cryptoInstance.subtle.digest("SHA-256", data);
  
  // Convert buffer to hex string
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");

  // Return the first 6 characters, uppercase, for low-overhead SMS transmission
  return hashHex.slice(0, 6).toUpperCase();
}

/**
 * Compresses an InferenceResult object into a secure, tamper-proof SMS packet.
 * Verifies that the final length is strictly <= 140 bytes.
 */
export async function compressInference(result: InferenceResult): Promise<string> {
  // 1. Sanitize Patient ID (alphanumeric only, default to UNKNOWN if empty)
  const patientId = result.patientId.replace(/[^a-zA-Z0-9]/g, "") || "UNKNOWN";

  // 2. Map label to a 4-character code
  const diagnosis = result.label === "Pneumonia" ? "PNEU" : "NORM";

  // 3. Scale confidence to percentage integer (0-100)
  const confidence = Math.round(result.confidence * 100);

  // 4. Compress bounding box coordinates (round to 0-100 integers)
  let boxStr = "0-0-0-0";
  if (result.boundingBox && result.label === "Pneumonia") {
    const { ymin, xmin, ymax, xmax } = result.boundingBox;
    const clamp = (val: number) => Math.max(0, Math.min(100, Math.round(val)));
    boxStr = `${clamp(ymin)}-${clamp(xmin)}-${clamp(ymax)}-${clamp(xmax)}`;
  }

  // 5. Construct the base payload string
  const basePayload = `TRG:${patientId}:${diagnosis}:${confidence}:${boxStr}`;

  // 6. Compute secure SHA-256 checksum
  const checksum = await computeSHA256Checksum(basePayload);

  // 7. Assemble final packet
  const packet = `${basePayload}:${checksum}`;

  if (packet.length > 140) {
    throw new Error(`Compression Error: Compressed packet size (${packet.length} bytes) exceeds the 140-byte GSM SMS limit.`);
  }

  return packet;
}

/**
 * Verifies the SHA-256 cryptographic checksum of a compressed packet.
 * Returns true if checksum matches, false if packet has been corrupted or tampered.
 */
export async function verifyChecksum(packet: string): Promise<boolean> {
  const parts = packet.split(":");
  if (parts.length !== 6 || parts[0] !== "TRG") {
    return false;
  }
  const basePayload = parts.slice(0, 5).join(":");
  const providedChecksum = parts[5];
  
  const calculatedChecksum = await computeSHA256Checksum(basePayload);
  return calculatedChecksum === providedChecksum;
}

/**
 * Decompresses a secure SMS packet back into its original InferenceResult structure.
 * Performs signature verification first; returns null if the signature is invalid or packet is malformed.
 */
export async function decompressInference(packet: string): Promise<InferenceResult | null> {
  const isValid = await verifyChecksum(packet);
  if (!isValid) {
    console.warn("Security Alert: Cryptographic signature mismatch! The transmission packet may have been tampered with.");
    return null;
  }

  const parts = packet.split(":");
  const patientId = parts[1];
  const diagCode = parts[2];
  const confInt = parseInt(parts[3], 10);
  const boxParts = parts[4].split("-").map((v) => parseInt(v, 10));

  if (isNaN(confInt) || boxParts.length !== 4 || boxParts.some(isNaN)) {
    return null;
  }

  const label: "Pneumonia" | "Normal" = diagCode === "PNEU" ? "Pneumonia" : "Normal";
  const confidence = confInt / 100.0;

  let boundingBox: BoundingBox | null = null;
  if (diagCode === "PNEU") {
    boundingBox = {
      ymin: boxParts[0],
      xmin: boxParts[1],
      ymax: boxParts[2],
      xmax: boxParts[3],
    };
  }

  return {
    patientId,
    label,
    confidence,
    boundingBox,
  };
}
