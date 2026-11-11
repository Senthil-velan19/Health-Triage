/**
 * Semantic Data Compression for EdgeTriage
 * ----------------------------------------
 * This module implements a highly optimized compression algorithm to pack
 * machine learning inference results into a single SMS packet (< 140 bytes).
 * It includes an FNV-1a 16-bit checksum to verify integrity at the receiving gateway.
 *
 * SMS Format:
 * TRG:<Patient ID>:<Diagnosis>:<Confidence>:<Bounding Box>:<Checksum>
 *
 * Example:
 * TRG:PID123:PNEU:95:12-45-40-85:8F3A
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
 * Computes a 16-bit FNV-1a hash of a given string.
 * This is a highly efficient, non-cryptographic hashing algorithm
 * used here as a checksum to verify transmission integrity.
 */
export function computeFNV1a16(data: string): string {
  let hash = 0x811c9dc5; // 32-bit FNV offset basis
  for (let i = 0; i < data.length; i++) {
    hash ^= data.charCodeAt(i);
    // Multiply by FNV prime 16777619
    hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
  }
  // Fold 32-bit hash to 16-bit
  const folded = ((hash >> 16) ^ (hash & 0xffff)) & 0xffff;
  return folded.toString(16).toUpperCase().padStart(4, "0");
}

/**
 * Compresses an InferenceResult object into an SMS-compatible string.
 * Ensures the output is strictly under 140 bytes.
 */
export function compressInference(result: InferenceResult): string {
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

  // 6. Compute checksum
  const checksum = computeFNV1a16(basePayload);

  // 7. Assemble final packet
  const packet = `${basePayload}:${checksum}`;

  if (packet.length > 140) {
    throw new Error(`Compressed packet size (${packet.length} bytes) exceeds the 140-byte SMS limit.`);
  }

  return packet;
}

/**
 * Verifies the checksum of a compressed packet.
 */
export function verifyChecksum(packet: string): boolean {
  const parts = packet.split(":");
  if (parts.length !== 6 || parts[0] !== "TRG") {
    return false;
  }
  const basePayload = parts.slice(0, 5).join(":");
  const providedChecksum = parts[5];
  const calculatedChecksum = computeFNV1a16(basePayload);
  return calculatedChecksum === providedChecksum;
}

/**
 * Decompresses an SMS packet back into an InferenceResult object.
 * Returns null if the packet is invalid or has a failing checksum.
 */
export function decompressInference(packet: string): InferenceResult | null {
  if (!verifyChecksum(packet)) {
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
