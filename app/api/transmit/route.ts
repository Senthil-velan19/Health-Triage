import { NextResponse } from "next/server";
import { verifyChecksum } from "../../../lib/securePayload";

// Configuration for simulation
const MAX_RETRIES = 3;
const BASE_BACKOFF_MS = 300;
const SIMULATED_PACKET_DROP_RATE = 0.35; // 35% chance to fail/timeout on any single link attempt (simulating 2G/GSM instability)

/**
 * Simulates a single HTTP POST request to an external SMS Gateway.
 * Simulates latency, timeouts, and connection losses common in rural GSM networks.
 */
async function postToUpstreamGateway(packet: string, attempt: number): Promise<{ success: boolean; latency: number }> {
  const start = Date.now();
  
  // Simulate latency of a 2G network link: 800ms - 2500ms
  const networkDelay = Math.floor(Math.random() * 1700) + 800;
  await new Promise((resolve) => setTimeout(resolve, networkDelay));
  
  const latency = Date.now() - start;

  // Simulate unstable network drops or timeouts
  if (Math.random() < SIMULATED_PACKET_DROP_RATE) {
    throw new Error(`[Attempt ${attempt}] Timeout: Connection reset by remote GSM cell tower.`);
  }

  return { success: true, latency };
}

/**
 * Sends the packet to the SMS Gateway, retrying on failure with Exponential Backoff.
 */
async function sendToSMSGatewayWithRetry(
  packet: string
): Promise<{ success: boolean; attempts: number; totalLatency: number; logs: string[] }> {
  const logs: string[] = [];
  let attempts = 0;
  let totalLatency = 0;

  while (attempts < MAX_RETRIES) {
    attempts++;
    logs.push(`Attempt ${attempts} starting...`);
    
    try {
      const result = await postToUpstreamGateway(packet, attempts);
      totalLatency += result.latency;
      logs.push(`Attempt ${attempts} succeeded in ${result.latency}ms.`);
      return {
        success: true,
        attempts,
        totalLatency,
        logs,
      };
    } catch (err) {
      const errorMsg = (err as Error).message;
      logs.push(`Attempt ${attempts} failed: ${errorMsg}`);
      
      if (attempts >= MAX_RETRIES) {
        break;
      }
      
      // Exponential backoff: delay double on each retry
      const backoffDelay = BASE_BACKOFF_MS * Math.pow(2, attempts);
      logs.push(`Waiting ${backoffDelay}ms before retrying (exponential backoff)...`);
      await new Promise((resolve) => setTimeout(resolve, backoffDelay));
      totalLatency += backoffDelay;
    }
  }

  return {
    success: false,
    attempts,
    totalLatency,
    logs,
  };
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { packet } = body;

    if (!packet) {
      return NextResponse.json(
        { error: "Bad Request: Missing 'packet' field in request body." },
        { status: 400 }
      );
    }

    // 1. Verify packet structure and checksum integrity
    const isValid = await verifyChecksum(packet);
    if (!isValid) {
      return NextResponse.json(
        { 
          error: "Transmission rejected: Checksum validation failed.", 
          packet,
          verified: false 
        },
        { status: 422 } // Unprocessable Entity
      );
    }

    // 2. Transmit through GSM cellular simulation
    const transmissionResult = await sendToSMSGatewayWithRetry(packet);

    if (transmissionResult.success) {
      // Return 200 with simulated transaction details
      return NextResponse.json({
        status: "TRANSMITTED",
        message: "SMS packet routed to central triage server successfully.",
        gatewayMessageId: `msg_${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
        networkAttempts: transmissionResult.attempts,
        totalDurationMs: transmissionResult.totalLatency,
        packetSize: packet.length,
        logs: transmissionResult.logs,
      });
    } else {
      // Return 504 gateway timeout if all retries failed due to unstable 2G
      return NextResponse.json(
        {
          status: "FAILED",
          error: "GSM cellular network unavailable. Upstream SMS Gateway connection timed out.",
          networkAttempts: transmissionResult.attempts,
          totalDurationMs: transmissionResult.totalLatency,
          logs: transmissionResult.logs,
        },
        { status: 504 }
      );
    }
  } catch (err) {
    return NextResponse.json(
      { error: "Internal Server Error", details: (err as Error).message },
      { status: 500 }
    );
  }
}
