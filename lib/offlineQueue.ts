/**
 * IndexedDB Offline Queue (Store & Forward)
 * -----------------------------------------
 * This module implements an offline-first storage queue using browser-native IndexedDB.
 * It is designed to handle extreme packet loss and long-duration cellular network
 * blackouts in remote clinics.
 *
 * Operational Flow:
 * ----------------
 * 1. When an edge-inference report is ready to be transmitted, the system checks if the
 *    GSM cellular link is active (checking both physical browser network state `navigator.onLine`
 *    and simulated environment overrides).
 * 2. If the link is offline or transmission fails, the cryptographically signed payload
 *    is intercepted and stored locally in IndexedDB as a queued packet.
 * 3. A background polling mechanism monitors network connectivity. When connection is restored:
 *    - The background worker opens the IndexedDB queue.
 *    - It flushes the packets sequentially, ensuring FIFO (First-In, First-Out) ordering.
 *    - Successfully transmitted packets are deleted from IndexedDB.
 *    - Failed attempts update the item's metadata and pause queue flushing to avoid packet flooding.
 *
 * This architecture guarantees zero medical data loss, enabling clinicians to continue scanning
 * patients offline and trusting the system to synchronize data once connectivity is available.
 */

export interface QueuedPayload {
  id?: number;              // Auto-incremented primary key in IndexedDB
  packet: string;           // The compressed, SHA-256 secure payload string
  patientId: string;        // Sanity ID for UI display and tracking
  timestamp: number;        // Epoch timestamp when the packet was queued
  attempts: number;         // Total failed transmission retries
  status: "queued" | "failed" | "transmitting";
}

const DB_NAME = "EdgeTriageOfflineDB";
const STORE_NAME = "offlineQueue";
const DB_VERSION = 1;

/**
 * Initializes and opens a connection to the IndexedDB database.
 * Creates the object store if it does not already exist.
 */
export function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof window === "undefined") {
      reject(new Error("IndexedDB is only available in the browser client environment."));
      return;
    }

    const request = window.indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => {
      reject(new Error(`IndexedDB Error: Failed to open database ${DB_NAME}.`));
    };

    request.onsuccess = () => {
      resolve(request.result);
    };

    request.onupgradeneeded = (event) => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        // Auto-incrementing IDs for packet sorting
        db.createObjectStore(STORE_NAME, { keyPath: "id", autoIncrement: true });
      }
    };
  });
}

/**
 * Adds a payload to the IndexedDB store-and-forward queue.
 */
export async function queuePayload(packet: string, patientId: string): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    const store = transaction.objectStore(STORE_NAME);

    const newItem: Omit<QueuedPayload, "id"> = {
      packet,
      patientId,
      timestamp: Date.now(),
      attempts: 0,
      status: "queued",
    };

    const request = store.add(newItem);

    request.onsuccess = () => {
      resolve(request.result as number);
    };

    request.onerror = () => {
      reject(new Error("IndexedDB Error: Failed to write payload to offline queue."));
    };
  });
}

/**
 * Fetches all queued payloads from IndexedDB, sorted by insertion order (chronological).
 */
export async function getQueuedPayloads(): Promise<QueuedPayload[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readonly");
    const store = transaction.objectStore(STORE_NAME);
    const request = store.getAll();

    request.onsuccess = () => {
      // Sort items chronologically by timestamp
      const sorted = (request.result as QueuedPayload[]).sort((a, b) => a.timestamp - b.timestamp);
      resolve(sorted);
    };

    request.onerror = () => {
      reject(new Error("IndexedDB Error: Failed to retrieve offline queue items."));
    };
  });
}

/**
 * Deletes a payload from IndexedDB by its unique ID.
 * Typically called after successful GSM server transmission.
 */
export async function deletePayload(id: number): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    const store = transaction.objectStore(STORE_NAME);
    const request = store.delete(id);

    request.onsuccess = () => {
      resolve();
    };

    request.onerror = () => {
      reject(new Error(`IndexedDB Error: Failed to delete payload with ID ${id} from queue.`));
    };
  });
}

/**
 * Updates a payload item in IndexedDB (e.g. updating attempts count or status).
 */
export async function updatePayload(payload: QueuedPayload): Promise<void> {
  if (payload.id === undefined) return;
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    const store = transaction.objectStore(STORE_NAME);
    const request = store.put(payload);

    request.onsuccess = () => {
      resolve();
    };

    request.onerror = () => {
      reject(new Error(`IndexedDB Error: Failed to update payload with ID ${payload.id} in queue.`));
    };
  });
}

/**
 * Clears the entire offline queue. Useful for debugging and resetting states.
 */
export async function clearQueue(): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    const store = transaction.objectStore(STORE_NAME);
    const request = store.clear();

    request.onsuccess = () => {
      resolve();
    };

    request.onerror = () => {
      reject(new Error("IndexedDB Error: Failed to clear offline queue."));
    };
  });
}

// Global flag to prevent concurrent overlapping queue sync runs
let isSyncing = false;

/**
 * Iterates through all queued items in IndexedDB and attempts sequentially to transmit them.
 * Halts processing immediately on the first failed packet to preserve chronological delivery order (FIFO).
 */
export async function syncQueue(
  transmitFn: (packet: string) => Promise<{ success: boolean; logs?: string[] }>,
  isOnlineFn: () => boolean,
  onQueueChanged?: (queue: QueuedPayload[]) => void
): Promise<void> {
  if (isSyncing) return;
  if (!isOnlineFn()) return;

  const queue = await getQueuedPayloads();
  if (queue.length === 0) return;

  isSyncing = true;

  try {
    for (const item of queue) {
      // Re-verify network status before sending next packet
      if (!isOnlineFn()) break;
      if (item.id === undefined) continue;

      // Update status to transmitting
      item.status = "transmitting";
      await updatePayload(item);
      if (onQueueChanged) {
        onQueueChanged(await getQueuedPayloads());
      }

      try {
        const result = await transmitFn(item.packet);
        if (result.success) {
          // Successfully transmitted, remove from queue
          await deletePayload(item.id);
        } else {
          // Fail. Mark status and increment attempt count
          item.status = "failed";
          item.attempts += 1;
          await updatePayload(item);
          // Halt execution of remaining queue items to avoid out-of-order delivery
          break;
        }
      } catch (err) {
        item.status = "failed";
        item.attempts += 1;
        await updatePayload(item);
        break;
      } finally {
        if (onQueueChanged) {
          onQueueChanged(await getQueuedPayloads());
        }
      }
    }
  } finally {
    isSyncing = false;
  }
}

/**
 * Starts a background timer to automatically synchronise queued IndexedDB packets.
 * Listens to native browser 'online' events to instantly sync, and falls back to polling.
 * Returns a cleanup unsubscribe callback function.
 */
export function startQueueSync(
  transmitFn: (packet: string) => Promise<{ success: boolean; logs?: string[] }>,
  isOnlineFn: () => boolean,
  onQueueChanged: (queue: QueuedPayload[]) => void,
  pollIntervalMs = 5000
): () => void {
  // Initial sync attempt
  syncQueue(transmitFn, isOnlineFn, onQueueChanged);

  // Setup periodic polling
  const intervalId = setInterval(() => {
    syncQueue(transmitFn, isOnlineFn, onQueueChanged);
  }, pollIntervalMs);

  // Trigger sync instantly when physical connection status changes to online
  const handleOnline = () => {
    syncQueue(transmitFn, isOnlineFn, onQueueChanged);
  };

  if (typeof window !== "undefined") {
    window.addEventListener("online", handleOnline);
  }

  // Return destructor
  return () => {
    if (intervalId) {
      clearInterval(intervalId);
    }
    if (typeof window !== "undefined") {
      window.removeEventListener("online", handleOnline);
    }
  };
}
