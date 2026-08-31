const THREAD_ID_STORAGE_KEY = "youth-allowance.thread-id.v1";

type SessionStorageLike = Pick<Storage, "getItem" | "setItem">;

export function getOrCreateThreadId(
  storage: SessionStorageLike = window.sessionStorage,
  createId: () => string = createThreadId,
): string {
  const existing = storage.getItem(THREAD_ID_STORAGE_KEY);
  if (existing) return existing;

  const threadId = createId();
  storage.setItem(THREAD_ID_STORAGE_KEY, threadId);
  return threadId;
}

function createThreadId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }

  return `tab-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
