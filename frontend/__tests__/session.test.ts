import { expect, test, vi } from "vitest";

import { getOrCreateThreadId } from "../lib/session";

class MemorySessionStorage {
  private readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

test("keeps the same thread id in the same tab session", () => {
  const storage = new MemorySessionStorage();
  const createId = vi.fn().mockReturnValueOnce("tab-a").mockReturnValueOnce("unused");

  expect(getOrCreateThreadId(storage, createId)).toBe("tab-a");
  expect(getOrCreateThreadId(storage, createId)).toBe("tab-a");
  expect(createId).toHaveBeenCalledTimes(1);
});

test("creates a different thread id for a separate tab session", () => {
  const firstTab = new MemorySessionStorage();
  const secondTab = new MemorySessionStorage();

  expect(getOrCreateThreadId(firstTab, () => "tab-a")).toBe("tab-a");
  expect(getOrCreateThreadId(secondTab, () => "tab-b")).toBe("tab-b");
});
