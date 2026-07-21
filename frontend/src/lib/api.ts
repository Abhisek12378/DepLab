import type { Conversation, CreateConversationInput, Exchange } from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const REQUEST_TIMEOUT_MS = 60_000;

interface ApiErrorPayload {
  error?: {
    code?: string;
    message?: string;
    request_id?: string;
  };
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly code = "request_failed",
    public readonly requestId?: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
    if (!response.ok) throw await responseError(response);
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("DepLab took too long to respond. Please try again.", "timeout");
    }
    throw new ApiError("Cannot reach the DepLab service. Check that the API is running.", "network_error");
  } finally {
    window.clearTimeout(timeout);
  }
}

async function responseError(response: Response): Promise<ApiError> {
  let payload: ApiErrorPayload = {};
  try {
    payload = (await response.json()) as ApiErrorPayload;
  } catch {
    // A safe generic message is better than exposing an unexpected server body.
  }
  return new ApiError(
    payload.error?.message ?? "DepLab could not complete the request.",
    payload.error?.code,
    payload.error?.request_id,
    response.status,
  );
}

export const api = {
  createConversation(input: CreateConversationInput): Promise<Conversation> {
    return apiRequest("/api/v1/conversations", {
      method: "POST",
      body: JSON.stringify(input),
    });
  },
  getConversation(id: string): Promise<Conversation> {
    return apiRequest(`/api/v1/conversations/${encodeURIComponent(id)}`);
  },
  sendMessage(id: string, content: string): Promise<Exchange> {
    return apiRequest(`/api/v1/conversations/${encodeURIComponent(id)}/messages`, {
      method: "POST",
      body: JSON.stringify({ content, client_message_id: crypto.randomUUID() }),
    });
  },
  deleteConversation(id: string): Promise<void> {
    return apiRequest(`/api/v1/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
  },
};
