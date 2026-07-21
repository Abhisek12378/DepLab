import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import type { Conversation, CreateConversationInput } from "../types";

const SESSION_KEY = "deplab.conversation_id";

type Phase = "restoring" | "setup" | "ready" | "sending";

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.requestId ? `${error.message} (Request ${error.requestId})` : error.message;
  }
  return "Something unexpected happened. Please try again.";
}

export function useConversation() {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [phase, setPhase] = useState<Phase>("restoring");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const id = sessionStorage.getItem(SESSION_KEY);
    if (!id) {
      setPhase("setup");
      return;
    }
    api.getConversation(id)
      .then((value) => {
        setConversation(value);
        setPhase("ready");
      })
      .catch((cause) => {
        if (cause instanceof ApiError && cause.status === 404) {
          sessionStorage.removeItem(SESSION_KEY);
        } else {
          setError(errorMessage(cause));
        }
        setPhase("setup");
      });
  }, []);

  const start = useCallback(async (input: CreateConversationInput) => {
    setError(null);
    setPhase("sending");
    try {
      const value = await api.createConversation(input);
      sessionStorage.setItem(SESSION_KEY, value.id);
      setConversation(value);
      setPhase("ready");
    } catch (cause) {
      setError(errorMessage(cause));
      setPhase("setup");
    }
  }, []);

  const send = useCallback(async (content: string) => {
    if (!conversation || phase === "sending") return false;
    setError(null);
    setPhase("sending");
    try {
      const exchange = await api.sendMessage(conversation.id, content);
      setConversation(exchange.conversation);
      setPhase("ready");
      return true;
    } catch (cause) {
      setError(errorMessage(cause));
      setPhase("ready");
      return false;
    }
  }, [conversation, phase]);

  const reset = useCallback(async () => {
    const id = conversation?.id;
    sessionStorage.removeItem(SESSION_KEY);
    setConversation(null);
    setError(null);
    setPhase("setup");
    if (id) await api.deleteConversation(id).catch(() => undefined);
  }, [conversation]);

  return {
    conversation,
    phase,
    error,
    clearError: () => setError(null),
    start,
    send,
    reset,
  };
}
