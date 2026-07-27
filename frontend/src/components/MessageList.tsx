import { useEffect, useRef, type RefObject } from "react";
import { Bot, UserRound } from "lucide-react";
import type { ChatMessage } from "../types";
import { ResultDetails } from "./ResultDetails";
import { StructuredAnswer } from "./StructuredAnswer";

interface MessageListProps {
  messages: ChatMessage[];
  sending: boolean;
}

export function MessageList({ messages, sending }: MessageListProps) {
  const end = useRef<HTMLDivElement>(null);
  const latestAssistant = useRef<HTMLElement>(null);
  useEffect(() => {
    const latestMessage = messages[messages.length - 1];
    if (!sending && latestMessage?.role === "assistant") {
      latestAssistant.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
      return;
    }
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, sending]);
  return (
    <div className="message-list" aria-live="polite">
      {messages.length === 0 && <EmptyChat />}
      {messages.map((message, index) => (
        <Message
          key={message.id}
          message={message}
          messageRef={
            index === messages.length - 1 && message.role === "assistant"
              ? latestAssistant
              : undefined
          }
        />
      ))}
      {sending && <Thinking />}
      <div ref={end} />
    </div>
  );
}

function Message({
  message,
  messageRef,
}: {
  message: ChatMessage;
  messageRef?: RefObject<HTMLElement | null>;
}) {
  const assistant = message.role === "assistant";
  return (
    <article className={`message-row ${message.role}`} ref={messageRef}>
      <div className="message-avatar">{assistant ? <Bot size={17} /> : <UserRound size={17} />}</div>
      <div className="message-content">
        <div className="message-meta"><strong>{assistant ? "DepLab" : "You"}</strong><time>{new Date(message.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time></div>
        <div className="message-bubble">{assistant && message.result ? <StructuredAnswer result={message.result} fallback={message.content} /> : message.content}</div>
        {assistant && message.result && <ResultDetails result={message.result} />}
      </div>
    </article>
  );
}

function EmptyChat() {
  return (
    <div className="empty-chat">
      <div className="empty-orbit"><Bot size={25} /></div>
      <span className="eyebrow">Workspace ready</span>
      <h2>What would you like to change?</h2>
      <p>Ask about an upgrade, downgrade, or exact package combination. You can ask follow-up questions naturally.</p>
    </div>
  );
}

function Thinking() {
  return (
    <div className="message-row assistant thinking-row">
      <div className="message-avatar"><Bot size={17} /></div>
      <div className="thinking"><span /><span /><span /><em>Ranking candidates and checking uv resolution…</em></div>
    </div>
  );
}
