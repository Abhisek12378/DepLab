import { FileCode2, History, MessageSquareText, Plus, ShieldCheck } from "lucide-react";
import { shortConversationId } from "../lib/presentation";
import type { Conversation } from "../types";
import { Brand } from "./Brand";
import { ChatComposer } from "./ChatComposer";
import { MessageList } from "./MessageList";

interface ChatWorkspaceProps {
  conversation: Conversation;
  busy: boolean;
  error: string | null;
  onSend: (content: string) => Promise<boolean>;
  onReset: () => Promise<void>;
  onClearError: () => void;
}

export function ChatWorkspace({ conversation, busy, error, onSend, onReset, onClearError }: ChatWorkspaceProps) {
  const packageCount = conversation.requirements_text.split(/\r?\n/).filter((line) => line.trim() && !line.trim().startsWith("#")).length;
  return (
    <main className="workspace">
      <aside className="workspace-sidebar">
        <Brand />
        <button className="new-analysis" type="button" onClick={() => void onReset()}><Plus size={16} /> New analysis</button>
        <nav className="sidebar-nav" aria-label="Conversation"><span className="nav-label">CURRENT</span><div className="nav-item active"><MessageSquareText size={16} /><div><strong>Dependency review</strong><small>{shortConversationId(conversation.id)}</small></div></div></nav>
        <div className="sidebar-spacer" />
        <div className="trust-card"><ShieldCheck size={17} /><div><strong>Evidence-aware</strong><p>Facts and predictions stay clearly separated.</p></div></div>
      </aside>

      <section className="chat-panel">
        <header className="chat-header"><div><span className="mobile-brand"><Brand /></span><h1>Dependency review</h1><p><span className="status-dot" /> Conversation memory active</p></div><div className="header-model"><span>MODEL</span><strong className="model-badge" tabIndex={0}>DepLab Hybrid<span className="model-tooltip" role="tooltip">Structured metadata model + failure head, trained on 4,109 real installation experiments.</span></strong></div></header>
        {error && <div className="global-error" role="alert"><span>{error}</span><button onClick={onClearError}>Dismiss</button></div>}
        <MessageList messages={conversation.messages} sending={busy} />
        <ChatComposer busy={busy} showExamples={conversation.messages.length === 0} onSend={onSend} />
      </section>

      <aside className="context-panel">
        <div className="context-heading"><History size={16} /><span>Analysis context</span></div>
        <div className="context-stat-grid"><div><span>Python</span><strong>{conversation.python_version}</strong></div><div><span>Platform</span><strong>Linux x64</strong></div></div>
        <div className="context-block"><div className="context-label"><FileCode2 size={14} /><span>requirements.txt</span><small>{packageCount} packages</small></div><pre>{conversation.requirements_text}</pre></div>
        <div className="memory-card"><div className="memory-pulse" /><div><strong>Follow-up memory</strong><p>DepLab uses the recent conversation to resolve references such as “that version.”</p></div></div>
        <p className="expiry-copy">This private session expires at {new Date(conversation.expires_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}.</p>
      </aside>
    </main>
  );
}
