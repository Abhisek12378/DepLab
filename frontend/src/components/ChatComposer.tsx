import { KeyboardEvent, useState } from "react";
import { ArrowUp, CornerDownLeft } from "lucide-react";

interface ChatComposerProps {
  busy: boolean;
  onSend: (content: string) => Promise<boolean>;
}

export function ChatComposer({ busy, onSend }: ChatComposerProps) {
  const [content, setContent] = useState("");

  async function submit() {
    const value = content.trim();
    if (!value || busy) return;
    if (await onSend(value)) setContent("");
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  }

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          value={content}
          onChange={(event) => setContent(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a dependency question or follow up on the answer…"
          maxLength={2000}
          rows={1}
          disabled={busy}
          aria-label="Message DepLab"
        />
        <button type="button" onClick={() => void submit()} disabled={!content.trim() || busy} aria-label="Send message"><ArrowUp size={19} /></button>
      </div>
      <div className="composer-help"><span><CornerDownLeft size={12} /> Enter to send</span><span>Prediction and published evidence · no runtime install</span></div>
    </div>
  );
}
