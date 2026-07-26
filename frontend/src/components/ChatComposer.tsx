import { KeyboardEvent, useState } from "react";
import { ArrowUp, CornerDownLeft } from "lucide-react";

interface ChatComposerProps {
  busy: boolean;
  showExamples: boolean;
  onSend: (content: string) => Promise<boolean>;
}

const EXAMPLE_QUESTIONS = [
  "Can I upgrade numpy to 2.0.2?",
  "What breaks if I move to Python 3.12?",
  "Suggest the safest full upgrade",
];

export function ChatComposer({ busy, showExamples, onSend }: ChatComposerProps) {
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
      {showExamples && (
        <div className="composer-examples" aria-label="Example questions">
          {EXAMPLE_QUESTIONS.map((question) => (
            <button type="button" key={question} onClick={() => setContent(question)} disabled={busy}>{question}</button>
          ))}
        </div>
      )}
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
      <div className="composer-help"><span><CornerDownLeft size={12} /> Enter to send</span><span>uv resolution + post-install prediction · no runtime install</span></div>
    </div>
  );
}
