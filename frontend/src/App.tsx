import { ChatWorkspace } from "./components/ChatWorkspace";
import { SetupScreen } from "./components/SetupScreen";
import { useConversation } from "./hooks/useConversation";

export default function App() {
  const state = useConversation();
  if (state.phase === "restoring") return <div className="app-loader"><div className="loader-mark">D</div><span>Restoring your workspace…</span></div>;
  if (!state.conversation) return <SetupScreen busy={state.phase === "sending"} error={state.error} onStart={state.start} />;
  return <ChatWorkspace conversation={state.conversation} busy={state.phase === "sending"} error={state.error} onSend={state.send} onReset={state.reset} onClearError={state.clearError} />;
}
