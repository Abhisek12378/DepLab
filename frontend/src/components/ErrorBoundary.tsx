import { Component, ErrorInfo, ReactNode } from "react";

export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("DepLab interface error", error.name, info.componentStack);
  }

  render() {
    if (this.state.failed) {
      return <main className="fatal-error"><div className="brand-mark">D</div><h1>The interface needs a fresh start.</h1><p>Your server-side conversation is safe. Reload the page to restore it.</p><button onClick={() => window.location.reload()}>Reload DepLab</button></main>;
    }
    return this.props.children;
  }
}
