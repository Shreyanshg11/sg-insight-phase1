import { Component } from "react";

/** Last line of defence: a render crash shows a recoverable message instead of a blank page. */
export default class ErrorBoundary extends Component {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error, info) {
    console.error("UI crashed:", error, info?.componentStack);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div role="alert" className="mx-auto mt-24 max-w-md space-y-4 p-6 text-center font-mono text-sm text-ink-100">
        <p>Something went wrong displaying this page.</p>
        <p className="text-xs text-ink-500">Your uploaded data is stored on the server and is not lost.</p>
        <button type="button" onClick={() => window.location.reload()}
          className="border border-signal-amber px-4 py-2 text-xs focus:outline focus:outline-2">Reload</button>
      </div>
    );
  }
}
