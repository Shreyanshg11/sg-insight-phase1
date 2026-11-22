import { useEffect, useState } from "react";
import { getAuthStatus, login, logout } from "../api/client";
import { AuthContext } from "./authContext";

export default function AuthGate({ children }) {
  const [state, setState] = useState({ loading: true, enabled: false, authenticated: true });
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let active = true;
    getAuthStatus().then((value) => { if (active) setState({ loading: false, ...value }); })
      .catch(() => { if (active) setState({ loading: false, enabled: true, authenticated: false, statusError: true }); });
    return () => { active = false; };
  }, []);

  const submit = async (event) => {
    event.preventDefault(); setSubmitting(true); setError("");
    try {
      await login(password);
      setPassword("");
      setState((current) => ({ ...current, authenticated: true }));
    } catch (failure) {
      setError(failure.message || "Sign in failed.");
    } finally { setSubmitting(false); }
  };

  if (state.loading) return <main className="min-h-screen bg-graphite-950 p-8 text-ink-300" aria-live="polite">Checking access…</main>;
  if (state.statusError) return <main className="min-h-screen bg-graphite-950 p-8 text-ink-100" role="alert">Unable to verify workspace access. Check the backend connection and reload this page.</main>;
  if (!state.enabled) return children;
  if (state.authenticated) {
    const signOut = async () => { await logout(); setState((current) => ({ ...current, authenticated: false })); };
    return <AuthContext.Provider value={{ enabled: true, signOut }}>{children}</AuthContext.Provider>;
  }

  return (
    <main className="min-h-screen bg-graphite-950 px-4 py-16 text-ink-100">
      <section className="mx-auto max-w-sm rounded-lg border border-graphite-border bg-graphite-900 p-6 shadow-xl">
        <p className="font-mono text-xs uppercase tracking-widest text-signal-amber">SG-Insight</p>
        <h1 className="mt-3 text-2xl font-semibold">Sign in</h1>
        <p className="mt-2 text-sm text-ink-400">Enter the operator password to open this workspace.</p>
        <form className="mt-6 space-y-4" onSubmit={submit}>
          <label className="block text-sm font-medium" htmlFor="operator-password">Password</label>
          <input id="operator-password" type="password" autoComplete="current-password" required maxLength={1024}
            value={password} onChange={(event) => setPassword(event.target.value)}
            className="w-full rounded border border-graphite-border bg-graphite-950 px-3 py-2 text-ink-100" />
          {error && <p role="alert" className="text-sm text-red-300">{error}</p>}
          <button disabled={submitting} className="w-full rounded bg-signal-amber px-4 py-2 font-semibold text-graphite-950 disabled:opacity-60">
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </section>
    </main>
  );
}
