import { useState, useRef, useEffect } from "react";
import { sendCopilotMessage, confirmCopilotAction, dismissCopilotAction } from "../../api/client";
import ConfidenceMeter from "../ConfidenceMeter";
import LoadingState from "../ui/LoadingState";
import ErrorState from "../ui/ErrorState";

const STEP_LABELS = {
  duplicate_manager: "Duplicate Manager",
  missing_value_center: "Missing Values",
  outlier_center: "Outliers",
  category_merge_center: "Category Merge",
  text_cleaning: "Text Cleaning",
  ratio: "Ratio Feature",
  date_features: "Date Features",
  bucketing: "Bucketing",
  scaling_normalization: "Scaling",
  encoding: "Encoding",
  log_transform: "Log Transform",

};

const CLEANING_ACTIONS = {
  duplicate_manager: ["drop_duplicates"],
  missing_value_center: ["ignore", "impute_mean", "impute_median", "impute_mode", "impute_constant", "drop_column", "drop_rows"],
  // Review-only in Copilot: outlier advice (flag_for_review) and fuzzy_merge suggestions are
  // never executable here, so they never get a Confirm button. The server enforces the same rule.
  outlier_center: [],
  category_merge_center: ["normalize_case_whitespace", "rename_category"],
  text_cleaning: ["trim", "lowercase", "uppercase", "remove_special_chars"],
};

const FEATURE_ACTIONS = {
  date_features: ["extract_date_parts"],
  ratio: ["create_ratio"],
  bucketing: ["quartile_bucket"],
  scaling_normalization: ["min_max_scale", "z_score_standardize", "robust_scale"],
  encoding: ["one_hot_encode", "label_encode"],
  log_transform: ["log1p_transform"],
};

function isValidProposal(action) {
  if (!action || typeof action !== "object" || Array.isArray(action)) return false;
  if (!action.action) return false;
  // A proposal is only confirmable if the server registered it (proposal_id).
  if (typeof action.proposal_id !== "string" || !action.proposal_id) return false;

  if (action.kind === "cleaning") {
    const allowed = CLEANING_ACTIONS[action.step];
    if (!allowed?.includes(action.action)) return false;

    const required = ["missing_value_center", "outlier_center", "category_merge_center", "text_cleaning"];
    if (required.includes(action.step) && (typeof action.column !== "string" || !action.column.trim())) return false;
    if (action.column != null && typeof action.column !== "string") return false;

  } else if (action.kind === "feature_engineering") {
    const allowed = FEATURE_ACTIONS[action.category];
    if (!allowed?.includes(action.action)) return false;

    const expected = action.category === "ratio" ? 2 : 1;
    if (
      !Array.isArray(action.columns) ||
      action.columns.length !== expected ||
      !action.columns.every(c => typeof c === "string" && c.trim()) ||
      new Set(action.columns).size !== action.columns.length
    ) return false;

    if (action.new_column_name != null &&
        (typeof action.new_column_name !== "string" || !action.new_column_name.trim())) return false;
  } else {
    return false;
  }

  return action.params == null ||
    (typeof action.params === "object" && !Array.isArray(action.params));
}

function proposalLabel(action) {
  if (!action) return "Invalid proposal";

  if (action.kind === "cleaning") {
    const step = STEP_LABELS[action.step] || action.step || "Cleaning";
    const description = action.action || "Action not specified";
    return `${step} — ${description}${action.column ? ` on ${action.column}` : ""}`;
  }

  if (action.kind === "feature_engineering") {
    const category = STEP_LABELS[action.category] || action.category || "Feature Engineering";
    const description = action.action || "Action not specified";
    const columns = Array.isArray(action.columns) ? action.columns.join(", ") : "";
    return `${category} — ${description}${columns ? ` on ${columns}` : ""}`;
  }

  return "Unsupported proposal type";
}

function ProposalCard({ action, reason, confidence, status, onConfirm, onDismiss, busy }) {
  const valid = isValidProposal(action);

  return (
    <div className="mt-2 rounded-lg border border-graphite-border bg-graphite-900 p-3">
      <p className="font-mono text-[10px] uppercase tracking-wider text-signal-amber">
        Proposed action
      </p>
      <p className="mt-1 text-xs text-ink-100">{proposalLabel(action)}</p>
      {reason && <p className="mt-1 text-xs text-ink-500">{reason}</p>}
      {confidence != null && (
        <div className="mt-2">
          <ConfidenceMeter confidence={confidence} label="Confidence" />
        </div>
      )}

      {status === "pending" && (
        <div className="mt-3 flex gap-2">
          <button
            onClick={onConfirm}
            disabled={!valid || busy}
            className="rounded bg-signal-amber px-3 py-1 font-mono text-xs font-medium text-graphite-950 hover:bg-signal-amber/90 disabled:opacity-40"
          >
            {busy ? "Applying..." : "Confirm"}
          </button>
          <button
            onClick={onDismiss}
            disabled={busy}
            className="rounded border border-graphite-border px-3 py-1 font-mono text-xs text-ink-300 hover:bg-graphite-700"
          >
            Not now
          </button>
        </div>
      )}

      {!valid && status === "pending" && (
        <p className="mt-2 text-xs text-signal-red">
          This proposal has an invalid structure and cannot be applied.
        </p>
      )}
      {status === "applied" && (
        <p className="mt-2 font-mono text-[11px] text-signal-green">✓ Applied</p>
      )}
      {status === "dismissed" && (
        <p className="mt-2 font-mono text-[11px] text-ink-700">Dismissed</p>
      )}
      {status === "unavailable" && (
        <p className="mt-2 font-mono text-[11px] text-ink-700">
          No longer available (expired, stale or already handled). Ask Copilot again.
        </p>
      )}
    </div>
  );
}

export default function CopilotPanel({ datasetId, onActionApplied }) {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        'Ask me anything about this dataset, or say something like "clean up the missing values in revenue" and I will propose it here first.',
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const confirmingRef = useRef(false);
  const [error, setError] = useState(null);
  const [awaitingContinue, setAwaitingContinue] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, loading]);

  const toApiHistory = (msgs) =>
    msgs
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role, content: m.content }));

  const send = async (text) => {
    if (!text.trim() || loading) return;

    const userMsg = { role: "user", content: text };
    const nextMessages = [...messages, userMsg];

    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    setError(null);
    setAwaitingContinue(false);

    try {
      const res = await sendCopilotMessage(datasetId, text, toApiHistory(messages));
      setMessages([
        ...nextMessages,
        {
          role: "assistant",
          content: res.reply,
          proposedAction: res.proposed_action,
          proposalStatus: res.proposed_action ? "pending" : null,
          hasMoreSuggestions: res.has_more_suggestions,
          unverifiedFigures: res.unverified_figures,
        },
      ]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleConfirm = async (msgIndex) => {
    const msg = messages[msgIndex];
    if (
      confirmingRef.current ||
      loading ||
      msg?.proposalStatus !== "pending" ||
      !isValidProposal(msg?.proposedAction)
    ) return;

    confirmingRef.current = true;

    setLoading(true);
    setError(null);

    try {
      const res = await confirmCopilotAction(datasetId, msg.proposedAction);

      setMessages((prev) => {
        const copy = [...prev];
        copy[msgIndex] = { ...copy[msgIndex], proposalStatus: "applied" };
        copy.push({
          role: "assistant",
          content: res.result_summary,
          isResult: true,
        });
        return copy;
      });

      onActionApplied?.();
      if (res.has_more_suggestions) setAwaitingContinue(true);
    } catch (e) {
      setError(e.message);
      // The server refused because the proposal is stale/expired/already handled: stop offering Confirm.
      if (/stale|expired|already|dismissed|rejected|no longer|not found/i.test(e.message || "")) {
        setMessages((prev) => {
          const copy = [...prev];
          if (copy[msgIndex]?.proposalStatus === "pending") copy[msgIndex] = { ...copy[msgIndex], proposalStatus: "unavailable" };
          return copy;
        });
      }
    } finally {
      confirmingRef.current = false;
      setLoading(false);
    }
  };

  const handleDismiss = async (msgIndex) => {
    if (confirmingRef.current || loading) return;
    const msg = messages[msgIndex];
    if (msg?.proposalStatus !== "pending") return;

    confirmingRef.current = true; // block Confirm while the server records the dismissal
    setError(null);
    try {
      if (msg.proposedAction?.proposal_id) {
        await dismissCopilotAction(datasetId, msg.proposedAction.proposal_id);
      }
      setMessages((prev) => {
        if (prev[msgIndex]?.proposalStatus !== "pending") return prev;
        const copy = [...prev];
        copy[msgIndex] = { ...copy[msgIndex], proposalStatus: "dismissed" };
        return copy;
      });
    } catch (e) {
      setError(e.message); // e.g. 409: it was already confirmed, so it must not look dismissed
    } finally {
      confirmingRef.current = false;
    }
  };

  return (
    <div className="flex h-full flex-col rounded-lg border border-graphite-border bg-graphite-800">
      <div className="border-b border-graphite-border px-3 py-2">
        <p className="font-mono text-[11px] uppercase tracking-wider text-signal-amber">
          Copilot
        </p>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 space-y-3 overflow-y-auto px-3 py-3"
        style={{ minHeight: 280, maxHeight: 420 }}
      >
        {messages.map((m, i) => (
          <div key={i}>
            <div
              className={`rounded-lg px-3 py-2 text-xs ${
                m.role === "user"
                  ? "ml-4 bg-graphite-700 text-ink-100"
                  : "mr-1 bg-graphite-900 text-ink-300"
              }`}
            >
              {m.content}
              {m.unverifiedFigures?.length > 0 && (
                <p className="mt-1 font-mono text-[10px] text-signal-red">
                  ⚠ unverified: {m.unverifiedFigures.map((f) => f.figure).join(", ")}
                </p>
              )}
            </div>

            {m.proposedAction && (
              <ProposalCard
                action={m.proposedAction}
                reason={m.proposedAction.reason}
                confidence={m.proposedAction.confidence}
                status={m.proposalStatus}
                onConfirm={() => handleConfirm(i)}
                onDismiss={() => handleDismiss(i)}
                busy={loading}
              />
            )}
          </div>
        ))}

        {loading && <LoadingState message="Thinking..." compact />}
        {error && <ErrorState message={error} compact />}

        {awaitingContinue && !loading && (
          <div className="flex items-center gap-2">
            <span className="font-mono text-[11px] text-ink-500">
              Keep going with more suggestions?
            </span>
            <button
              onClick={() => send("Yes, give me another suggestion.")}
              className="rounded border border-signal-amber/50 px-2 py-0.5 font-mono text-[11px] text-signal-amber hover:bg-signal-amber/10"
            >
              Yes
            </button>
            <button
              onClick={() => setAwaitingContinue(false)}
              className="rounded border border-graphite-border px-2 py-0.5 font-mono text-[11px] text-ink-300 hover:bg-graphite-700"
            >
              No
            </button>
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex gap-2 border-t border-graphite-border p-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask or tell the copilot what to do..."
          className="flex-1 rounded border border-graphite-border bg-graphite-900 px-2 py-1.5 text-xs text-ink-100 placeholder:text-ink-700 focus:border-signal-amber focus:outline-none"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="rounded bg-signal-amber px-3 py-1.5 font-mono text-xs font-medium text-graphite-950 hover:bg-signal-amber/90 disabled:opacity-40"
        >
          Send
        </button>
      </form>
    </div>
  );
}
