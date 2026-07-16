import { useState, useRef } from "react";
import { uploadDataset } from "../api/client";
import ErrorState from "../components/ui/ErrorState";

export default function UploadPage({ onUploaded }) {
  const [goal, setGoal] = useState("");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const handleFile = async (file) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const result = await uploadDataset(file, goal.trim() || null);
      onUploaded(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl px-4 py-10 sm:px-6 sm:py-20">
      <p className="font-mono text-xs uppercase tracking-widest text-signal-amber">SG-Insight</p>
      <h1 className="mt-2 font-display text-2xl font-semibold text-ink-100 sm:text-3xl">
        Upload a dataset to begin
      </h1>
      <p className="mt-2 max-w-lg text-sm text-ink-500">
        CSV or Excel. Every cleaning step from here is proposed with a reason and a
        confidence reading — you confirm before anything changes.
      </p>

      <label className="mt-8 block text-xs font-medium uppercase tracking-wider text-ink-500">
        Investigation goal <span className="text-ink-700">(optional, but recommended)</span>
      </label>
      <input
        value={goal}
        onChange={(e) => setGoal(e.target.value)}
        placeholder="e.g. find why revenue dropped in Q2"
        className="mt-2 w-full rounded border border-graphite-border bg-graphite-800 px-3 py-2 text-sm text-ink-100 placeholder:text-ink-700 focus:border-signal-amber focus:outline-none"
      />

      <div
        role="button"
        tabIndex={0}
        aria-label="Upload dataset — drop a CSV or Excel file, or press Enter to browse"
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFile(e.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        className={`mt-6 flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-10 text-center transition-colors focus-visible:outline-none sm:px-6 sm:py-14 ${
          dragging ? "border-signal-amber bg-graphite-800" : "border-graphite-border bg-graphite-900"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <p className="font-mono text-sm text-ink-300">
          {loading ? "Uploading…" : "Drop a .csv or .xlsx file, or click to browse"}
        </p>
      </div>

      {error && <div className="mt-4"><ErrorState message={error} onRetry={() => setError(null)} /></div>}
    </div>
  );
}
