// End-to-end UI check: renders the real WorkbenchPage in jsdom against a live backend (see run-e2e.sh).
import { createRequire } from "module";
import { readFileSync } from "fs";
const WORK = process.env.E2E_WORK;
const require = createRequire(WORK + "/");
const { JSDOM } = require("jsdom");

const BASE = `http://127.0.0.1:${process.env.E2E_PORT || "8124"}`;
let failures = 0;
const check = (name, ok, extra = "") => { console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  -> " + extra : ""}`); if (!ok) failures++; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, what, timeout = 8000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) { try { const v = fn(); if (v) return v; } catch {} await sleep(30); }
  throw new Error("timeout waiting for: " + what);
}

// ---- browser globals
const dom = new JSDOM("<!doctype html><div id=root></div>", { url: BASE, pretendToBeVisual: true });
for (const k of ["window", "document", "navigator", "HTMLElement", "Node", "MutationObserver", "getComputedStyle", "Element", "Event", "MouseEvent"])
  try { globalThis[k] = dom.window[k]; } catch { Object.defineProperty(globalThis, k, { value: dom.window[k], configurable: true }); }
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
globalThis.requestAnimationFrame = (cb) => setTimeout(cb, 16);
globalThis.matchMedia = globalThis.window.matchMedia = () => ({ matches: false, addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {} });
globalThis.window.scrollTo = () => {}; dom.window.Element.prototype.scrollIntoView = () => {}; dom.window.Element.prototype.scrollTo = () => {}; dom.window.HTMLElement.prototype.scrollTo = () => {};
globalThis.IS_REACT_ACT_ENVIRONMENT = false;

// ---- instrumented fetch (records every request; can simulate a network failure on dismiss)
const realFetch = globalThis.fetch; const log = []; let failDismiss = false;
globalThis.fetch = async (url, opts = {}) => {
  const method = (opts.method || "GET").toUpperCase(); log.push({ method, url: String(url) });
  if (failDismiss && String(url).includes("/dismiss")) throw new TypeError("Failed to fetch");
  return realFetch(url, opts);
};
const post = (p) => log.filter((r) => r.method === "POST" && r.url.includes(p));

const { mount, mountCrash } = require(WORK + "/bundle.cjs");
const api = async (path, opts) => (await realFetch(BASE + path, opts)).json();

// ---- upload a real dataset to the real backend
const form = new FormData(); form.append("file", new Blob([readFileSync(WORK + "/data.csv")], { type: "text/csv" }), "ui_test.csv");
const dataset = await (await realFetch(BASE + "/api/dataset/upload", { method: "POST", body: form })).json();
const ds = dataset.dataset_id; check("uploaded dataset", !!ds, ds);
const state = async () => ({ q: await api(`/api/dataset/${ds}/quality`), h: await api(`/api/dataset/${ds}/history`) });
// The UI updates optimistically; wait until the server has committed before asserting on its state.
async function stateWhere(pred) {
  const t0 = Date.now(); let st;
  while (Date.now() - t0 < 8000) { st = await state(); if (pred(st)) return st; await sleep(50); }
  return st;
}

const container = document.getElementById("root");
const click = (el) => el.dispatchEvent(new window.MouseEvent("click", { bubbles: true, cancelable: true }));
const buttons = () => [...container.querySelectorAll("button")];
const btn = (re) => buttons().find((b) => re.test(b.textContent.trim()));
const text = () => container.textContent;
const cards = () => [...container.querySelectorAll("span")].filter((s) => s.textContent.startsWith("Recommended action:")).map((s) => {
  const footer = s.parentElement; const bs = [...footer.querySelectorAll("button")].map((b) => b.textContent.trim());
  return { action: s.textContent.replace("Recommended action:", "").trim(), buttons: bs };
});
const openCleaning = async () => { await waitFor(() => btn(/^Cleaning$/), "Cleaning nav"); click(btn(/^Cleaning$/)); await waitFor(() => /Cleaning recommendations \(/.test(text()), "cleaning tab"); };

let unmount = mount(container, dataset);
await openCleaning();
await waitFor(() => cards().length === 5, "5 recommendation cards");
const v0 = (await state()).h.current_version;

// 1. outlier card: no Apply, has Mark as reviewed + info note
let outlier = cards().find((c) => c.action === "flag_for_review");
check("outlier card is rendered", !!outlier);
check("outlier card has NO Apply button", !outlier.buttons.some((b) => /^Apply/.test(b)), JSON.stringify(outlier.buttons));
check("outlier card offers 'Mark as reviewed'", outlier.buttons.includes("Mark as reviewed"));
check("review-only explanation is shown", /Review-only: these values are statistically unusual/.test(text()));
const others = cards().filter((c) => c.action !== "flag_for_review" && c.action !== "fuzzy_merge");
check("other (supported) cards still have Apply", others.length >= 3 && others.every((c) => c.buttons.includes("Apply")), others.map((c) => c.action).join(","));

// 2. dismiss the outlier -> persisted on server, no data change, no version, no apply request
click(btn(/^Mark as reviewed$/));
await waitFor(() => cards().length === 4, "card removed");
check("header shows 4 open · 1 dismissed", /Cleaning recommendations \(4 open · 1 dismissed\)/.test(text()));
await waitFor(() => post("/dismiss").length === 1, "dismiss request");
let st = await stateWhere((x) => x.q.recommendation_counts.dismissed === 1);
check("server persisted exactly one dismissed recommendation", st.q.recommendation_counts.dismissed === 1 && st.q.recommendation_counts.open === 4, JSON.stringify(st.q.recommendation_counts));
check("dismissal created NO new version / history", st.h.current_version === v0 && st.h.history.length === 0, `v=${st.h.current_version}`);
check("no apply-action request was sent", post("/apply-action").length === 0);

// 3. 'reload': unmount, mount a brand-new page instance -> still dismissed
unmount(); container.innerHTML = ""; unmount = mount(container, dataset); await openCleaning();
await waitFor(() => cards().length === 4, "4 cards after reload");
check("dismissal survives a page reload", cards().length === 4 && !cards().some((c) => c.action === "flag_for_review"));
check("dismissed count still shown after reload", /4 open · 1 dismissed/.test(text()));

// 4. show dismissed + restore
click(btn(/^Show dismissed \(1\)$/));
await waitFor(() => btn(/^Restore$/), "restore button");
check("dismissed list is shown with the outlier item", /Dismissed \(1\)/.test(text()) && /outlier center/.test(text()));
click(btn(/^Restore$/));
await waitFor(() => cards().length === 5, "restored card");
st = await stateWhere((x) => x.q.recommendation_counts.dismissed === 0);
check("restore re-opens it on the server", st.q.recommendation_counts.dismissed === 0 && st.q.recommendation_counts.open === 5);
check("restore changed no data/version", st.h.current_version === v0 && st.h.history.length === 0);

// 5. failed dismiss rolls back and shows an error (no silent loss)
failDismiss = true;
click(buttons().find((b) => /^Dismiss$/.test(b.textContent.trim())));
await waitFor(() => /Failed to fetch/.test(text()), "error message");
await waitFor(() => cards().length === 5, "rollback");
check("failed dismiss rolls back the card and shows the error", cards().length === 5 && /Failed to fetch/.test(text()));
failDismiss = false;
st = await state(); check("server unchanged after the failed dismiss", st.q.recommendation_counts.dismissed === 0);

// 6. dismiss all five -> 'all dismissed' state (distinct from 'none')
for (let i = 0; i < 5; i++) {
  const b = buttons().find((x) => /^(Dismiss|Ignore|Mark as reviewed)$/.test(x.textContent.trim()));
  const n = cards().length; click(b); await waitFor(() => cards().length === n - 1, `dismiss ${i + 1}`);
}
await waitFor(() => /All recommendations dismissed/.test(text()), "all-dismissed state");
check("all dismissed -> distinct 'All recommendations dismissed' state", /All recommendations dismissed/.test(text()) && !/No recommendations available/.test(text()));
check("state explains nothing was applied", /Nothing was applied and your data is unchanged/.test(text()));
check("'Show dismissed' control is offered", !!btn(/^Show dismissed$/));
st = await stateWhere((x) => x.q.recommendation_counts.dismissed === 5); check("server: 5 dismissed, 0 open, still version 1", st.q.recommendation_counts.dismissed === 5 && st.h.current_version === v0);

// 7. reload -> still 'all dismissed'; recompute never clears it
unmount(); container.innerHTML = ""; unmount = mount(container, dataset); await openCleaning();
await waitFor(() => /All recommendations dismissed/.test(text()), "all-dismissed after reload");
check("all-dismissed state persists across reload", true);
check("still zero apply-action requests in the whole session", post("/apply-action").length === 0);

// 8. a clean dataset shows 'none', not 'all dismissed'
const f2 = new FormData(); f2.append("file", new Blob(["a,b\n" + Array.from({ length: 30 }, (_, i) => `${i},${30 - i}`).join("\n")], { type: "text/csv" }), "clean.csv");
const clean = await (await realFetch(BASE + "/api/dataset/upload", { method: "POST", body: f2 })).json();
unmount(); container.innerHTML = ""; unmount = mount(container, clean); await openCleaning();
await waitFor(() => /No recommendations available|All recommendations dismissed|Cleaning recommendations \(0/.test(text()), "clean empty state");
check("clean dataset shows 'No recommendations available' (not 'all dismissed')", /No recommendations available/.test(text()) && !/All recommendations dismissed/.test(text()), text().slice(0, 0));
await waitFor(() => btn(/^Machine Learning$/), "Machine Learning navigation");
click(btn(/^Machine Learning$/));
await waitFor(() => { const b = btn(/^Train and evaluate$/); return /Train and evaluate a model/.test(text()) && b && !b.disabled; }, "supervised model workspace");
check("machine learning workspace is reachable", /Train and evaluate a model/.test(text()));
click(btn(/^Train and evaluate$/));
await waitFor(() => /Evaluation complete/.test(text()), "held-out regression evaluation", 15000);
check("regression evaluation reports held-out metrics", /rmse/i.test(text()) && /held-out rows/.test(text()));
check("held-out predictions are shown", /Held-out predictions/.test(text()));
check("cross-validation is reported next to the holdout metrics", /5-fold cross-validation on the training rows \(r2\)/.test(text()));
click(btn(/^Report$/));
await waitFor(() => /Smart Report Builder/.test(text()) && /regression_model/.test(text()), "model run in report workspace", 15000);
check("model evaluation can be included in a report", /regression_model/.test(text()));

// A render crash is contained by the error boundary (no blank page)
unmount(); container.innerHTML = "";
{
  const origError = console.error; console.error = () => {};
  unmount = mountCrash(container);
  await waitFor(() => /Something went wrong/.test(text()), "error boundary message");
  console.error = origError;
  check("render crash shows a recoverable message with a Reload button", !!btn(/^Reload$/) && !!container.querySelector('[role="alert"]'));
}

unmount();
console.log(failures === 0 ? "\nALL UI CHECKS PASSED" : `\n${failures} UI CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
