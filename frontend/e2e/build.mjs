// Bundles the REAL WorkbenchPage for the jsdom end-to-end check (see run-e2e.sh). Dev tooling only:
// esbuild/jsdom are installed into a temp dir, never added to package.json.
import { createRequire } from "module";
import path from "path";
const WORK = process.env.E2E_WORK, PORT = process.env.E2E_PORT || "8124";
const { build } = createRequire(WORK + "/")("esbuild");
await build({
  entryPoints: [path.resolve("entry.jsx")], bundle: true, outfile: path.join(WORK, "bundle.cjs"), platform: "node", format: "cjs",
  jsx: "automatic", nodePaths: [path.join(WORK, "node_modules"), path.resolve("../node_modules")],
  loader: { ".css": "empty", ".svg": "empty", ".png": "empty" },
  define: { "import.meta.env.VITE_API_URL": JSON.stringify(`http://127.0.0.1:${PORT}`), "import.meta.env.DEV": "false", "import.meta.env": "{}" },
  logLevel: "error",
});
