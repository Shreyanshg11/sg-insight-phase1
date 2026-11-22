import { spawn, spawnSync } from "node:child_process";
import { createWriteStream, existsSync, mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import net from "node:net";
import { once } from "node:events";

const frontendDir = path.resolve(".");
const e2eDir = path.join(frontendDir, "e2e");
const rootDir = path.resolve(frontendDir, "..");
const backendDir = path.join(rootDir, "backend");
const pythonCandidates = [
  path.join(backendDir, ".venv", "Scripts", "python.exe"),
  path.join(backendDir, ".venv313", "Scripts", "python.exe"),
  path.join(backendDir, ".venv", "bin", "python"),
];
const python = process.env.PYTHON || pythonCandidates.find((candidate) => existsSync(candidate)
  && spawnSync(candidate, ["-c", "import fastapi, pandas, sklearn, uvicorn"], { stdio: "ignore" }).status === 0)
  || (process.platform === "win32" ? "py" : "python3");

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { stdio: "inherit", shell: false, ...options });
  if (result.status !== 0) throw new Error(`${command} exited with code ${result.status ?? result.error?.message}`);
}

async function unusedPort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => server.once("error", reject).listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

const work = mkdtempSync(path.join(tmpdir(), "sg-insight-e2e-"));
const port = Number(process.env.E2E_PORT || await unusedPort());
const env = { ...process.env, E2E_WORK: work, E2E_PORT: String(port), SG_INSIGHT_DATA_DIR: path.join(work, "data"),
  PYTHONDONTWRITEBYTECODE: "1", SG_INSIGHT_AUTH_PASSWORD_HASH: "", SG_INSIGHT_SESSION_SECRET: "", ENVIRONMENT: "development", AI_PROVIDER: "ollama" };
let server;
let backendLog;
try {
  const availableDeps = path.join(frontendDir, "node_modules", ".codex-e2e-deps", "node_modules");
  const tempDeps = path.join(work, "node_modules");
  if (!existsSync(path.join(availableDeps, "esbuild")) || !existsSync(path.join(availableDeps, "jsdom"))) {
    run("npm", ["install", "--prefix", work, "--no-save", "--silent", "esbuild", "jsdom"], { shell: process.platform === "win32" });
  }
  const modulePath = existsSync(path.join(availableDeps, "esbuild")) ? availableDeps : tempDeps;
  env.NODE_PATH = [modulePath, process.env.NODE_PATH].filter(Boolean).join(path.delimiter);
  mkdirSync(path.join(work, "data"), { recursive: true });
  run(python, ["-c", "import sys; sys.path.insert(0, '.'); from tests.test_recommendation_lifecycle import make_df; import os; make_df().to_csv(os.path.join(os.environ['E2E_WORK'], 'data.csv'), index=False)"], { cwd: backendDir, env });
  run(process.execPath, ["build.mjs"], { cwd: e2eDir, env });
  backendLog = createWriteStream(path.join(work, "backend.log"));
  server = spawn(python, ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", String(port), "--log-level", "warning"], { cwd: backendDir, env, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  server.stdout.pipe(backendLog);
  server.stderr.pipe(backendLog);
  let ready = false;
  for (let attempt = 0; attempt < 120; attempt++) {
    if (server.exitCode !== null) throw new Error(`Backend exited early. Log: ${path.join(work, "backend.log")}`);
    try { if ((await fetch(`http://127.0.0.1:${port}/api/healthz`, { signal: AbortSignal.timeout(1000) })).ok) { ready = true; break; } } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  if (!ready) throw new Error(`Backend did not become ready. Log: ${path.join(work, "backend.log")}`);
  run(process.execPath, ["run.mjs"], { cwd: e2eDir, env });
} finally {
  if (server && server.exitCode === null) {
    server.kill();
    await once(server, "exit");
  }
  if (backendLog && !backendLog.closed) { backendLog.end(); await once(backendLog, "close"); }
  rmSync(work, { recursive: true, force: true });
}
