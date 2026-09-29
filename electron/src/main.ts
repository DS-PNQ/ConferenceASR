// Main process: window + Python backend lifecycle (find, CUDA-detect, spawn, reap).
import { app, BrowserWindow, Menu } from "electron";
import { spawn, execFile, ChildProcess } from "child_process";
import * as path from "path";
import * as fs from "fs";

const PORT = 8000;
const BASE = `http://127.0.0.1:${PORT}`;
let backend: ChildProcess | null = null;
let win: BrowserWindow | null = null;

function log(...a: unknown[]) {
  console.log("[conflive]", ...a);
}

function locateBackend(): string {
  const env = process.env.CONF_LIVE_BACKEND;
  if (env && fs.existsSync(path.join(env, "run.py"))) return env;
  // dev layout: electron/ is a sibling of run.py; packaged: resources/backend
  const cands = [
    path.join(__dirname, "..", ".."), // electron/dist -> repo root (tsc/dev)
    path.join(process.resourcesPath || "", "backend"),
    path.join(app.getAppPath(), "..", "backend"),
  ];
  for (const d of cands) {
    if (d && fs.existsSync(path.join(d, "run.py"))) return d;
  }
  throw new Error("Python backend (run.py) not found. Set CONF_LIVE_BACKEND.");
}

function runCmd(cmd: string, args: string[], timeoutMs: number): Promise<{ code: number; out: string }> {
  return new Promise((resolve) => {
    let out = "";
    const p = spawn(cmd, args, { shell: false, windowsHide: true });
    const kill = setTimeout(() => { try { p.kill(); } catch { /* noop */ } }, timeoutMs);
    p.stdout?.on("data", (d) => (out += d.toString()));
    p.stderr?.on("data", (d) => (out += d.toString()));
    p.on("error", () => { clearTimeout(kill); resolve({ code: 127, out }); });
    p.on("close", (code) => { clearTimeout(kill); resolve({ code: code ?? 127, out }); });
  });
}

async function locatePython(backendDir: string): Promise<string> {
  // Must import the server deps — a bare venv (torch but no fastapi) is skipped.
  const check = ["-c", "import fastapi, uvicorn, yaml, numpy"];
  const venvWin = path.join(backendDir, ".venv", "Scripts", "python.exe");
  const cands = [
    ...(fs.existsSync(venvWin) ? [venvWin] : []),
    "py",
    "python",
  ];
  for (const c of cands) {
    const args = c === "py" ? ["-3", ...check] : check;
    log("probing python:", c);
    const r = await runCmd(c, args, 30000);
    if (r.code === 0) return c;
  }
  throw new Error("No Python with backend deps (need fastapi, uvicorn, pyyaml, numpy).");
}

async function cudaPresent(): Promise<boolean> {
  const r = await runCmd("nvidia-smi", ["-L"], 8000);
  return r.code === 0 && /gpu/i.test(r.out);
}

async function healthOk(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/api/health`);
    if (!res.ok) return false;
    const h = (await res.json()) as { ok?: boolean };
    return h.ok === true;
  } catch {
    return false;
  }
}

async function ensureBackend(): Promise<void> {
  if (await healthOk()) {
    log("attached to running backend");
    return;
  }
  const backendDir = locateBackend();
  const device = (await cudaPresent()) ? "cuda" : "cpu";
  log(`no backend on :${PORT}; starting on ${device} (${backendDir})`);
  const python = await locatePython(backendDir);
  backend = spawn(python, ["run.py"], {
    cwd: backendDir,
    windowsHide: true,
    env: { ...process.env, PYTHONPATH: backendDir, DEVICE: device },
  });
  backend.stdout?.on("data", (d) => log("[backend]", d.toString().trimEnd()));
  backend.stderr?.on("data", (d) => log("[backend:err]", d.toString().trimEnd().slice(0, 500)));
  backend.on("exit", (code) => log("backend exited:", code));
  const deadline = Date.now() + 3 * 60 * 1000;
  for (;;) {
    if (backend.exitCode !== null && backend.exitCode !== undefined) {
      throw new Error(`Backend exited early (code ${backend.exitCode}).`);
    }
    if (await healthOk()) {
      log("backend healthy");
      return;
    }
    if (Date.now() > deadline) throw new Error("Backend did not answer /api/health in 3 min.");
    await new Promise((r) => setTimeout(r, 1500));
  }
}

async function createWindow(): Promise<void> {
  // No native menu bar: the UI is fully custom (sidebar nav). The default
  // Electron menu renders as a light strip that clashes with the design.
  Menu.setApplicationMenu(null);
  win = new BrowserWindow({
    width: 1200,
    height: 780,
    minWidth: 980,
    minHeight: 640,
    title: "ConfLive",
    backgroundColor: "#F2F2F7",
    autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, "preload.js") },
  });
  win.setMenu(null);
  await win.loadFile(path.join(__dirname, "..", "ui", "dist", "index.html"));
  win.on("closed", () => (win = null));
}

app.whenReady().then(async () => {
  try {
    await ensureBackend();
  } catch (e) {
    log("backend failed:", (e as Error).message);
  }
  await createWindow();
});

app.on("window-all-closed", () => {
  try {
    backend?.kill();
  } catch { /* noop */ }
  if (process.platform !== "darwin") app.quit();
});
