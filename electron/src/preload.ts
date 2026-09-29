// Preload: nothing privileged needed (renderer talks to localhost directly).
// Kept as a stable bridge point for future native hooks.
import { contextBridge } from "electron";

contextBridge.exposeInMainWorld("conflive", {
  version: "1.0.0",
  backendPort: 8000,
});
