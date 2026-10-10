// Preload: the renderer talks to the backend on localhost directly; the only
// native hook is recolouring the title-bar caption buttons on theme change.
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("conflive", {
  version: "1.0.0",
  backendPort: 8000,
  setTheme: (theme: string) => ipcRenderer.send("theme", theme),
});
