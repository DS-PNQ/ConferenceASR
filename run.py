import uvicorn
import yaml

with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=cfg.get("host", "127.0.0.1"),
                port=int(cfg.get("port", 8000)), reload=False)
