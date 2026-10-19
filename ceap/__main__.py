import argparse
import json
from pathlib import Path
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="CEAP Lab local server and reproducible experiment runner")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve"); serve.add_argument("--port", type=int, default=8000); serve.add_argument("--host", default="127.0.0.1")
    experiment = sub.add_parser("experiment")
    experiment.add_argument("--config", type=Path)
    experiment.add_argument("--episodes", type=int, default=200)
    experiment.add_argument("--policies", default="ceap,immediate,fixed,threshold,weakest")
    experiment.add_argument("--output", type=Path, default=Path("data/cli-experiment"))
    experiment.add_argument("--kind", default="E3")
    args = parser.parse_args()
    if args.command == "experiment":
        from .experiments import execute_job
        data = json.loads(args.config.read_text()) if args.config else Config().model_dump()
        if "config" in data:
            payload = data
            payload["episodes"] = args.episodes
        else:
            payload = {"config": Config.model_validate(data).model_dump(), "episodes": args.episodes,
                       "policies": args.policies.split(","), "experiment": args.kind}
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Output directory must be empty to preserve reproducibility.")
        args.output.mkdir(parents=True, exist_ok=True)
        execute_job(args.output, payload)
        print((args.output/"progress.json").read_text())
    else:
        import uvicorn
        uvicorn.run("ceap.app:app", host=getattr(args, "host", "127.0.0.1"), port=getattr(args, "port", 8000))


if __name__ == "__main__": main()
