"""Entry point for one background job: `python bg_worker.py` ← JSON {fn, args} on stdin → JSON on stdout.
Everything the job logs goes to stderr so stdout stays a clean JSON document."""
from __future__ import annotations

import json
import sys


def main() -> int:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    job = json.loads(sys.stdin.read() or "{}")
    import bg_compute
    fn = bg_compute.JOBS[job["fn"]]
    result = fn(*job.get("args", []))
    sys.stdout.write(json.dumps(result, default=str))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
