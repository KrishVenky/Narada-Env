# Root-level entry point for openenv validate compatibility.
import sys
import os

_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "envs")
if _src not in sys.path:
    sys.path.insert(0, _src)

from clindetect.server.app import app, run  # noqa: F401


def main():
    run()


if __name__ == "__main__":
    main()
