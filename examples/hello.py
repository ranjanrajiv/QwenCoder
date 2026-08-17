"""A harmless program for exercising the sandbox by hand.

    python -m python_dpo sandbox run --file examples/hello.py

Prints a greeting and then reports a few properties of the environment it finds itself in,
which is a quick way to see the isolation actually holding: the UID is non-root, the Docker
socket is absent, and no host environment variable is visible.
"""

import os

print("hello from the sandbox")
print(f"uid={os.getuid()} (non-root when the sandbox is configured correctly)")
print(f"docker socket present: {os.path.exists('/var/run/docker.sock')}")
print(f"cwd: {os.getcwd()}")
print(f"HF_TOKEN visible: {os.environ.get('HF_TOKEN') is not None}")
