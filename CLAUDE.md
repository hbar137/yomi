# yomi

Train an LLM to convert Japanese text to yomi/furiganan. LLM needs to be lightweight and runable on CPU for inference. Training will be done in runpod. Need to collect data and prepare for training.

## CRITICAL: Always confirm before implementing

**NEVER start writing code or editing files without presenting the approach to the user first and getting explicit approval.** This applies to ALL changes — planned fixes, discovered issues, improvements, refactors, everything. Investigate and explain what you'd do, then wait for a "go ahead" before touching any files.

The only exception is if the user explicitly says "just do it" or "implement it" for a specific change.

## CRITICAL: Never run sudo commands directly

**Do not invoke `sudo` from your tools.** You don't have the user's password and the prompt will hang. Whenever a step needs root (editing files under `/etc/`, `systemctl`, `nginx -t`, `apt`, `chmod` on host paths, etc.), write the commands into a script under `/tmp/` (e.g. `/tmp/<app>-<task>.sh`), `chmod +x` it, then ask the user to run it themselves with `sudo bash /tmp/<app>-<task>.sh` (or `sudo /tmp/<app>-<task>.sh`). Make the script idempotent where possible, and have it `set -e` and echo what it's about to do before doing it. After the user confirms they ran it, verify the result via non-sudo means (e.g. `curl`, reading the file as the user, `docker ps`).

## Tech / runtime

- **Language / stack**: Decide later
- **Container internal port**: `8080`
- **Host port**: `3047` (publicly served at https://yomi.eulerai.net via nginx)
- **Storage**: no persistent storage required.
- **Auth**: nginx fronts the app and delegates to Authelia (`include snippets/authelia-*.conf` in the server block — see DEPLOY.md). The app itself does not handle login.
- **Secrets**: never embed secrets in the image or repo. The host's `~/secrets/` is bind-mounted read-only at `/secrets` inside the container; env vars point at paths under that mount (e.g. `GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp/sa_tts_key.json`).
- **Uploads**: nginx `client_max_body_size` is bumped to `10m` for this server block (default 1 MB would silently truncate).
- **Resource budget**: container limited to `8192m` RAM / `6.0` CPU. Keep idle RSS well below the cap so healthchecks and short-lived child processes have headroom (the cgroup OOM-kills cleanly with exit 0, which looks like a graceful shutdown in logs).

## Build & run

```bash
# Build image and (re)create container
docker compose up -d --build

# Logs
docker compose logs -f --tail 100

# Stop
docker compose down
```

Full deploy procedure in `DEPLOY.md`.
