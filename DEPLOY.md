# DEPLOY.md

Deployment instructions for the `yomi` container.

## Overview

| Item | Value |
|---|---|
| Container name | `yomi` |
| Image name | `yomi` |
| Internal port | `8080` |
| Host port | `3047` |
| Public URL | https://yomi.eulerai.net |
| Auth | Authelia SSO (nginx → /api/authz/auth-request) |
| Storage | none persistent |
| Secrets | `~/secrets/` bind-mounted read-only at `/secrets` |
| Memory limit | 8192 MiB |
| CPU limit | 6.0 |
| Log retention | 3 × 10 MiB rotated json-file |

## Prerequisites on host

- Docker + `docker compose` plugin
- Wildcard DNS `*.eulerai.net` (already in place)
- nginx with the shared `eulerai` site config and Authelia snippets installed (already in place; see `/etc/nginx/snippets/authelia-*.conf`)
- Public host port `3047` not in use by any other app
- `~/secrets/` populated with whatever this app needs (chmod 700 on the dir, 600 on files)

## First-time setup

```sh
cd ~/yomi
docker compose up -d --build
```

## Redeploy

```sh
cd ~/yomi
git pull
docker compose up -d --build
```

## docker-compose.yml

```yaml
services:
  yomi:
    build: .
    image: yomi
    container_name: yomi
    restart: unless-stopped
    ports:
      - "127.0.0.1:3047:8080"
    volumes:
      - ${HOME}/secrets:/secrets:ro
    environment:
      - TZ=America/Chicago
    mem_limit: 8192m
    cpus: 6.0
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:8080/ >/dev/null 2>&1 || exit 1"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

## nginx server block

Add this to `/etc/nginx/sites-enabled/eulerai` (then `sudo nginx -t && sudo systemctl reload nginx`):

```nginx
server {
    listen 443 ssl;
    server_name yomi.eulerai.net;
    ssl_certificate /etc/letsencrypt/live/eulerai.net/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/eulerai.net/privkey.pem;
    include snippets/authelia-location.conf;
    include snippets/authelia-authrequest.conf;
    client_max_body_size 10m;
    location / {
        proxy_pass http://127.0.0.1:3047;
        proxy_http_version 1.1;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade           $http_upgrade;
        proxy_set_header Connection        "upgrade";
    }
}
```

## Auth (Authelia SSO)

This app sits behind the shared Authelia instance running on this box. The two `include` lines in the nginx server block are the only thing required — Authelia's `*.eulerai.net` rule already covers any new subdomain automatically, and the cookie scope (`domain=eulerai.net`) makes login at any one app valid for all of them.

- To add or change users: edit `~/authelia/config/users_database.yml`. Generate an argon2id hash with:
  ```sh
  docker run --rm authelia/authelia:4.39 authelia crypto hash generate argon2 --password 'newpassword'
  ```
  The file is hot-reloaded within ~1 minute (configured `refresh_interval: 1m`).
- To revoke all sessions everywhere: `cd ~/authelia && docker compose restart`.

## Secrets

All secrets live on the **host** under `~/secrets/` and are bind-mounted read-only into the container at `/secrets`. The container never sees the host filesystem outside that directory, and secrets are never baked into the image or checked into git.

Reference them via env vars that point at paths inside `/secrets`:

```yaml
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp/sa_tts_key.json
      - OPENAI_API_KEY_FILE=/secrets/openai/key.txt
```

If your app reads from `OPENAI_API_KEY` (string) rather than `OPENAI_API_KEY_FILE` (path), set it via `env_file: ~/secrets/<app>.env` or read at startup from the file. **Do not** put plaintext secret values into `docker-compose.yml` or commit a `.env` to the repo.

Host paths to maintain:
- `~/secrets/gcp/` — service account JSON files (chmod 600, dir 700)
- `~/secrets/<app>/` — per-app keys, tokens, and API credentials

## Restore on a fresh machine

The host is backed up daily, so disaster recovery is mostly about reassembling the parts on a new box:

```sh
# 1. Clone the repo
git clone <repo-url> ~/yomi

# 2. Restore stateful data from backup
#    (preserves DB, uploads, anything in ./data/)
rsync -a /path/to/backup/yomi/data/  ~/yomi/data/

# 3. Restore host secrets
rsync -a /path/to/backup/secrets/  ~/secrets/
chmod 700 ~/secrets && find ~/secrets -type f -exec chmod 600 {} \;

# 4. Bring it up
cd ~/yomi && docker compose up -d --build
```

Anything outside `~/<app>/data/`, the repo itself, and `~/secrets/` is reproducible from `git pull && docker compose up -d --build` — no other host state matters.

## Verify

```sh
docker ps --filter name=yomi
docker logs --tail 30 yomi
curl -s -o /dev/null -w "loopback HTTP %{http_code}\n" http://127.0.0.1:3047/
docker stats --no-stream yomi
```

Healthy idle target: well under 8192 MiB RSS, near-zero CPU. If RSS sits within ~10% of the cap, raise `mem_limit` in `docker-compose.yml` — the cgroup OOM-kills cleanly with exit 0, so a restart loop with no visible error is almost always memory.

## Footprint troubleshooting

- Restart loop with `exit 0` and no error in logs → almost always `mem_limit` too tight; raise it. Healthcheck child processes briefly double resident memory.
- High idle CPU on a Go app → ensure `GOMEMLIMIT` is set (and is ~90 % of `mem_limit`); without it Go's GC fights the cgroup ceiling.
- Container stops responding under load → check `docker stats` against `cpus` cap.
- Healthcheck failing but app works → the `wget` in the test command may not exist in your image. Replace with whatever your runtime image has (`curl`, or a no-op `["CMD", "true"]` to disable).
- 413 Request Entity Too Large from nginx → `client_max_body_size` is too low (default 1 MB); raise it in the server block.
