# Pando engine and API

Pando is a word-chain game. This repository contains the model-backed
connectivity engine, durable game state, and the HTTP API consumed by the web
interface.

## Local setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m gensim.downloader --download word2vec-google-news-300
```

The model download only happens once. On its first real use, the engine creates
a native cached representation and builds the canonical word pool. Both are
reused by later runs.

Start the API from the repository root:

```bash
PANDO_DATABASE_PATH=./data/pando.sqlite3 \
  .venv/bin/uvicorn algorithm.api:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is then available at
`http://127.0.0.1:8000/docs`.

## Browser connection

During local development, the browser identifies the current player with the
`X-Player-ID` header. Register the player after login or app initialization:

```js
const API_URL = import.meta.env.VITE_API_URL ?? "/api";
const playerHeaders = {
  "Content-Type": "application/json",
  "X-Player-ID": currentUser.id,
};

await fetch(`${API_URL}/me`, {
  method: "PUT",
  headers: playerHeaders,
  body: JSON.stringify({ display_name: currentUser.displayName }),
});
```

The main game operations are:

```js
export async function saveDraft(start, end, words) {
  return request(`/puzzles/${encodeURIComponent(start)}/${encodeURIComponent(end)}/draft`, {
    method: "PUT",
    body: JSON.stringify({ words }),
  });
}

export async function clearDraft(start, end) {
  return request(`/puzzles/${encodeURIComponent(start)}/${encodeURIComponent(end)}/draft`, {
    method: "DELETE",
  });
}

export async function submitSequence(start, end, words) {
  return request(`/puzzles/${encodeURIComponent(start)}/${encodeURIComponent(end)}/submissions`, {
    method: "POST",
    body: JSON.stringify({ words }),
  });
}

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { ...playerHeaders, ...options.headers },
  });
  const result = await response.json();
  if (!response.ok) throw result;
  return result;
}
```

Use these read endpoints to populate the interface:

- `GET /api/puzzles/{start}/{end}/draft`
- `GET /api/puzzles/{start}/{end}/submissions`
- `GET /api/puzzles/{start}/{end}/personal-best`
- `GET /api/puzzles/{start}/{end}/leaderboard` (public)

Invalid submissions return HTTP 422 with the full verification result,
including each failed link and its reason.

## Production configuration

```bash
PANDO_DATABASE_PATH=/var/lib/pando/pando.sqlite3
PANDO_AUTH_MODE=trusted-header
PANDO_PLAYER_ID_HEADER=X-Authenticated-User-ID
PANDO_WARM_MODEL=1
PANDO_CORS_ORIGINS=https://pando.example.com
```

In `trusted-header` mode, deploy the API behind an authentication proxy. The
proxy must remove any incoming identity header and replace it with the verified
account ID. Do not expose the API directly when this mode is enabled.

Prefer serving the website and `/api` from the same public origin. If they use
different origins, list the web origins in `PANDO_CORS_ORIGINS`, separated by
commas.

Run one API worker initially. The word-vector model is large, and every worker
process maps its own model. SQLite is configured with WAL mode and a write busy
timeout; move to PostgreSQL before running multiple API instances or handling
heavy concurrent writes.

## API contract

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process health check |
| `PUT` | `/api/me` | Register or rename the authenticated player |
| `PUT` | `/api/puzzles/{start}/{end}/draft` | Create or replace a draft |
| `GET` | `/api/puzzles/{start}/{end}/draft` | Load the draft |
| `DELETE` | `/api/puzzles/{start}/{end}/draft` | Clear only the draft |
| `POST` | `/api/puzzles/{start}/{end}/submissions` | Verify and save a sequence |
| `GET` | `/api/puzzles/{start}/{end}/submissions` | List the player's submissions |
| `GET` | `/api/puzzles/{start}/{end}/personal-best` | Load the player's best |
| `GET` | `/api/puzzles/{start}/{end}/leaderboard` | Load global best scores |

Connector words are the score. Fixed start and end words are not counted.

## Tests

The store and API suites use fake verification and do not load the model:

```bash
.venv/bin/python -m unittest algorithm.test_game_store algorithm.test_api -v
```

The connectivity suite uses the downloaded production model:

```bash
.venv/bin/python -m unittest algorithm.test_connectivity -v
```
