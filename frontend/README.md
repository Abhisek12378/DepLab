# DepLab web

React and TypeScript interface for the DepLab conversation API.

## Local development

1. Start the FastAPI service from the repository root on port 8000.
2. Install and start this frontend:

```bash
npm install
npm run dev
```

The development server opens on `http://localhost:5173` and proxies `/api` to
FastAPI. Set `VITE_API_BASE_URL` only when the API uses a different origin.

## Quality checks

```bash
npm run typecheck
npm run build
```

Conversation data is owned by the backend. This client keeps only the opaque
conversation ID in `sessionStorage`, renders model content without raw HTML,
uses bounded file/message inputs, and displays deterministic constraint evidence
separately from ML predictions.
