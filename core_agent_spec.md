# PAF Core Agent – Client API Specification

_Last updated: 2025-07-10_

---

## 1. Introduction
The **PAF Core Agent** exposes a set of HTTP endpoints (most of them JSON, some SSE-based) that allow clients to:

* Stream chat conversations driven by the UPEE engine
* Attach files (both inline and large via signed URL) to those conversations
* Inspect or test file upload behaviour for debugging
* Invoke stand-alone file-processing helpers (chunking / summarisation)
* Query health / capability information

All endpoints are served from the same FastAPI application – the examples below assume the service is reachable at:
```
http://localhost:8000
```
Replace this base URL as appropriate for your deployment.

---

## 2. Common Conventions

| Item | Value |
|------|-------|
| Base path | `/api` |
| Authentication | _None required_ for the endpoints covered by this spec (a full Auth API exists under `/api/auth`, but it is **optional**). |
| Request/response format | `application/json; charset=utf-8` (except the chat **stream** endpoint which is `text/event-stream`). |
| Error handling | Standard HTTP status codes (+ JSON body with `detail` for FastAPI‐generated errors). |

### 2.1 File Upload Formats

The Core Agent supports **two different file upload methods** depending on the endpoint:

#### 2.1.1 Debug Endpoints - Multipart Form Data (10MB limit)
**Endpoints**: `/api/debug/inspect-request`, `/api/debug/test-file-processing`

* Uses `multipart/form-data` format
* Maximum **10 MB per file** – larger uploads rejected with HTTP 400
* Files sent as standard form parts, metadata as JSON in `payload` field

Example:
```bash
curl -X POST "http://localhost:8000/api/debug/inspect-request" \
  -F "payload={\"message\":\"Analyse this doc\",\"model\":\"gpt-4o\"};type=application/json" \
  -F "files=@report.xlsx;type=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
```

#### 2.1.2 Main Chat Endpoint - JSON with Base64 (100MB limit)  
**Endpoint**: `/api/chat/stream`

* Uses standard JSON request body
* Files embedded as **base64-encoded content** in `FileContent` objects
* Maximum **100 MB per file** (before base64 encoding)
* Supports signed URLs for large files (though not currently implemented)

Example:
```bash
curl -X POST "http://localhost:8000/api/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze this Excel file",
    "files": [{
      "file_name": "data.xlsx",
      "content": "UEsDBBQAAAAIABZ1lVKOd...",
      "file_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "file_size": 8742
    }]
  }'
```

### 2.2 Excel File Processing Capabilities

The Core Agent includes a sophisticated **File Processing Agent** that can handle Excel files (.xlsx, .xls, .xlsm, .xlsb):

**Supported Operations:**
* Parse multiple worksheets within a single Excel file
* Extract data structure (rows, columns, data types)
* Generate summary statistics for numeric columns  
* Display first 5 rows of each sheet for preview
* Handle base64-encoded Excel content automatically

**Requirements:**
* Files must be ≤ size limits (10MB for debug, 100MB for chat)
* Python packages: `pandas` and `openpyxl` (typically pre-installed)
* Content should be properly base64-encoded for main chat endpoint

**Example Output:**
```
EXCEL FILE ANALYSIS for report.xlsx:
Total sheets: 2

--- Sheet: Campaign_Data ---
Shape: 150 rows × 8 columns
Columns: ['Campaign', 'Date', 'Impressions', 'Clicks', 'Cost', 'CTR', 'CPC', 'Conversions']

First 5 rows:
Campaign      Date  Impressions  Clicks  Cost
Summer_Sale   2024-01-01    1500      75  125.50
Winter_Promo  2024-01-02    2200     110  180.75
...

Summary statistics for numeric columns:
         Impressions     Clicks       Cost
count      150.00     150.00     150.00
mean      1850.33      92.15     155.42
std        425.12      21.33      45.67
...
```

### 2.3 Request Parameters

**Token Limits:**
- `max_tokens`: Optional integer (default: dynamically calculated, capped at 100,000)
- When not specified, the system estimates output tokens based on request complexity and applies a 2x multiplier, capped at 100,000 tokens
- Explicit values override the automatic calculation

### 2.4 Schema References (abridged)
```jsonc
// FileContent (preferred)
{
  "file_name": "report.xlsx",
  "file_path": "/optional/full/path",           // optional
  "content": "<BASE64>",                        // one of content OR signed_url is required
  "signed_url": "https://...",                 // optional
  "file_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "file_size": 8742,
  "line_start": 1,                               // optional – partial extracts
  "line_end": 120,                               // optional
  "metadata": { "sheet": "Campaign_Data" }    // optional
}

// FileContext (legacy; still accepted)
{
  "path": "/src/app/main.py",
  "content": "<BASE64 OR PLAIN TEXT>",
  "line_start": 1,
  "line_end": 200
}

// ChatRequest
{
  "message": "Explain the insights from this report",
  "show_thinking": false,
  "files": [ <FileContent|FileContext> ],
  "history": [
    { "role": "user", "content": "previous question" },
    { "role": "assistant", "content": "previous answer" }
  ],
  "model": "gpt-4o",
  "temperature": 0.7,
  "max_tokens": 100000,
  "memory_limit": 10,
  "context_window_size": 8192,
  "metadata": { "ticket_id": "ABC-123" }
}
```

---

## 3. Endpoints

### 3.1 Chat – `POST /api/chat/stream`
**Description**: Streams the assistant’s answer using **Server-Sent Events (SSE)** while the UPEE engine iterates through _Understand → Plan → Execute → Evaluate_.

*Request headers*
```
Content-Type: application/json
Accept: text/event-stream
```

*Request body*: `ChatRequest` (see schema above).

*Response*
Content-Type: `text/event-stream`. You will receive a sequence of SSE events terminated by a `[DONE]` marker.

| Event type | Payload (`data`) | Notes |
|------------|-----------------|-------|
| `thinking` | JSON encoded `{ "phase": "understand", "content": "…" }` | Only emitted when `show_thinking = true`. |
| `content`  | Plain text chunk                                 | The assistant’s answer, chunked. |
| `complete` | JSON `{ "total_tokens": 123, "duration": 4.2, "model": "gpt-4o" }` | End-of-answer summary. |
| `error`    | JSON `{ "error": "msg", "error_type": "..." }`       | Terminal error – stream ends afterwards. |
| `done`     | Literal `[DONE]`                                  | Always emitted last. |

Sample curl:
```bash
curl -N -X POST "http://localhost:8000/api/chat/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d @chat_request.json
```
(The `-N` flag disables curl’s buffering so you see tokens in real time.)

---

### 3.2 Chat Meta
| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/api/chat/models`     | List all available LLM models plus their providers and health. |
| `GET`  | `/api/chat/providers`  | Detailed provider breakdown. |
| `GET`  | `/api/chat/status`     | High-level service status and feature flags. |

All three return `200 OK` with a JSON body. No parameters are required.

---

### 3.3 Debug Helpers

#### 3.3.1 `POST /api/debug/inspect-request`
Inspects **exactly** what the core agent would receive – very useful to verify file encoding, sizes, headers, etc.

*Request body*: identical to `ChatRequest`.

*Response* (`200 OK`, `application/json`):
```jsonc
{
  "timestamp": 1720543201.123,
  "request_inspection": {
    "message": { "length": 42, "preview": "Explain th…" },
    "files": {
      "count": 1,
      "file_details": [
        {
          "index": 0,
          "schema_type": "FileContent",
          "file_name": "report.xlsx",
          "file_size": 8742,
          "validation": { "is_valid": true, "errors": [] }
        }
      ]
    },
    "history": { "has_history": false, "message_count": 0 },
    "options": { "model": "gpt-4o", "temperature": 0.7 }
  },
  "raw_request_info": { "content_type": "application/json", … },
  "file_processing_capabilities": { "pandas_available": true, … }
}
```

#### 3.3.2 `POST /api/debug/test-file-processing`
Runs both the low-level `FileProcessor` and (if needed) the **agentic** file-processing pipeline on the supplied files – without invoking the whole UPEE chat loop.

Returns an array of per-file results:
```jsonc
{
  "timestamp": 1720543201.456,
  "files_received": 1,
  "processing_results": [
    {
      "file_index": 0,
      "file_name": "report.xlsx",
      "basic_processing": {
        "success": true,
        "requires_agentic": false,
        "content_length": 12000,
        "processing_status": "ok"
      }
    }
  ]
}
```

---

### 3.4 Stand-alone File Processing (optional)
> _Note: These routes are declared in the codebase (`app/api/files.py`) but **may require enabling** in `app/main.py` before they are reachable._

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/files/process`   | Bulk smart-chunking (with optional summary) – request body `{ "files": [FileContext], "include_summaries": true }` → structured JSON response with chunks & stats. |
| `POST` | `/api/files/summarize` | Summarise a single file – body `{ "file": FileContext, "summary_type": "abstractive" }`. |
| `POST` | `/api/files/analyze`   | Deeper structural analysis of one file – body `FileContext`. |
| `GET`  | `/api/files/types`     | Enumerates supported file types / extensions. |

---

### 3.5 Health & Diagnostics
* `GET /api/health` – Aggregated health of LLM providers, worker agents, etc. Returns a JSON payload matching the `HealthStatus` schema.
* `GET /api/db/test` – Simple PostgreSQL connectivity check (returns DB version).

---

## 4. Error Handling
* **4xx** – Client input invalid (see `detail` field for reason).
* **5xx** – Internal error; response _may_ include extra diagnostic information for `debug` endpoints.
* Streaming (`/chat/stream`) errors are emitted as a final `error` SSE event followed by termination.

---

## 5. Versioning & Stability Guarantees
This specification targets **PAF Core Agent v1.0.0**. While effort is made to preserve backwards compatibility, endpoints marked _optional_ or _experimental_ may change in future minor versions. Subscribe to release notes or pin your agent image/tag to avoid surprises.

---

## 6. Changelog
| Date | Notes |
|------|-------|
| 2025-07-10 | Initial public spec extracted from source code. | 