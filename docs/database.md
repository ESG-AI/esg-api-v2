# Database Schema Documentation

> **Engine:** PostgreSQL  
> **ORM:** SQLAlchemy 2.x  
> **JSONB columns:** PostgreSQL-native — cannot be swapped to SQLite/MySQL without a migration  
> **Source of truth:** [`db/models.py`](../db/models.py)

---

## Visual Diagram

![Database ERD](./database_erd.png)

---

## Entity-Relationship Diagram (Mermaid)

```mermaid
erDiagram
    users {
        int     id              PK
        string  email           UK "NOT NULL"
        string  hashed_password         "NOT NULL"
        enum    role                    "free | paid | admin, default=free"
        bool    is_active               "default=true"
        datetime created_at             "default=now()"
    }

    refresh_tokens {
        int     id          PK
        string  token       UK  "NOT NULL — opaque random hex"
        int     user_id     FK  "→ users.id"
        datetime expires_at     "NOT NULL"
        bool    revoked         "default=false"
    }

    documents {
        int     id                  PK
        string  filename
        datetime upload_date            "default=now()"
        string  s3_object_key
        int     file_size
        string  user_id                 "String, NOT a FK — legacy Clerk ID; migrate to int FK later"
        jsonb   extraction_quality
        jsonb   token_usage
        jsonb   performance_metrics
    }

    analysis_results {
        int     id                      PK
        int     document_id             FK  "→ documents.id"
        string  indicator_code              "e.g. GRI-302-1"
        string  indicator_title
        string  indicator_type              "governance | economic | social | environmental"
        string  indicator_subtype
        text    indicator_description
        int     score                       "0–4"
        text    reasoning
        jsonb   token_usage
    }

    score_summaries {
        int     id              PK
        int     document_id     FK  UK  "→ documents.id (1-to-1)"
        float   spdi_index_score         "Sum of all indicator scores"
    }

    users         ||--o{ refresh_tokens  : "has many"
    documents     ||--o{ analysis_results : "has many"
    documents     ||--||  score_summaries  : "has one"
```

---

## Tables

### `users`

Stores registered user accounts.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `INTEGER` | PK, auto-increment | Internal user ID |
| `email` | `VARCHAR` | UNIQUE, NOT NULL, indexed | Login email |
| `hashed_password` | `VARCHAR` | NOT NULL | bcrypt hash — never store plain text |
| `role` | `ENUM` | NOT NULL, default `free` | One of `free`, `paid`, `admin` |
| `is_active` | `BOOLEAN` | NOT NULL, default `true` | Soft-delete flag — set to `false` to ban |
| `created_at` | `TIMESTAMP` | default `now()` | Account creation time (UTC) |

**Indexes:** `email` (unique), `id` (PK)

---

### `refresh_tokens`

Long-lived opaque tokens used to obtain new access tokens without re-login.  
Stored in DB so they can be **individually revoked** (logout) or **bulk revoked** (ban / role change).

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `INTEGER` | PK | — |
| `token` | `VARCHAR` | UNIQUE, NOT NULL, indexed | 128-char random hex string |
| `user_id` | `INTEGER` | FK → `users.id`, NOT NULL | Token owner |
| `expires_at` | `TIMESTAMP` | NOT NULL | Hard expiry (default: 30 days from issue) |
| `revoked` | `BOOLEAN` | NOT NULL, default `false` | Set to `true` on logout or role change |

**Relationship:** `users` 1 → many `refresh_tokens`  
**Cascade:** deleting a user deletes all their refresh tokens

> [!NOTE]
> Tokens are **rotated** on every `/auth/refresh` call — the old token is revoked and a new one issued. This limits the window of abuse if a token is stolen.

---

### `documents`

Represents a PDF that has been uploaded and analysed.  
One document can be analysed multiple times (e.g. different GRI types) — `analysis_results` are upserted per indicator code.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `INTEGER` | PK | — |
| `filename` | `VARCHAR` | indexed | Original filename as provided by the client |
| `upload_date` | `TIMESTAMP` | default `now()` | Time of first analysis (UTC) |
| `s3_object_key` | `VARCHAR` | — | S3 object key — used to retrieve the PDF |
| `file_size` | `INTEGER` | — | Bytes |
| `user_id` | `VARCHAR` | nullable, indexed | ⚠️ Legacy string field (was a Clerk user ID). Kept for historical data — see Known Issues |
| `owner_id` | `INTEGER` | nullable, FK → `users.id`, indexed | ✅ Proper FK added in migration `003_documents_owner_fk`. Set for all documents created after auth migration. |
| `extraction_quality` | `JSONB` | — | PDF extraction diagnostics (pages, avg chars/page, ESG term coverage, etc.) |
| `token_usage` | `JSONB` | — | Aggregate OpenAI token usage for this document |
| `performance_metrics` | `JSONB` | — | Timing breakdown (extraction, AI eval, DB save, etc.) |

**Relationships:**
- 1 → many `analysis_results` (cascade delete)
- 1 → 1 `score_summaries` (cascade delete)
- many → 1 `users` via `owner_id` (nullable FK)

**Indexes:** `filename`, `user_id`, `owner_id`, `id` (PK)

#### `extraction_quality` JSONB shape
```json
{
  "total_pages": 42,
  "characters_extracted": 98241,
  "words_extracted": 16500,
  "avg_chars_per_page": 2339.07,
  "esg_terms_found": ["sustainability", "emissions", "governance"],
  "esg_term_coverage": "27.3%",
  "extraction_issues": [],
  "extraction_success": true
}
```

#### `performance_metrics` JSONB shape
```json
{
  "total_processing_time_seconds": 47.3,
  "extraction_time_seconds": 3.1,
  "ai_evaluation_time_seconds": 43.2,
  "db_save_time_seconds": 0.8,
  "indicator_processing_times": {
    "GRI-302-1": 0.42,
    "GRI-401-1": 0.38
  }
}
```

---

### `analysis_results`

One row per **indicator per document**.  
Scores can be manually corrected by an admin via `PATCH /documents/{id}/indicator/{code}`.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `INTEGER` | PK | — |
| `document_id` | `INTEGER` | FK → `documents.id`, indexed | Parent document |
| `indicator_code` | `VARCHAR` | indexed | GRI indicator code, e.g. `GRI-302-1` |
| `indicator_title` | `VARCHAR` | — | Human-readable title |
| `indicator_type` | `VARCHAR` | indexed | `governance`, `economic`, `social`, or `environmental` |
| `indicator_subtype` | `VARCHAR` | indexed | Sub-category within the type |
| `indicator_description` | `TEXT` | — | Full indicator description |
| `score` | `INTEGER` | — | AI-assigned score: `0`–`4` |
| `reasoning` | `TEXT` | — | AI explanation for the score |
| `token_usage` | `JSONB` | — | Per-indicator OpenAI token usage |

**Relationship:** `documents` 1 → many `analysis_results`

#### `token_usage` JSONB shape
```json
{
  "total_tokens": 1823,
  "prompt_tokens": 1700,
  "response_tokens": 123
}
```

---

### `score_summaries`

A **1-to-1** aggregate table — one row per document.  
The SPDI index is automatically recalculated whenever any `analysis_results.score` changes.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `INTEGER` | PK | — |
| `document_id` | `INTEGER` | FK → `documents.id`, UNIQUE | 1-to-1 with `documents` |
| `spdi_index_score` | `FLOAT` | — | Sum of all indicator scores for this document |

**Relationship:** `documents` 1 → 1 `score_summaries`

> [!TIP]
> The SPDI index is a simple sum across all evaluated indicators. Maximum possible score = `number_of_indicators × 4`.

---

## Relationship Summary

```
users
 └── refresh_tokens   (1 → many, cascade delete)

documents
 ├── analysis_results (1 → many, cascade delete)
 └── score_summaries  (1 → 1,    cascade delete)
```

---

> [!WARNING]
> **`documents.user_id` is a legacy `VARCHAR` from the Clerk era — not a real FK.**
> The new `owner_id INTEGER FK → users.id` was added in migration `003_documents_owner_fk`.
> Old documents with only `user_id` populated should be back-filled:
> ```sql
> UPDATE documents
> SET owner_id = CAST(user_id AS INTEGER)
> WHERE user_id ~ '^[0-9]+$' AND owner_id IS NULL;
> ```
> Once back-filled and verified, `user_id` can be dropped in a future migration.

> [!NOTE]
> **Alembic is configured** under `alembic/` with three versioned migrations:
>
> | Revision | Description |
> |---|---|
> | `001_initial` | Baseline schema — `documents`, `analysis_results`, `score_summaries` |
> | `002_auth` | Auth system — `users`, `refresh_tokens` |
> | `003_documents_owner_fk` | Adds `owner_id` FK column to `documents` |
>
> **Existing Neon DB:** stamp the baseline first, then upgrade:
> ```bash
> alembic stamp 001_initial   # mark baseline as already applied
> alembic upgrade head         # apply 002 + 003
> ```
> **Fresh DB:** run all migrations from scratch:
> ```bash
> alembic upgrade head
> ```

---

## Migration History

All migrations live in [`alembic/versions/`](../alembic/versions/).

---

## Repository Layer

| Repository | File | Responsibilities |
|---|---|---|
| `UserRepository` | [`db/repositories/user.py`](../db/repositories/user.py) | User CRUD, role updates, deactivation, refresh token lifecycle |
| `DocumentRepository` | [`db/repositories/document.py`](../db/repositories/document.py) | Document upsert (create/update), listing, retrieval by ID |
| `AnalysisRepository` | [`db/repositories/analysis.py`](../db/repositories/analysis.py) | Per-indicator score/reasoning updates + SPDI recalculation |
