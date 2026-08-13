# Agentic API Standards: Responses API, Messages API, MCP, and Tool Calling Under the Hood

**Audience:** Platform engineers running vLLM on B200/B300 in a disconnected enclave, migrating agent workloads per the [deployment SOP](../SOP-production-ai-model-deployment.md).
**Status:** Researched August 2026 against primary sources. Fast-moving facts are dated inline (vLLM v0.26.0 shipped July 25, 2026; the MCP 2026-07-28 spec is one month old). Re-verify anything dated here before building on it a quarter from now.

---

## Key takeaways

- The industry has settled on **two agent-native wire protocols**: OpenAI's **Responses API** (`/v1/responses`) and Anthropic's **Messages API** (`/v1/messages`). Everything else — chat completions, the Assistants API — is either legacy-for-agents or dead. The Assistants API sunsets **August 26, 2026** (this month).
- Chat completions is **stateless** (client re-sends everything, every turn); Responses is **stateful** (server stores the transcript, client continues it with `previous_response_id`). Statefulness is not a convenience feature — it is how reasoning models keep their reasoning between turns, and OpenAI measures real quality and cache-hit gains from it.
- The Responses API replaces the `choices[].message` blob with **typed output items** (`message`, `reasoning`, `function_call`, `web_search_call`, ...) and replaces raw streaming chunks with **named semantic events** (`response.created`, `response.output_text.delta`, `response.completed`, ...). Your parsing code branches on `type`, not on position.
- The migration from chat completions is mostly mechanical renames (`messages` → `input`/`instructions`, `max_tokens` → `max_output_tokens`, `response_format` → `text.format`, `finish_reason` → `status` + `incomplete_details`) plus one real structural change: the tool loop returns `function_call` items and consumes `function_call_output` items keyed by `call_id`.
- Anthropic's Messages API is a **content-block** protocol: a message is a list of typed blocks (`text`, `thinking`, `tool_use`, `tool_result`), and the tool loop is "echo the assistant's blocks back, answer every `tool_use` with a `tool_result`." vLLM implements it natively, which is why Claude Code can point at your enclave with one environment variable.
- vLLM's built-in `/v1/responses` keeps response state **in engine memory** — it does not survive restarts or span replicas. The **`vllm-project/agentic-api`** Rust gateway adds durable state (SQLite), WebSocket streaming, `/v1/messages`, and an explicit tool-ownership model; the SOP makes it the production front door.
- **MCP (Model Context Protocol)** is the standard for wiring tools/data into agents. Current spec is **2026-07-28** (stateless core, header-based routing); the widely deployed prior revision is 2025-11-25. In an air gap you run **local stdio servers only**, mirror the code, and skip the public registry entirely.
- Tool calling in vLLM is a three-stage pipeline: **chat template renders tool schemas into the prompt → the model emits its family-specific call syntax → `--tool-call-parser` converts that text into OpenAI JSON**. A missing or wrong parser is the number-one way a new model silently breaks agents.
- Structured output (xgrammar-backed guided decoding) turns "usually valid JSON" into "always valid JSON" by masking illegal tokens at decode time. Use it for every machine-parsed output; know its costs (first-request schema compile, no recursive schemas).
- Migration plan in one line: put the agentic-api gateway in front of everything in passthrough mode, turn on `/v1/responses` beside `/v1/chat/completions`, migrate agents newest-first, and build the **"response not found → replay full history"** fallback into your client wrapper on day one.

---

## 1. Why the API shape matters for agents

An agent is a loop: the model thinks, asks for a tool, your code runs the tool, the model reads the result, and around it goes. The API is the contract for that loop, and the contract decides three expensive things:

1. **Who stores the conversation.** If the server remembers nothing (chat completions), the client re-sends the entire transcript every turn — hundreds of kilobytes by turn 30 — and the server re-prefills it (prefix caching softens this but never removes it, and only if routing cooperates).
2. **Whether the model's reasoning survives between turns.** Reasoning models produce internal thinking that improves later turns. A stateless API forces the client to either drop it (quality loss) or ship it back and forth (cost and complexity). A stateful API keeps it server-side.
3. **Who owns the tool loop.** With chat completions, every team hand-writes the loop differently. Agent-native APIs make tool calls and tool results first-class typed items, and can even execute some tools server-side.

Keep those three questions in mind; every section below is one protocol's answer to them.

---

## 2. Evolution: chat completions → Assistants → Responses

**Chat completions (`/v1/chat/completions`, 2023).** The API that made "OpenAI-compatible" a product category. Stateless, `messages[]` in, one `choices[0].message` out. Every serving engine, gateway, and framework speaks it, which is precisely why it will not disappear — but it predates agents, reasoning models, and server-side tools, and it shows.

**Assistants API (beta, late 2023 — sunset August 26, 2026).** OpenAI's first attempt at server-side agent state: persistent `Assistants`, `Threads`, and `Runs` objects with built-in tools. It never left beta. OpenAI announced its deprecation on **August 26, 2025** and is removing it on **August 26, 2026**, with a published migration path to the Responses API and the newer Conversations API. If you ever see Assistants-style code (`client.beta.assistants.*`, `threads`, `runs`), treat it as dead code to be ported. (Verified against OpenAI's deprecations page and the official announcement thread.)

**Responses API (`/v1/responses`, March 2025 — current).** The consolidation. It keeps chat completions' request/response ergonomics but adds server-side conversation state, typed input/output items, first-class reasoning items, semantic streaming events, background execution, and built-in server-side tools (web search, file search, code interpreter, computer use, and remote MCP servers — on OpenAI's cloud; in your enclave those built-ins are exactly the things you disable or replace).

OpenAI's stated positioning, quoted from the migration guide: **"While Chat Completions remains supported, Responses is recommended for all new projects."** Chat completions is maintained indefinitely as the ecosystem-compatibility API; agents belong on Responses. OpenAI also publishes two concrete numbers for why: about a **3% SWE-bench improvement** on reasoning models when reasoning items persist between turns, and **40–80% better cache utilization** versus chat completions.

This matches the SOP's stance (§6): keep `/v1/chat/completions` as a passthrough for legacy and batch tooling; move the agent path to `/v1/responses`.

---

## 3. The Responses API, dissected

### 3.1 Request anatomy

A minimal request looks almost like chat completions:

```json
{
  "model": "agent-default",
  "input": "Summarize the failed jobs from last night."
}
```

`input` accepts either a plain string (sugar for one user message) or a **list of typed items**. Items are the universal currency of this API — the same shapes appear in requests and responses:

```json
{
  "model": "agent-default",
  "instructions": "You are the cluster operations agent. Use tools; never guess.",
  "input": [
    { "role": "user",
      "content": [ { "type": "input_text", "text": "Check disk usage on gpu-07." } ] },
    { "type": "function_call_output",
      "call_id": "call_9xk2",
      "output": "{\"pct_used\": 91}" }
  ],
  "tools": [
    { "type": "function",
      "name": "run_ssh_command",
      "description": "Run a read-only command on a named cluster node.",
      "parameters": {
        "type": "object",
        "properties": {
          "node":    { "type": "string" },
          "command": { "type": "string" }
        },
        "required": ["node", "command"],
        "additionalProperties": false
      },
      "strict": true } ],
  "tool_choice": "auto",
  "store": true,
  "previous_response_id": "resp_a1b2c3",
  "max_output_tokens": 2048,
  "metadata": { "team": "sre", "trace_id": "t-4481" }
}
```

The important request fields, verified against the API reference (developers.openai.com, mid-2026):

| Field | What it does |
|---|---|
| `model` | Model or alias name (your gateway resolves aliases, per the SOP). |
| `input` | String, or list of items: messages, `function_call_output`, images, files, or prior output items you are replaying. |
| `instructions` | System/developer guidance. **Not inherited** via `previous_response_id` — re-send it every call. |
| `tools` | Tool definitions. Types include `function` (yours), plus OpenAI-hosted `web_search`, `file_search`, `code_interpreter`, `computer`/`computer_use_preview`, `mcp`, `image_generation`, `shell`/`local_shell`, `apply_patch`. Only `function` (and gateway-implemented ones) make sense in an enclave. |
| `tool_choice` | `auto`, `none`, `required`, or a specific tool by name. |
| `store` | Persist this response server-side (default `true` on OpenAI; stored responses are retained ~30 days there). `store: false` = fully stateless call. |

| Field | What it does |
|---|---|
| `previous_response_id` | Continue from a stored response — the server prepends that chain's items to your `input`. |
| `conversation` | Attach the call to a **Conversation object** (`/v1/conversations`, the Assistants-threads replacement): items flow in from the conversation and results are appended to it automatically. Use `conversation` *or* `previous_response_id`, not both. |
| `background` | `true` runs the response asynchronously; poll `GET /v1/responses/{id}` or resume its stream. For long agent turns. |
| `include` | Opt-in extras in the output, e.g. web-search sources, code-interpreter outputs, input image URLs, logprobs, and `reasoning.encrypted_content` (see §3.4). |
| `max_output_tokens` | Output cap (the `max_tokens` replacement). |
| `parallel_tool_calls` / `max_tool_calls` | Allow multiple calls per turn / cap total tool invocations. |
| `reasoning` | `{ "effort": "low"\|"medium"\|"high", "summary": ... }` for reasoning models. |
| `text` | `{ "format": {...} }` for structured output (the `response_format` replacement) and `verbosity`. |
| `stream` | Semantic-event streaming (§3.3). |
| `metadata` | Up to 16 key-value pairs, echoed back — your audit hook. |
| `truncation`, `temperature`, `top_p`, `service_tier`, `prompt_cache_key`, `safety_identifier` | Carry over with familiar meanings; the last three are OpenAI-cloud-oriented knobs. |

### 3.2 Response anatomy: typed output items

Where chat completions gives you `choices[0].message` (one blob with optional `tool_calls` bolted on), Responses gives you `output`: an **ordered list of typed items** that reads like a transcript of what the model did:

```json
{
  "id": "resp_d4e5f6",
  "object": "response",
  "status": "completed",
  "model": "agent-default",
  "output": [
    { "type": "reasoning",
      "id": "rs_001",
      "summary": [ { "type": "summary_text", "text": "Need current disk usage; call the ssh tool." } ] },
    { "type": "function_call",
      "id": "fc_001",
      "call_id": "call_9xk2",
      "name": "run_ssh_command",
      "arguments": "{\"node\": \"gpu-07\", \"command\": \"df -h /\"}" },
    { "type": "message",
      "id": "msg_001",
      "role": "assistant",
      "content": [ { "type": "output_text", "text": "Checking gpu-07 now.", "annotations": [] } ] }
  ],
  "incomplete_details": null,
  "usage": { "input_tokens": 812, "output_tokens": 63, "total_tokens": 875 }
}
```

Item types you will actually see (server-side tool items only appear if that tool runs server-side):

| Output item type | Meaning |
|---|---|
| `message` | Assistant text; `content` holds `output_text` blocks (and `refusal` blocks). SDKs add an `output_text` convenience property that concatenates them. |
| `reasoning` | A reasoning model's thinking — as a summary and/or an opaque encrypted payload. Persisted server-side when `store: true`; this is the item that makes multi-turn reasoning work. |
| `function_call` | The model wants *your* tool: `name`, `arguments` (JSON string), and a `call_id`. |
| `function_call_output` | Your answer to a `function_call` — appears in `input` when you send it, and in stored history. Keyed by the same `call_id`. |
| `web_search_call`, `file_search_call`, `code_interpreter_call`, `computer_call` (+`computer_call_output`), `image_generation_call`, `mcp_call` | Records of server-side/hosted tool executions. |
| `compaction` | Record of server-side context compression (newer addition; appears when context management triggers). |

Two status fields replace `finish_reason`:

- `status`: `in_progress` (background mode), `completed`, `failed`, `incomplete`, `cancelled`.
- `incomplete_details`: why an incomplete response stopped, e.g. `{"reason": "max_output_tokens"}` — the analog of `finish_reason: "length"`.

> **Common pitfall:** code ported from chat completions that reads "the message" as `output[0]`. With a reasoning model, `output[0]` is usually a `reasoning` item. Always filter by `type`; never index by position.

### 3.3 Streaming: semantic events, not chunk diffs

Chat completions streaming sends anonymous chunks, each carrying a `choices[0].delta` you must accumulate — and tool-call streaming is notoriously fiddly (arguments arrive as string fragments spread across chunks, identified only by index).

Responses streaming sends **named, typed SSE (Server-Sent Events)**, each with a `type` and a `sequence_number`. You subscribe to the events you care about and ignore the rest. Core lifecycle events, confirmed against the streaming reference (exact set as of mid-2026; the full catalog is larger):

| Event | Fires when |
|---|---|
| `response.created` | The response object exists; carries the full initial object (and the `id` you may need for fallback later). |
| `response.in_progress` | Generation is under way. |
| `response.output_item.added` / `response.output_item.done` | A new output item (message, function_call, reasoning, ...) starts / finishes. |
| `response.content_part.added` / `response.content_part.done` | A content part within an item starts / finishes. |
| `response.output_text.delta` / `response.output_text.done` | Incremental assistant text / final text for that part. |
| `response.function_call_arguments.delta` / `.done` | Incremental JSON arguments for a tool call — scoped to one item, no index juggling. |
| `response.reasoning_summary_text.delta` | Incremental reasoning summary (when exposed). |
| `response.completed` / `response.failed` / `response.incomplete` | Terminal states, carrying the final response object. |
| `error` | Stream-level error. |

```python
stream = client.responses.create(model="agent-default", input="...", stream=True)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "response.completed":
        final = event.response
```

The practical wins: your UI can show "calling `run_ssh_command`..." the moment `response.output_item.added` arrives with a `function_call`, and background responses can re-attach to a stream mid-flight (events are sequence-numbered for resume).

### 3.4 State semantics: store, chains, forks

- **`store: true`** persists the response and everything it produced (including reasoning items). On OpenAI's cloud, stored responses are retained for about 30 days by default (as of mid-2026); on your own gateway, *you* define retention — the SOP suggests purging response state older than 30 days and backing up the state DB.
- **`previous_response_id`** continues a chain. The server hydrates the whole prior transcript; you send only what is new. Two caveats verified in OpenAI's docs: `instructions` do **not** carry over (resend them), and billing-wise **all prior input tokens in the chain are still counted as input tokens** each turn — statefulness saves bandwidth, client complexity, and (with caching) latency, not raw token accounting.
- **Forking is free.** Any stored response ID can be the parent of *multiple* continuations. Point two different requests at the same `previous_response_id` and you have branched the conversation — the basis for tree-search agents, A/B replays, and "retry that turn with a different tool result" debugging. Chat completions can only do this by keeping N copies of the transcript client-side.
- **Conversation objects** (`/v1/conversations`) are the third state option: a durable container that automatically accumulates items across many responses. Think "thread," minus the Assistants baggage.
- **Stateless mode with reasoning intact:** with `store: false`, request `include: ["reasoning.encrypted_content"]` and replay the returned encrypted reasoning items in the next `input`. This exists for zero-data-retention deployments and is worth knowing even on-prem: it is the pattern for keeping reasoning continuity *without* trusting the server's store.

---

## 4. Migrating from chat completions: the mapping

Field-by-field, verified against OpenAI's "Migrate to the Responses API" guide (mid-2026):

| Chat completions | Responses | Notes |
|---|---|---|
| `messages` | `input` | String or item list. |
| `messages[0]` with `role: "system"`/`"developer"` | `instructions` | Top-level; resend every call when chaining. |
| `max_tokens` / `max_completion_tokens` | `max_output_tokens` | Rename. |
| `response_format` | `text.format` | Structured outputs moved under `text`. |
| `tools[].function.{name,...}` | `tools[].{name,...}` | Function defs are flattened ("internally tagged") — no nested `function` wrapper. |

| Chat completions | Responses | Notes |
|---|---|---|
| `tools[].function.strict` (off by default) | `strict` (attempted by default) | Responses tries strict schema mode unless the schema can't support it, then falls back. |
| `n` (multiple completions) | — | Removed; issue separate requests (or fork a stored response). |
| `choices[0].message.content` | `output[]` items / `output_text` helper | Filter items by `type`. |
| `choices[0].message.tool_calls[]` | `function_call` output items | `call_id` replaces `tool_calls[i].id`. |
| `{"role": "tool", "tool_call_id": ...}` message | `function_call_output` input item | `{"type": "function_call_output", "call_id": ..., "output": "..."}`. |
| `finish_reason` (`stop`/`length`/`tool_calls`) | `status` + `incomplete_details` | `length` → `status: "incomplete"` with `reason: "max_output_tokens"`; "wants tools" → presence of `function_call` items. |
| streamed `chunk.choices[0].delta` | typed events (§3.3) | Branch on `event.type`. |
| — (client-held history) | `store` / `previous_response_id` / `conversation` | The new state machinery. |

What stays the same: the `model` parameter, roles, sampling knobs (`temperature`, `top_p`), authentication, and the general token-billing model.

### Before/after: a complete tool loop

**Before — chat completions.** The client owns the transcript and re-sends it every turn:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://gateway.enclave.local:9000/v1", api_key="internal")

SYSTEM = "You are the cluster ops agent. Use tools; never guess."
TOOLS = [{
    "type": "function",
    "function": {                     # <-- nested wrapper (chat-completions shape)
        "name": "run_ssh_command",
        "description": "Run a read-only command on a named cluster node.",
        "parameters": {
            "type": "object",
            "properties": {"node": {"type": "string"}, "command": {"type": "string"}},
            "required": ["node", "command"],
        },
    },
}]

messages = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Check disk usage on gpu-07."}]

while True:
    resp = client.chat.completions.create(
        model="agent-default", messages=messages, tools=TOOLS)
    msg = resp.choices[0].message
    messages.append(msg)                          # grow the transcript, forever
    if not msg.tool_calls:
        break
    for tc in msg.tool_calls:
        result = dispatch(tc.function.name, json.loads(tc.function.arguments))
        messages.append({"role": "tool",
                         "tool_call_id": tc.id,
                         "content": json.dumps(result)})

print(msg.content)
```

**After — Responses.** The server owns the transcript; the client sends only tool outputs plus a pointer:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://gateway.enclave.local:9000/v1", api_key="internal")

SYSTEM = "You are the cluster ops agent. Use tools; never guess."
TOOLS = [{
    "type": "function",               # <-- flattened (Responses shape)
    "name": "run_ssh_command",
    "description": "Run a read-only command on a named cluster node.",
    "parameters": {
        "type": "object",
        "properties": {"node": {"type": "string"}, "command": {"type": "string"}},
        "required": ["node", "command"],
        "additionalProperties": False,
    },
    "strict": True,
}]

response = client.responses.create(
    model="agent-default", instructions=SYSTEM, tools=TOOLS, store=True,
    input="Check disk usage on gpu-07.")

while True:
    calls = [it for it in response.output if it.type == "function_call"]
    if not calls:
        break
    outputs = [{"type": "function_call_output",
                "call_id": c.call_id,
                "output": json.dumps(dispatch(c.name, json.loads(c.arguments)))}
               for c in calls]
    response = client.responses.create(
        model="agent-default", instructions=SYSTEM, tools=TOOLS, store=True,
        previous_response_id=response.id,          # continue server-side
        input=outputs)                             # send ONLY the new items

print(response.output_text)
```

Note what shrank: no growing `messages` array, no system-message re-send inside the payload body (only the short `instructions` string), and the reasoning items a thinking model produced in turn 1 are still in play at turn 5 without you touching them.

---

## 5. Anthropic's Messages API: the other standard

### 5.1 Why it exists in your stack

Anthropic's `/v1/messages` is not an OpenAI dialect — it is a separate protocol with its own SDKs and its own ecosystem of clients, of which **Claude Code** is the one you will meet first (an agentic coding CLI that speaks only this protocol). The SOP's position: `/v1/responses` is the primary agent API; `/v1/messages` is the second protocol worth serving, because Anthropic-native tooling is worth having in the enclave and translation shims are where tool calls go to die.

### 5.2 Request anatomy

```json
{
  "model": "agent-default",
  "max_tokens": 2048,
  "system": "You are the cluster ops agent. Use tools; never guess.",
  "messages": [
    { "role": "user", "content": "Check disk usage on gpu-07." }
  ],
  "tools": [
    { "name": "run_ssh_command",
      "description": "Run a read-only command on a named cluster node.",
      "input_schema": {
        "type": "object",
        "properties": {
          "node":    { "type": "string" },
          "command": { "type": "string" }
        },
        "required": ["node", "command"]
      } }
  ],
  "tool_choice": { "type": "auto" }
}
```

Differences from both OpenAI APIs that trip people up:

- `max_tokens` is **required**, not optional.
- The system prompt is a top-level `system` field, never a message role.
- Tool schemas use `input_schema` (OpenAI uses `parameters`), and there is no `type: "function"` wrapper at all.
- `tool_choice` is an object: `{"type": "auto"}`, `{"type": "any"}` (must use some tool), `{"type": "tool", "name": "..."}` (must use that tool), `{"type": "none"}`.
- Real Anthropic endpoints authenticate with `x-api-key` plus an `anthropic-version` header rather than a `Bearer` token — relevant when you configure gateways and header-forwarding.

### 5.3 Content blocks: the core idea

A message's `content` is either a string or a **list of typed content blocks**. The response is always blocks:

```json
{
  "id": "msg_01ABC",
  "role": "assistant",
  "stop_reason": "tool_use",
  "content": [
    { "type": "thinking",
      "thinking": "Need live data; call the ssh tool on gpu-07.",
      "signature": "EqQBCk..." },
    { "type": "text", "text": "Let me check that node." },
    { "type": "tool_use",
      "id": "toolu_01XYZ",
      "name": "run_ssh_command",
      "input": { "node": "gpu-07", "command": "df -h /" } }
  ]
}
```

- **`text`** — assistant prose.
- **`thinking`** — extended-thinking (reasoning) content, carrying a cryptographic `signature`. The replay rule: when you continue the conversation, pass thinking blocks back **byte-for-byte unchanged** — the API validates the signature and rejects tampered blocks. This is Anthropic's client-side answer to what Responses solves server-side with stored reasoning items. (vLLM's implementation separates thinking via `--reasoning-parser`; the same "don't mangle it on replay" discipline applies.)
- **`tool_use`** — a tool call: `id`, `name`, and `input` as a **parsed JSON object** (not a string — one less `json.loads` than OpenAI).
- **`tool_result`** — your answer, sent in the *next user message*, keyed by `tool_use_id`.

The tool loop shape: append the assistant's content blocks verbatim as an `assistant` message, then send one `user` message containing a `tool_result` block **for every** `tool_use` block (all results in a single user message when calls were parallel — splitting them degrades the model's parallel-calling behavior):

```json
{ "role": "user", "content": [
    { "type": "tool_result",
      "tool_use_id": "toolu_01XYZ",
      "content": "{\"pct_used\": 91}" } ] }
```

Loop control hangs off **`stop_reason`**: `end_turn` (done), `tool_use` (answer the tools and continue), `max_tokens`, `stop_sequence`, plus newer values like `pause_turn` (server-side tool loop paused; re-send to resume) and `refusal`. Failed tools are reported as a `tool_result` with `"is_error": true` — never silently dropped.

Streaming is SSE with its own event family: `message_start` → per-block `content_block_start` / `content_block_delta` (`text_delta`, `thinking_delta`, `input_json_delta` for tool arguments) / `content_block_stop` → `message_delta` (carries final `stop_reason` and usage) → `message_stop`. Note it is block-scoped like Responses events, not chunk-positional like chat completions.

Like chat completions — and unlike Responses — `/v1/messages` is **stateless**: the client re-sends full history each turn. Anthropic mitigates the cost with prompt caching (`cache_control` breakpoints) rather than server-side state.

### 5.4 Serving `/v1/messages` from vLLM

Two supported paths, both verified August 2026:

1. **vLLM native.** Current vLLM implements the Anthropic Messages API directly, and ships a documented Claude Code integration (docs.vllm.ai → Serving → Integrations → Claude Code). Recipe:

   ```bash
   vllm serve /models/org/model-NVFP4 \
     --served-model-name my-model \
     --enable-auto-tool-choice \
     --tool-call-parser <parser-for-this-model>

   # Claude Code side:
   export ANTHROPIC_BASE_URL="http://gpu-node:8000"   # client appends /v1/messages
   export ANTHROPIC_AUTH_TOKEN="dummy"                 # any value; vLLM doesn't check
   export ANTHROPIC_DEFAULT_OPUS_MODEL="my-model"
   export ANTHROPIC_DEFAULT_SONNET_MODEL="my-model"
   export ANTHROPIC_DEFAULT_HAIKU_MODEL="my-model"
   claude
   ```

   Documented caveats: avoid `/` in served model names (use `--served-model-name`), and on older vLLM (≤0.17.1) set `"CLAUDE_CODE_ATTRIBUTION_HEADER": "0"` in Claude Code settings — the client injects a per-request attribution hash that silently destroys prefix caching, exactly the failure mode the SOP's "byte-stable prompts" rule warns about.

2. **agentic-api gateway.** The gateway exposes `/v1/messages` in front of the per-model vLLM pools, so Anthropic-protocol clients ride the same alias routing, authn, and metrics as everything else. (The repo README, fetched August 2026, lists the Anthropic Messages endpoint as active while also marking some Messages/Interactions functionality as still landing — treat exact coverage as version-dependent and re-test your Claude Code flows on every gateway upgrade.)

Also of note: vLLM's v0.26 Rust frontend has an accepted RFC (vllm #47753) for `/v1/messages` plus `/v1/messages/count_tokens` in the Rust serving path — a signal that Anthropic-protocol support is becoming a first-class frontend concern, not a bolt-on.

---

## 6. Server-side support status in vLLM (as of v0.26.0, July 2026)

| Capability | `vllm serve` built-in | `agentic-api` gateway |
|---|---|---|
| `/v1/responses` (+ `GET /{id}`, `/{id}/cancel`) | Yes — text-generation models | Yes (POST + WebSocket transport) |
| State backing | **In-engine memory** (response + message stores) | **SQLite** response store on disk |
| State survives restart / spans replicas | No / No | Yes / yes-with-shared-DB (move to Postgres-class at fleet scale, per SOP) |
| `/v1/messages` (Anthropic) | Yes (native; Claude Code integration documented) | Yes |
| `/v1/conversations` | No (feature request open, vllm #24479) | Yes |

Additional gateway facts from the repo (August 2026): it is Rust (`agentic-server` crate); launches either fused — `vllm serve <MODEL> --agentic-api` — or standalone against a running engine — `agentic-api --llm-api-base http://127.0.0.1:8000`; listens on **port 9000** by default and waits up to 10 minutes for vLLM readiness; streams over SSE and WebSocket; and is validated against the *Open Responses* compatibility suite with replay-cassette tests against recorded OpenAI and vLLM traffic.

Its **tool-ownership model** is the design piece worth internalizing, because it answers "who runs this tool?" explicitly per tool instead of by accident:

- **Gateway-owned** — executed inside the gateway (e.g. its web-search/file-search integrations). *In the enclave, gateway-owned tools must be your internal services only; the stock web-search integration (You.com-backed, keyed by `YOU_API_KEY`) points at the internet and stays disabled.*
- **Client-owned** — returned to the client to execute, Codex-style (`shell`, editor tools). This is behavior-identical to a classic client-side loop and is where every migrated agent starts.
- **Provider-owned** — passed through to vLLM itself (e.g. things the engine handles natively). Unknown tool shapes are preserved and round-tripped rather than rejected.

Why the SOP says "built-in for single replica, gateway for production": vLLM's own maintainers flag (RFC #24603) that the in-memory stores make it impossible to route a follow-up `previous_response_id` request to a *different* replica — the state lives in one process's RAM. The moment you have two replicas behind a load balancer, engine-local state is a correctness bug, not a convenience. The gateway centralizes the state and makes the vLLM pool stateless again — which also means the gateway's DB is now **production state**: back it up, define retention, and include it in DR tests.

---

## 7. MCP: the Model Context Protocol, deep dive

### 7.1 What it is and what problem it solves

MCP is an open, JSON-RPC-based protocol (originated by Anthropic, November 2024; now governed openly with an official registry and multi-vendor backing) that standardizes how an AI application discovers and uses external capabilities. Before MCP, every framework had its own tool-plugin format; MCP is the "USB standard" — write a tool server once, and any MCP-capable client (Claude Code, IDEs, agent frameworks, the OpenAI Responses `mcp` tool type) can use it.

### 7.2 Architecture: hosts, clients, servers

```
+---------------------------- Host process ----------------------------+
|  (Claude Code, an IDE, your agent app)                               |
|                                                                      |
|   +-- MCP client 1 --- stdio ------> MCP server A (git tools)        |
|   +-- MCP client 2 --- stdio ------> MCP server B (internal DB)      |
|   +-- MCP client 3 -- streamable --> MCP server C (HTTP, remote)     |
|                          HTTP                                        |
+----------------------------------------------------------------------+
```

- **Host** — the AI application the user runs. It owns the model conversation and the security policy.
- **Client** — a connector object inside the host; one client per server connection.
- **Server** — a program exposing capabilities. It can be a 200-line script wrapping one internal API.

### 7.3 The primitives

Server-side primitives (what a server offers):

| Primitive | What it is | Analogy |
|---|---|---|
| **Tools** | Model-invoked functions with JSON-schema inputs (`tools/list`, `tools/call`) | POST endpoints for the model |
| **Resources** | Application-read data identified by URI (files, table schemas, dashboards) | GET endpoints for the host |
| **Prompts** | User-selected prompt templates (slash-command material) | Saved queries |

Client-side primitives (what a server may ask of the host): **sampling** (server requests an LLM completion through the host — the host's model, host's approval), **roots** (host tells the server which directories are in scope), and **elicitation** (server asks the user a structured question mid-operation). The 2025-11-25 revision added an experimental **tasks** primitive ("call now, fetch later" for long-running tool calls); 2026-07-28 moved tasks out of the experimental core into an official **extension**.

### 7.4 Transports

Two official transports:

- **stdio** — the host launches the server as a subprocess and speaks JSON-RPC over stdin/stdout. Local, no network listener, no auth story needed beyond "you chose to execute this binary." **This is the enclave default.**
- **Streamable HTTP** — a single HTTP endpoint accepting POSTed JSON-RPC, optionally upgrading responses to an SSE stream. This replaced the older dedicated HTTP+SSE transport (deprecated since the 2025-03-26 revision); any doc showing a separate `/sse` endpoint is describing the legacy shape.

### 7.5 Spec versions and the 2026 shake-up

MCP revisions are date-named. The timeline that matters:

| Revision | Highlights |
|---|---|
| 2025-03-26 | Streamable HTTP replaces HTTP+SSE; OAuth 2.1 authorization framework. |
| 2025-06-18 | Structured tool output; elicitation; resource-server auth clarifications (RFC 8707 resource indicators). |
| **2025-11-25** | The one-year release, and the broadly deployed baseline today: experimental **tasks**, OpenID Connect discovery, icons metadata, incremental scope consent, URL-mode elicitation, sampling-with-tools, OAuth **Client ID Metadata Documents (CIMD)**. |
| **2026-07-28** | **Current.** Stateless core: the `initialize` handshake and session IDs are gone; cross-call state travels as explicit server-minted handles in tool arguments. **MRTR** (multi round-trip requests: server returns `input_required`, client retries with answers) replaces server-initiated requests over open streams. **Header-based routing**: `Mcp-Protocol-Version`, `Mcp-Method`, `Mcp-Name` HTTP headers let gateways route/authorize without parsing bodies. Auth hardening (RFC 9207 issuer validation; CIMD over dynamic client registration). Formal extensions framework. |

The 2026-07-28 release is **final** (confirmed via the official MCP blog, July 2026), but it is weeks old as of this writing: expect most servers and SDKs you mirror to still target 2025-11-25 for a while, and check each SDK's negotiated protocol version. The stateless redesign exists precisely so remote MCP servers can sit behind ordinary load balancers — the same "state doesn't belong in the worker" lesson as §6.

**Auth story in one paragraph:** stdio servers inherit the host's identity — no protocol-level auth. Streamable HTTP servers use OAuth 2.1: the MCP server is a *resource server*, tokens are audience-bound to it (resource indicators), and since 2025-11-25 clients can identify themselves via CIMD (a hosted JSON document describing the client) instead of per-server dynamic registration. In an enclave with an internal PKI and no external IdP, the pragmatic pattern is: stdio wherever possible; for internal HTTP servers, mTLS via your internal CA plus network policy, with OAuth only if you already run an internal identity provider.

### 7.6 The MCP registry

The official registry (registry.modelcontextprotocol.io, launched **September 8, 2025**, still in **preview** as of August 2026 — no durability guarantees, breaking changes possible) is a metadata catalog of publicly available servers: names (`io.github.org/server`), versions, and *pointers* to where the code lives (npm, PyPI, container registries). It is a discovery index, not a package host.

### 7.7 MCP in an air-gapped enclave

The registry, remote servers, and OAuth flows all assume internet. Your posture:

1. **Local stdio servers only, mirrored like everything else.** MCP server code enters via the §4 bundle pipeline: fetch on the connected side, pin versions, hash, review, sign, transfer. Serve Python/Node dependencies from the internal devpi/npm mirror. No `npx some-server@latest` inside — that is a supply-chain hole and a guaranteed runtime failure.
2. **No remote registries.** Host configs must not reference registry URLs. If you need discovery at scale, run an internal sub-registry (the registry code is Apache-licensed and designed for sub-registries); at small scale, a git repo listing approved servers is fine.
3. **Security-review every server before admission.** An MCP server is arbitrary code with model-triggerable side effects. Review checklist: what binaries/APIs it touches; whether tool descriptions could carry prompt-injection payloads ("tool poisoning"); whether tool outputs are attacker-influenced (a file-reading tool reading attacker-authored files is an injection vector); network egress (must be none or internal-only); and least-privilege execution (own service account, read-only where possible). Pin the reviewed version by hash; re-review on upgrade.
4. **Confused-deputy caution.** The agent holds the union of every connected server's powers. A read-tool that can see secrets plus a write-tool that can reach anything internal is an exfiltration pair even with zero internet.

### 7.8 How MCP relates to the Responses API

They are complementary layers, and they meet in one place: the Responses API defines an **`mcp` tool type**. On OpenAI's cloud, `{"type": "mcp", "server_url": ..., ...}` makes *OpenAI's infrastructure* connect to a remote MCP server and run its tools server-side, emitting `mcp_call` output items. In your enclave the same concept maps onto the gateway's tool-ownership model: an MCP server your platform team runs can back a **gateway-owned** tool (the gateway executes MCP calls inside the enclave), while developer-workstation MCP servers (Claude Code + stdio) stay entirely **client-owned**. Same protocol, two insertion points; the difference is only *who* hosts the MCP client.

---

## 8. Tool calling under the hood in vLLM

### 8.1 The three-stage pipeline

An open-weights model does not emit OpenAI JSON natively. Every family was trained on its own tool-call syntax, so vLLM translates at both ends:

```
            request.tools (JSON schemas)
                     |
        [1] chat template renders schemas
            into the prompt, family-style
                     |
                 model decodes
                     |
        [2] model emits ITS OWN syntax, e.g.
            <tool_call>{"name": ..., "arguments": ...}</tool_call>   (Hermes/Qwen)
            [TOOL_CALLS][{"name": ..., "arguments": ...}]            (Mistral)
            {"name": ..., "parameters": ...}                        (Llama 3 JSON)
            [get_weather(city="SF"), get_time(tz="PST")]            (pythonic)
                     |
        [3] --tool-call-parser recognizes that syntax and
            converts it to OpenAI tool_calls JSON (and, for
            /v1/responses, into function_call items)
                     |
            client sees standard OpenAI shapes
```

Stage 1 is the **chat template** (Jinja, shipped in `tokenizer_config.json`): it must know how to render `tools`, `tool`-role results, and prior assistant tool calls. Stage 3 is the **parser**, enabled with:

```bash
vllm serve /models/org/model \
  --enable-auto-tool-choice \
  --tool-call-parser <name> \
  [--chat-template /path/override.jinja]   # only when the shipped template is broken
```

`--enable-auto-tool-choice` lets the model decide for itself when to call tools (`tool_choice: "auto"`, the agentic default). Without it, only forced modes work.

### 8.2 Parser catalog (vLLM v0.26-era, from the in-tree docs, August 2026)

| `--tool-call-parser` | Model families |
|---|---|
| `hermes` | Hermes 2 Pro/Theta, Hermes 3; Qwen2.5 (Hermes-style template built in) |
| `mistral` | Mistral Instruct family (often with `tool_chat_template_mistral_parallel.jinja`) |
| `llama3_json` | Llama 3.1 / 3.2 JSON tool calling |
| `llama4_pythonic` / `pythonic` | Llama 4 / small Llama 3.2, ToolACE, Ultravox |
| `granite`, `granite4`, `granite-20b-fc` | IBM Granite 3.x / 4.0 / 20B-FC |
| `internlm` | InternLM 2.5 chat |
| `jamba` | AI21 Jamba 1.5 |
| `xlam` | Salesforce xLAM (and Qwen-xLAM) |
| `deepseek_v3` / `deepseek_v31` | DeepSeek-V3 & R1 / DeepSeek-V3.1 |
| `openai` | gpt-oss-20b / gpt-oss-120b (Harmony format) |
| `kimi_k2` | Moonshot Kimi-K2 |
| `qwen3_xml` | Qwen3-Coder |
| `glm45` / `glm47` | GLM-4.5/4.6 / GLM-4.7 |
| `hunyuan_a13b`, `cohere_command3`, `longcat`, `olmo3`, `functiongemma`, `gigachat3`, `apertus` | Tencent Hunyuan-A13B, Cohere Command-A, Meituan LongCat, AI2 Olmo-3, Google FunctionGemma, GigaChat 3, Swiss AI Apertus |
| `seed_oss` | ByteDance Seed-OSS (SOP-referenced; landed in the v0.26 Rust frontend — verify the exact name against `vllm serve --help` on your pinned build) |

This list churns every release — new families add parsers, names occasionally change. The authoritative source is your **pinned** build's `docs/features/tool_calling.md` (mirror it into the enclave with the recipes repo) and `vllm serve --help`. If a brand-new model has no parser, you can ship one as a plugin: subclass `ToolParser`, implement `extract_tool_calls()` (complete output) and `extract_tool_calls_streaming()` (incremental), register it, and load with `--tool-parser-plugin /path/plugin.py --tool-call-parser my_parser`.

### 8.3 Tool-choice modes and how each is enforced

| `tool_choice` | Mechanism | Cost/caveat |
|---|---|---|
| `"auto"` | Parser extracts calls from free text | Malformed output possible **unless** a tool sets `strict: true` (then guided decoding constrains arguments) |
| `"required"` | Guaranteed ≥1 call; **structured-outputs backend enforces schema** | Always schema-valid |
| named function | Forced via guided decoding | **First use compiles an FSM — expect seconds of extra latency once per schema**, then cached |
| `"none"` | Tools still rendered into the prompt by default | Add `--exclude-tools-when-tool-choice-none` to reclaim those prompt tokens |

Streaming tool calls work in `auto` mode when the parser implements incremental extraction (all mainstream ones do): clients receive standard OpenAI `tool_calls` deltas / `response.function_call_arguments.delta` events while the model is still generating.

### 8.4 What breaks when the parser is wrong — symptoms table

> **Common pitfalls.** This is the SOP's "#1 cause of new-model agent breakage," so learn the signatures:
>
> - **No parser configured:** tool-call syntax arrives as ordinary assistant text — the agent framework sees "the model answered" (with `<tool_call>{"name"...` gibberish in the chat) and no tool ever runs. Nothing errors. This is the worst failure because dashboards stay green.
> - **Wrong parser for the family:** empty `tool_calls`, truncated/garbled `arguments`, or parser exceptions in vLLM logs (`Error in extracting tool call from response`). Frameworks then retry, so the metric to watch is **tool-call parse failure / retry rate** (SOP §7 golden signal).
> - **Parser right, template wrong:** turn 1 works, multi-turn fails — the template can't render `tool` results or prior calls back into the prompt, so the model re-calls the same tool forever or hallucinates results. Fix with a `--chat-template` override.
> - **Streaming-only breakage:** non-streamed calls parse fine but streamed arguments arrive mangled — an `extract_tool_calls_streaming()` bug for that parser/version. Pin known-good, re-run the 20-prompt tool gate on every engine upgrade (SOP runbook #1: "parsers break silently across versions").
> - **Prompt-injection twist:** since calls are parsed from generated text, a model that *quotes* tool-call syntax (e.g. while explaining it) can emit an accidental call. Defensive dispatch: validate every parsed call against the request's tool list and schema before executing.

---

## 9. Structured outputs: guaranteed-valid JSON

### 9.1 Mechanism

Guided decoding constrains generation at the logit level: the backend compiles your JSON schema (or regex/grammar/choice list) into an automaton, and at each decode step masks every token that cannot extend a valid string. Invalid JSON is not repaired after the fact — it is **unrepresentable**. In vLLM (v0.26) the backends are **xgrammar** (default-favored, compiled, very fast per-step) and **llguidance**, selected by `--structured-outputs-config.backend` with `auto` choosing per request.

### 9.2 The API surface

Two client shapes, one engine feature:

```python
# (a) OpenAI-standard response_format — works on chat completions;
#     the Responses-API equivalent is text.format
resp = client.chat.completions.create(
    model="agent-default",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "diag", "schema": DiagReport.model_json_schema()},
    },
)

# (b) vLLM's unified extension parameter (choice / regex / json / grammar /
#     structural_tag) — the old guided_json/guided_regex fields were
#     REMOVED in v0.12.0; update any pre-2026 snippets you find
resp = client.chat.completions.create(
    model="agent-default",
    messages=[...],
    extra_body={"structured_outputs": {"choice": ["healthy", "degraded", "failed"]}},
)
```

And the relationship to **tool schemas**: `response_format`/`text.format` constrains the *message body*; tool schemas constrain *arguments*. In vLLM the same guided-decoding machinery backs both — `tool_choice: "required"` and named-function forcing always run schema-constrained, and `strict: true` on a tool opts its arguments into constrained decoding under `auto`. The SOP's rule ("structured-output validity rate ≈ 100% must be a blocking gate") is achievable precisely because this is decode-time enforcement, not validation.

### 9.3 Costs and caveats

- **Schema compile time.** First request with a new schema pays grammar/FSM compilation — seconds for complex schemas. Warm production schemas at deploy time; compiled artifacts are cached after that.
- **Schema feature limits.** No recursive schemas; numeric bounds (`minimum`/`maximum`), string length constraints, and rich `additionalProperties` values are unsupported or ignored depending on backend. Set `additionalProperties: false` and keep schemas boring. Validate the semantic constraints client-side.
- **Constrained ≠ correct.** The output will parse; it can still be wrong. Guided decoding also slightly reshapes the model's distribution — for open-ended prose fields, over-tight schemas can degrade quality.
- **Reasoning interaction.** With reasoning parsing enabled, constraints apply to the final content, not the thinking. Known edge (v0.26 docs): Qwen3-Coder with reasoning may need `--structured-outputs-config.enable_in_reasoning=True` to combine both features.

---

## 10. Framework compatibility against self-hosted endpoints

**OpenAI Agents SDK (Python).** Defaults to the **Responses API** — which is exactly what your gateway serves, so first try the native path:

```python
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_default_openai_api, set_tracing_disabled

set_default_openai_client(AsyncOpenAI(
    base_url="http://gateway.enclave.local:9000/v1", api_key="internal"))
# Fallback if a feature mismatch bites (many non-OpenAI endpoints only do chat):
# set_default_openai_api("chat_completions")
set_tracing_disabled(True)   # MANDATORY in the enclave: tracing exports to OpenAI by default
```

That tracing line is the air-gap landmine: the SDK's trace exporter phones `api.openai.com` unless disabled (or redirected with `use_for_tracing=False` + your own processor). Treat it like `VLLM_NO_USAGE_STATS` — set it everywhere, verify with egress logs.

**LangChain / LangGraph.** LangGraph is model-agnostic; point `ChatOpenAI(base_url=..., model="agent-default")` at the gateway and graphs run unchanged. Details that matter (verified against langchain-openai reference, 2026): set **`use_responses_api=True` explicitly** if you want the Responses path — the SDK otherwise infers the API from the model name, which misfires on aliases; pass vLLM-specific parameters via `extra_body`; and know that provider-specific reasoning fields (`reasoning_content` on chat completions) are dropped by the generic class (open issue langchain #35059) — one more reason reasoning-heavy agents belong on `/v1/responses`, where reasoning is a typed item.

**Claude Code.** Speaks `/v1/messages` only. Point it at vLLM directly or at the gateway via `ANTHROPIC_BASE_URL` (full recipe in §5.4). Its heavy, repetitive system prompt makes it a prefix-caching stress test — watch the hit-rate metric after enabling.

**Codex-style clients.** OpenAI's Codex CLI speaks the Responses API and expects `shell`/editor tools to come back for local execution — the gateway's *client-owned* tool category exists for exactly this pattern.

---

## 11. Practical migration cookbook (gateway-first, per SOP §6)

**Step 0 — inventory.** From gateway logs (you did put everything behind one endpoint, right?), list clients by API and traffic: chat-completions agents, batch/eval tooling (guidellm etc. — stays on chat completions forever, that's fine), and Anthropic-protocol clients.

**Step 1 — gateway in passthrough.** Stand up agentic-api in front of the per-model vLLM pools; all `/v1/chat/completions` traffic flows through unchanged. You gain the choke point: uniform authn, per-client metrics, and the place migration progress becomes measurable. Put the SQLite state DB on persistent storage **now** and add it to backups — it becomes production state the moment step 2 lands.

**Step 2 — dual-stack.** Enable `/v1/responses` (and `/v1/messages` if you have Anthropic-native clients) beside the passthrough. The same aliases (`agent-default`, `agent-long-context`) must resolve identically on every endpoint — promotion stays an alias flip regardless of protocol.

**Step 3 — migrate agents newest-first, reasoning-first.** The mechanical change per client is §4's before/after. Reasoning-model agents move first: they gain the most (server-side reasoning persistence between turns). All tools start **client-owned** — behavior-identical to today, zero new trust decisions.

**Step 4 — the state-loss fallback, from day one.** A stateful API means a new failure mode: the gateway restores from an older backup, retention purges a chain, or a request lands somewhere the state isn't — and `previous_response_id` dangles. OpenAI's own semantics here are an error on the create call ("response not found" style, HTTP 404/400). Your agent SDK wrapper must degrade to stateless replay, which means keeping a client-side shadow transcript even while you normally don't send it:

```python
import openai

class ResilientSession:
    """Responses-API session that survives server-side state loss."""

    def __init__(self, client, model, instructions, tools):
        self.client, self.model = client, model
        self.instructions, self.tools = instructions, tools
        self.last_id = None
        self.shadow = []          # every item we sent or received, in order

    def step(self, new_items):
        kwargs = dict(model=self.model, instructions=self.instructions,
                      tools=self.tools, store=True)
        try:
            resp = self.client.responses.create(
                previous_response_id=self.last_id, input=new_items, **kwargs)
        except (openai.NotFoundError, openai.BadRequestError):
            # State gone (restore, retention purge, replica miss):
            # replay the FULL shadow history statelessly, then re-anchor.
            resp = self.client.responses.create(
                input=self.shadow + new_items, **kwargs)
        self.last_id = resp.id
        self.shadow += new_items + [it.model_dump() for it in resp.output]
        return resp
```

Costs of the shadow: client memory and the discipline to append faithfully. Benefits: DR restores don't strand thousand-turn agents, and the pattern doubles as your exporter for audit logs. (If you run `store: false` anywhere, the same shadow plus `include: ["reasoning.encrypted_content"]` is the whole state strategy.)

**Step 5 — promote tools server-side selectively.** Once traffic is stable, promote your boring, internal, high-frequency tools (retrieval, code-exec sandbox, internal APIs — including platform-run MCP servers per §7.8) from client-owned to gateway-owned. Each promotion removes one client↔gateway round-trip per agent step; at 30 steps per task that is real latency. Internet-backed built-ins stay off; that is an architecture invariant, not a config default.

**Step 6 — retire the legacy agent path.** Per-client metrics from step 1 tell you when agent traffic on chat completions hits zero. Freeze it for agents (410 for new API keys, grandfather old ones one release cycle); keep the endpoint alive for batch/eval tooling indefinitely. Define response-state retention (e.g. 30 days), test the restore path, and document loudly: **restoring the gateway without its DB orphans every `previous_response_id` in the fleet** — which is survivable only because step 4 made every client able to replay.

---

## Study questions

1. **What is the fundamental difference between `/v1/chat/completions` and `/v1/responses`, and why does it matter more for agents than for chatbots?**
   Answer: Chat completions is stateless — the client re-sends the full transcript every call — while Responses is stateful: the server stores each response and the client continues via `previous_response_id`. Agents make many turns per task with huge shared prefixes and (on reasoning models) internal reasoning that must survive between turns, so statefulness cuts payloads, preserves reasoning items server-side, and standardizes the tool loop.

2. **When exactly does the Assistants API stop working, and what replaced its "thread" concept?**
   Answer: It was deprecated August 26, 2025 and sunsets August 26, 2026. Conversation objects (`/v1/conversations`) plus stored Responses chains replace threads/runs.

3. **Name four request-field renames in the chat-completions → Responses migration.**
   Answer: `messages` → `input` (+ system prompt → top-level `instructions`), `max_tokens`/`max_completion_tokens` → `max_output_tokens`, `response_format` → `text.format`, and `finish_reason` (response side) → `status` + `incomplete_details`. Tool definitions also flatten: `tools[].function.name` → `tools[].name`.

4. **In a Responses stream, how do you know a tool call is starting and how do its arguments arrive?**
   Answer: A `response.output_item.added` event carries a `function_call` item; arguments then stream as `response.function_call_arguments.delta` events scoped to that item, ending with `.done`. You never diff anonymous chunks as in chat completions.

5. **Why must `instructions` be re-sent on every `previous_response_id` continuation, and what is the billing caveat of chaining?**
   Answer: Instructions are deliberately not inherited from the previous response, so a stale system prompt can't silently persist. Billing-wise, all prior input tokens in the chain are still billed as input each turn — state saves payload and client complexity, not token count (caching saves the compute).

6. **How does forking a conversation work in the Responses API?**
   Answer: Any stored response ID can be used as `previous_response_id` by multiple different follow-up requests; each continuation becomes an independent branch of the same history. This enables retries, A/B turns, and tree-style agents without client-side transcript copies.

7. **In Anthropic's Messages API, how are tool results returned to the model, and what is the rule for parallel calls?**
   Answer: As `tool_result` content blocks inside the *next user message*, each carrying the `tool_use_id` of the call it answers; when the model made several `tool_use` calls in one turn, all `tool_result` blocks go back in a single user message. Errors are flagged with `is_error: true` rather than omitted.

8. **Why does the SOP route production `/v1/responses` traffic through the agentic-api gateway instead of vLLM's built-in support?**
   Answer: vLLM's built-in implementation keeps response/message state in engine memory, so it neither survives restarts nor spans replicas — a follow-up with `previous_response_id` cannot be routed to a different server. The gateway persists state in SQLite (upgradeable), adds `/v1/messages`, WebSocket streaming, and an explicit gateway/client/provider tool-ownership model.

9. **What changed structurally in the MCP 2026-07-28 spec versus 2025-11-25, and why?**
   Answer: The protocol core went stateless: the `initialize` handshake and session IDs were removed, cross-call state moved to explicit handles in tool arguments, server-initiated requests became multi round-trip requests (`input_required` + client retry), and method/tool names moved into HTTP headers (`Mcp-Method`, `Mcp-Name`). The goal is deployment on ordinary load balancers and gateways without shared session state.

10. **Your new model "works" in chat but every agent tool call comes back as garbled text in the assistant message. What happened and what is the fix?**
    Answer: No (or wrong) `--tool-call-parser`: the model emitted its family-specific call syntax and vLLM passed it through as plain text, so no `tool_calls` JSON was produced and nothing errored. Fix: launch with `--enable-auto-tool-choice --tool-call-parser <family>` (verify the family's parser exists in your pinned vLLM), re-run the 20-prompt tool-call gate, and monitor tool-call parse-failure rate.

11. **How does guided decoding guarantee schema-valid JSON, and what are its two main costs?**
    Answer: The backend (xgrammar/llguidance) compiles the schema into an automaton and masks every token that would make the output invalid at each decode step — invalid JSON becomes unrepresentable. Costs: first-request schema compilation latency (seconds, then cached) and schema feature limits (no recursion, no numeric/length constraints — enforce those client-side).

12. **What one line of code prevents an air-gap violation when using the OpenAI Agents SDK against your gateway?**
    Answer: `set_tracing_disabled(True)` — the SDK otherwise exports traces to api.openai.com even when all model traffic points at your internal `base_url`.

---

## Sources

Primary sources fetched or verified August 2026:

- OpenAI — Migrate to the Responses API: https://developers.openai.com/api/docs/guides/migrate-to-responses
- OpenAI — Responses API reference (create): https://developers.openai.com/api/reference/resources/responses/methods/create
- OpenAI — Responses streaming events reference: https://developers.openai.com/api/reference/resources/responses/streaming-events
- OpenAI — Conversation state guide: https://developers.openai.com/api/docs/guides/conversation-state
- OpenAI — Deprecations (Assistants API sunset): https://developers.openai.com/api/docs/deprecations
- OpenAI community announcement — Assistants API beta deprecation, August 26 2026 sunset: https://community.openai.com/t/assistants-api-beta-deprecation-august-26-2026-sunset/1354666
- vLLM — Tool calling docs (parser catalog): https://github.com/vllm-project/vllm/blob/main/docs/features/tool_calling.md
- vLLM — Structured outputs docs: https://github.com/vllm-project/vllm/blob/main/docs/features/structured_outputs.md
- vLLM — OpenAI-compatible server (Responses endpoints): https://github.com/vllm-project/vllm/blob/main/docs/serving/online_serving/openai_compatible_server.md
- vLLM — Claude Code integration: https://docs.vllm.ai/en/latest/serving/integrations/claude_code/
- vLLM — RFC: Responses API full functionality without stores (state-routing problem): https://github.com/vllm-project/vllm/issues/24603
- vLLM — RFC: Anthropic Messages API in the Rust frontend: https://github.com/vllm-project/vllm/issues/47753
- vLLM — v0.26.0 release: https://github.com/vllm-project/vllm/releases/tag/v0.26.0
- vllm-project/agentic-api (gateway README): https://github.com/vllm-project/agentic-api
- Anthropic — Messages API and tool-use documentation: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- MCP — 2025-11-25 spec release (one-year anniversary post): https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/
- MCP — the 2026-07-28 specification: https://blog.modelcontextprotocol.io/posts/2026-07-28/
- MCP — registry preview announcement: https://blog.modelcontextprotocol.io/posts/2025-09-08-mcp-registry-preview/
- MCP — official registry: https://registry.modelcontextprotocol.io/
- OpenAI Agents SDK — configuration (custom clients, tracing): https://openai.github.io/openai-agents-python/config/
- LangChain — ChatOpenAI reference (`use_responses_api`, OpenAI-compatible providers): https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI
- Companion documents: [SOP: Production AI Model Deployment](../SOP-production-ai-model-deployment.md), [Primer](../PRIMER.md)
