# P/D pressure routing: per-PR stress and benchmark evidence

## PR4 real-router high-pressure simulation

This starts the actual compiled Rust router against four local HTTP engines
(2P+2D) implemented in `capacity_sim.py`. The Python engines never choose a
route. Actual cache-aware decisions, pressure scraping/validation, pair admission,
metadata handoff and request occupancy execute in the Rust router. There is no
model, accelerator or physical KV connector in this experiment.

Each simulated engine has 128 capacity blocks and 8 service slots. Background
requests hold 116 blocks (90.625%) on hot workers or 16 (12.5%) on low-pressure
workers. A foreground request reserves `3 + ceil(input_words / 64)` blocks;
unallocated waiting requests do not count as used blocks. The metric is
`(background + allocated foreground blocks) / 128`. Capacity and slot exhaustion
cause actual asynchronous waiting. Prefill service takes
`0.03 + (input_words - cached_prefix_words) * 0.0005` seconds, with up to 32
prefix words reused after the first visit. Decode simulates metadata transfer
and emits one chunk every 0.01 seconds. These are declared simulation parameters,
identical across arms, not calibrated vLLM or NPU timings.

```bash
export RUSTUP_TOOLCHAIN=1.95.0
export CARGO_BUILD_JOBS=2
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_INCREMENTAL=0
cargo build --features cpu,ready --bin vllm-router
python3 -B scripts/benchmarks/pd_pressure/run.py stress \
  --stage 4 --base-ref BASE_COMMIT_SHA \
  --router-binary /absolute/path/to/vllm-router \
  --num-prompts 64 --request-rate 32 --stress-concurrency 64 \
  --repeat 2 --timeout 300 --out-dir /absolute/new/results
```

Requires Python 3.10+, aiohttp, Linux/WSL process groups and an existing router
binary. `--simulation-profile` can be repeated to select `low`, `p_hot`, `d_hot`,
`both_hot`, `all_hot`, `recovery`, `signal_fault` (default: all). Each profile
compares off with P-only on, D-only on and both on. `--simulation-modes p|d|both`
selects candidates; `--simulation-modes aa` instead runs off/off A/A. Every
comparison has independently reset AB then BA pairs (`--repeat >= 2`), the same
fixed input/arrival trace and eight sequential warmup calls. There are eight
initial burst arrivals, followed by the configured sustained rate. Long/short
inputs are 257/65 words, not vLLM tokens. No arm-specific speedup is injected.

`recovery` releases the hot background after 1 second. `signal_fault` exports
an incorrect P1 epoch between 0.6 and 1.4 seconds while leaving its health and
capacity intact; a sufficiently long trace must exercise real-router identity
fallback. The profile starts after warmup and 150 ms before replay; both the
profile events and measured request times are retained. Queueing/phase results
should be interpreted using those timestamps, not assumed instantaneous polling.

Every round drains, captures backend request/handoff records, and shuts down
the owned router process before the next round. AB/BA includes on-to-off
restart comparisons. Existing in-flight attempt accounting across mode changes
is separately covered by Rust admission tests; this is not a hot-reload test.
Counter deltas must account for every measured decision in enabled stages;
backend IDs, completions and handoffs must reconcile one-to-one; both backend
allocations and actual router P/pending-D/active-D occupancy must end at zero.

`simulated-results.md/json` summarize all rounds. Each child contains the PR3
plan, raw request JSONL, raw Prometheus samples, process logs, source/binary
hashes, `backend-audit.json`, `reconciled.json` and paired reports. Completion
denominators include every planned request. TTFT is first generated content
minus HTTP-call start; queue time is additionally recorded inside each engine.
The predeclared functional budgets are zero failures, client lag <=100 ms and
mean paired per-length-group TTFT regression <=15%. Any violations remain in
the output. Two pairs yield no confidence interval; use >=3 for a paired-round
bootstrap interval. Mock `performance_accepted` remains false: simulated
improvement characterizes this capacity/service model, not NPU production gains.

`scripts/benchmarks/pd_pressure/run.py` records per-PR evidence. PR1/2 and the
official-CLI adapter use the standard library; PR3's HTTP client and fixtures
also require `aiohttp` (already a project dependency), with Python 3.10+.
PR3 can start only the processes explicitly declared in an experiment plan.
Stage 4 starts a real Rust router and local finite-capacity simulated workers; other actions do not launch deployment services. CPU stress and live
benchmarks are different evidence: CPU tests establish routing/accounting
behavior; `vllm bench serve` can measure client-observed TTFT through the public
HTTP API. HTTP P/D is the primary validation path; gRPC P/D remains a
compatibility check. The same public-API benchmark can exercise either backend
transport when the corresponding deployment is explicitly configured.

The actual P/D connector, engine capabilities, and backend readiness must be
established separately. The runner does not infer them from an HTTP URL.

## Entry points by PR

| Stage | CPU stress filter | Current status |
|---|---|---|
| PR1 admission | Library `routers::http::pd_router::admission_tests::pd_cache_aware_pr1_stress` | Shared accounting check. |
| PR1 HTTP P/D | Target `test_pd_routing`, exact filter `test_pd_routing::http_pd_cache_aware_pr1_stress` | Primary routing stress. |
| PR1 gRPC P/D | Target `grpc_pd_routing`, exact filter `grpc_pd_cache_aware_pr1_stress` | Compatibility stress. |
| PR2 admission | Library `routers::http::pd_router::admission_tests::pd_cache_aware_pr2_stress` | Shared shadow/baseline accounting. |
| PR2 HTTP P/D | Target `pd_pressure_routing`, exact filter `http_pd_pressure_pr2_stress` | Four independent stage-mode combinations. |
| PR2 gRPC P/D | Target `grpc_pd_routing`, exact filter `grpc_pd_pressure_pr2_stress` | Shadow compatibility and baseline preservation. |
| PR3 | Python `mock_pd.selftest`, through `stress --stage 3` | A/A and off/shadow tooling selftests, explicit experiment plans and strict reports. |
| PR4 | `high_pressure.suite`, through `stress --stage 4` | Real Rust HTTP router, opt-in on, capacity/queue simulation and paired reports. |

Each later PR must implement its stress test, register its actual entry point,
update the tests, and document the new evidence before
its entry point can succeed. There is no arbitrary filter override that would
silently turn an unimplemented stage into a passing stage.

All commands below run from the repository root in Linux/WSL. Replace the
explicit paths and comparison commit with the artifacts actually under test.
Every `--out-dir` must be new: existing evidence is never overwritten. Keep
evidence outside the checkout/build target so cleanup does not remove it.

### PR1 CPU stress

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0
export CARGO_TARGET_DIR="$HOME/.cache/router-kv-pressure-pd-target"
export CARGO_BUILD_JOBS=2

python3 scripts/benchmarks/pd_pressure/run.py stress \
  --stage 1 --base-ref BASE_COMMIT_SHA \
  --cargo "$HOME/.cargo/bin/cargo" --profile release \
  --repeat 2 --stress-rounds 24 --stress-concurrency 16 --timeout 3600 \
  --out-dir /absolute/path/to/benchmark-artifacts/pr1-stress-run1
```

Each repeat runs admission, HTTP P/D, and gRPC P/D in that order. All three
suites must pass; a missing filter resulting in zero tests is a failure.
The actual Cargo commands include `--locked
--release --features cpu,ready --message-format=json`, the appropriate `--lib`
or `--test test_pd_routing` / `--test grpc_pd_routing` target, and its exact test
filter from the table above. Libtest
receives `--ignored --exact --nocapture --test-threads=1`. Internal concurrency
remains available; libtest runs one selected test at a time. Two repetitions are
the default. `--stress-rounds` and `--stress-concurrency` override the corresponding
`ROUTER_PD_STRESS_ROUNDS` and `ROUTER_PD_STRESS_CONCURRENCY` environment values
(defaults 24 and 16).

All three suites emit `PD_PRESSURE_STRESS_RESULT` JSON, which is retained separately
and in the manifest. Pair-admission stress checks concurrent reservations,
cleanup and sentinel preservation. HTTP P/D mock stress exercises candidate
tokenization, P/D prefix hits, per-request handoff parameters, completion counts,
and zero final occupancy. The gRPC P/D mock stress retains compatibility checks
for one P and one D dispatch per request, KV handoff parameters, token consistency,
completion counts, and zero final occupancy. The result kinds are `pair_admission`,
`http_pd`, and `grpc_pd`; HTTP and gRPC records share the request/completion/handoff
counts and occupancy fields. Recorded rounds and concurrency must match the
requested values; admission `attempts` and HTTP/gRPC `requests` must each equal
`rounds * concurrency` exactly. The runner rejects a missing result, failed
invariants, zero passed tests, or a missing test-executable hash. Cargo's JSON
build messages provide the executable path; libtest's raw output and summaries
are retained in the same log. Compilation errors remain failures; the runner
does not install dependencies or rewrite a lockfile.

These are CPU mock stresses. Their wall time, pair-admission time, and mock
completion time are **not TTFT**, do not measure GPU prefill/decode, and do not
prove a capacity-policy performance improvement.

For an initial functional check, use `--profile test`; this omits `--release`
and uses Cargo's test profile. Optional low-memory build settings are
`CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0`.
The selected profile and these environment values are recorded. Test-profile
timings are functional evidence only and must not be used to assess performance.

Use separate target directories for incompatible branches/toolchains rather
than overwriting a retained reference binary. The global Cargo download cache
may be reused. A 32-core machine with only 8 GiB assigned to WSL should not use
32 concurrent compiler jobs without a separate memory check.

`--stage 3` runs the Python tooling selftest described below; it does not invoke
a Rust test filter. `--stage 4` still records `not_implemented` and exits 3.

### PR2 CPU shadow stress

Use the same stress command with `--stage 2` and a new output directory.
Each repeat runs `admission_shadow`, `http_shadow`, and `grpc_shadow` with the
exact ignored filters in the table. With `N = stress_rounds * stress_concurrency`:

- Admission emits `pair_shadow`: exactly N attempts, zero final load entries,
  preserved sentinel, and `shadow_actual_baseline: true`.
- HTTP emits `http_pd_shadow`: exactly 4N requests, completions and handoffs;
  four unique cases covering off/off, shadow/off, off/shadow and shadow/shadow;
  each case has exactly N requests/completions/handoffs and zero final occupancy.
- gRPC emits `grpc_pd_shadow`: exactly N requests/completions/handoffs, zero
  final occupancy, and `shadow_actual_baseline: true`.

The runner rejects incomplete or duplicated cases, incorrect counts, missing
baseline-preservation assertions, zero tests and missing executable hashes.
All assertions concern CPU mock routing, observation and lifecycle behavior;
PR2 keeps actual routing on the baseline and provides no TTFT benefit evidence.
PR3 adds experiment tooling; actual pressure overrides remain a PR4 feature.

### Configure native P/D pressure observation

The Rust CLI and Python launcher accept the same `--pd-pressure-config` JSON
string; the Python constructor keyword is `pd_pressure_config`. The Rust
`RouterConfig` JSON field is the structured object `pd_pressure`. Omission
means both stages are off. Each stage defaults to off independently. `shadow`
keeps the baseline; `on` commits the proposed low-pressure target after both
stage selections succeed. When both are off, no pressure
collector starts and existing Rust routing modes retain their original path.
Invalid configuration is rejected even when off. Python MiniLB rejects any
explicit pressure configuration rather than silently ignoring it.

An enabled stage requires native vLLM P/D mode and that stage's effective
`cache_aware` policy (stage override, otherwise the global policy). Program
Scheduling and IGW are unsupported. Enabled pressure also rejects the vLLM ZMQ
`--vllm-discovery-address` / `discovery_address` path: its candidates are rebuilt
without the stable registry bindings required by the collector. Omission or
explicit all-off preserves the existing Rust discovery behavior.
In `on`, the original target must have `usage >= high` (default 0.85).
Alternatives must belong to the same original eligible stage pool and have
`usage <= low` (default 0.70). The complete stage snapshot must pass the same
identity, age, skew and layout checks as shadow. The original cache-aware rule
is evaluated again inside that low-pressure subset, including affinity and
its existing imbalance logic. A larger request count does not veto a candidate.
Invalid signals, missing effective tokens or no low-pressure candidate retain
the stage baseline. Only the final actual pair updates cache trees and occupancy.
`vllm_router_pd_pressure_overrides_total{stage}` counts committed actual overrides;
decision counters distinguish mode `on`/reason `override` from shadow proposals.
Modes are startup configuration: drain and restart with `off` to roll back;
there is no runtime configuration API.
HTTP P/D is the primary deployment path;
gRPC shares routing compatibility but still needs an explicit HTTP(S) metrics
endpoint. Neither an HTTP inference URL nor a gRPC URL establishes the metrics
address, authentication, identity or metric layout.

Save the following illustrative object as `pd-pressure.json`, replacing the
addresses, model and expected identity with deployment-verified values:

```json
{
  "prefill": {
    "mode": "shadow",
    "high": 0.9,
    "low": 0.8,
    "mappings": [{
      "inference_endpoint": "http://prefill:8000",
      "dp_rank": null,
      "metrics_url": "http://prefill-exporter:9100/metrics",
      "usage": {
        "name": "vllm:kv_cache_usage_perc",
        "labels": {"engine": "0", "model_name": "my-model"}
      },
      "identity": {
        "name": "router_pd_engine_identity",
        "labels": {"engine": "0", "model_name": "my-model"}
      },
      "expected_engine_id": "prefill-engine-0",
      "expected_instance_epoch": "pod-start-42",
      "expected_layout_fingerprint": "layout-v1"
    }]
  },
  "decode": {"mode": "off"}
}
```

```bash
vllm-router --vllm-pd-disaggregation \
  --prefill http://prefill:8000 --decode http://decode:8000 \
  --policy cache_aware --pd-pressure-config "$(cat pd-pressure.json)"
```

`router_pd_engine_identity` is an **example custom exporter contract**, not a
standard vLLM metric. The deployment must establish that the exporter reads the
identified live engine and reports its current process generation and KV layout.
Copying configured strings into an exporter does not establish that identity.
A native usage gauge with only an `engine` index is insufficient. Without a
verified identity source the observation is invalid and routing stays on the
baseline; this feature is not ready to use with unmodified vLLM metrics alone.
An empty mapping list is allowed for unsupported-signal fallback testing, and
provides no capacity observation.

One scrape response must contain exactly one usage series selected by its
configured name/labels, with a finite value in [0, 1], and exactly one selected
identity series with value **1**. For the example, the relevant response is:

```text
vllm:kv_cache_usage_perc{engine="0",model_name="my-model"} 0.93
router_pd_engine_identity{engine="0",model_name="my-model",stage="prefill",endpoint="http://prefill:8000",dp_rank="aggregate",engine_id="prefill-engine-0",instance_epoch="pod-start-42",layout_fingerprint="layout-v1"} 1
```

The identity labels `stage`, `endpoint`, `dp_rank`, `engine_id`,
`instance_epoch` and `layout_fingerprint` must equal the mapping. Every usage
selector label, including `engine` and `model_name`, must also match on identity.
If the usage series carries any reserved identity label, that label must agree
as well; an aggregate identity cannot validate a rank-specific usage series.
For a rank-specific mapping use numeric `dp_rank` in JSON, its decimal string in
both the usage selector and identity series, and that rank's verified engine
identity; a null rank uses identity `dp_rank="aggregate"`. Matching labels do not
prove identity without the deployment verification above.

Missing/duplicate series, nonfinite/out-of-range usage, changed engine epoch or
layout, authentication errors, oversized responses and timeouts invalidate the
latest sample; a prior good sample cannot mask a new failure. Freshness and
cross-worker skew remain bounded. Collection is capped at **500 ms** and age
at **2 s from scrape start**, including for other consumers of the shared API.
The defaults are a 1 s poll interval and 1.5 s maximum cross-worker skew;
deployments can configure stricter limits but cannot relax the common caps.
Scrape completion time is not the engine's
source sample time: the latter is unknown unless separately supplied. Optional
`auth_token_env` names the environment variable holding a bearer token; secrets
are not embedded in URLs or configuration. The mapping must follow deployment
changes; the router does not infer a new epoch or layout from a reused address.

### Shared observations and the first consumer

`src/engine_pressure/` owns collection, parsing, engine/rank binding, generation
fencing, failure invalidation, and freshness. `EnginePressureRuntime::observe`
returns `PressureObservation::Known(EngineSample)` or `Unknown(reason)` for an
explicit `EngineKey` and the caller's current worker object. This object check
prevents a replacement at the same URL from borrowing the previous process's
sample. The stage snapshot uses the same identity and freshness validation,
then checks completeness, timing skew and equal KV layout across candidates.
No policy, affinity, high/low threshold, or request count enters collection.

`src/kv_pressure/` is the native PD adapter. It translates the existing JSON
configuration to collection settings and keeps the off/shadow controls and
high/low thresholds. Its first consumer is native HTTP PD `cache_aware` without
a usable KV event source. A high baseline is `usage >= high` (default 0.85);
eligible low candidates satisfy `usage <= low` (default 0.70). The policy runs
its original affinity rule again within that subset. There is no additional
request-count veto. Shadow records the proposal and commits the baseline only.

This boundary is coordinated with the per-engine, 500 ms / 2 s proposal in
[PR #585](https://github.com/vllm-project/router/pull/585). This patch does not
enable that PR's full-engine admission filter or a second collector alongside
it; the implementations must converge before both are enabled together.
Shared availability does not enable a `kv_aware` consumer automatically.
Program Scheduling's router-accounted placement estimate remains unchanged and
is not presented as measured engine occupancy. Combining affinity/cost policies
or demonstrating superiority over `kv_aware` requires separate evaluation.

Validation includes policy-free observation API tests, invalid-data and stale
sample tests, HTTP P/D shadow and failure injection, and per-stage actual
selection counters. Upstream random ties and deferred Decode commitment remain
in force: Decode observes pressure at its actual selection point after Prefill
where the connector uses deferred commitment. Older benchmark results from
`2fb33999` plus the prototype patches are not results for this revision.

### Probe the installed vLLM CLI

```bash
python3 scripts/benchmarks/pd_pressure/run.py probe \
  --base-ref BASE_COMMIT_SHA \
  --vllm /absolute/path/to/vllm \
  --out-dir /absolute/path/to/benchmark-artifacts/cli-probe-run1
```

This records `vllm --version`, `vllm bench --help`, and the `serve` and `forward`
help output. It starts no model or server. Exit 3 means at least one requested
capability is unavailable; the manifest identifies each capability separately.
An executable that cannot start is an error, not evidence that a benchmark ran.
Grouped help is expanded with `--help=all` when the CLI advertises that option.

A local inspection on 2026-11-09 found vLLM **0.31.0** with `bench serve` but no
`bench forward`. The installed binary, rather than the latest documentation,
decides availability. No compatible vLLM version is installed automatically.

### Explicit paired TTFT runs through HTTP P/D

First start and validate baseline and candidate routers and their homogeneous
HTTP P/D backends outside this runner. Keep Program scheduling disabled. Specify the
actual connector; unknown connector behavior is not a validated P/D setup.
Both router URLs must support `/v1/chat/completions`; supply router addresses,
not individual worker addresses. gRPC P/D compatibility measurements can use
the same command with a separately validated gRPC deployment. Record backend
transport in deployment evidence; the runner does not detect it from the URL.

```bash
python3 scripts/benchmarks/pd_pressure/run.py serve \
  --stage 1 --base-ref BASE_COMMIT_SHA \
  --vllm /absolute/path/to/vllm --model /absolute/path/to/model \
  --baseline-url http://127.0.0.1:30001 \
  --candidate-url http://127.0.0.1:30002 \
  --baseline-binary /absolute/path/to/baseline/vllm-router \
  --candidate-binary /absolute/path/to/candidate/vllm-router \
  --input-len 1024 --output-len 64 --num-prompts 32 \
  --request-rate 4 --seed 17 --repeat 3 --timeout 1800 \
  --out-dir /absolute/path/to/benchmark-artifacts/pr1-serve-run1
```

This explicitly executes `vllm bench serve` with the `openai-chat` backend,
random token-length inputs, the public chat endpoint, and `--save-result --save-detailed`.
It requests p50/p95/p99 for TTFT, TPOT, ITL, and end-to-end latency explicitly.
Each pair uses the same seed, requested input/output lengths, prompt count, and
arrival-rate parameters. Baseline/candidate execution order alternates between
pairs. Every CLI invocation and its raw output are retained. `completed` must
equal the requested prompt count in each saved result; exit 0 alone is not
sufficient.

Binary hashes are computed from the files supplied by the caller. The runner
does **not** attest that a remote URL is running that file: preserve deployment
commands, engine/router startup logs, connector settings, model revisions,
hardware topology, and readiness evidence alongside the run. Avoid concurrent
baseline and candidate engine workloads competing for the same GPU. Establish
and record comparable warm-up/cold-cache conditions yourself for this legacy
`serve` action. PR3's separate `experiment` action executes explicit lifecycle
hooks; it does not guess how to reset a particular engine or connector.

These small defaults exercise the measurement pipeline. They do not guarantee
KV pressure, a particular prefix-hit rate, controlled output termination, or
coverage of the two-long-versus-three-short scenario. A complete capacity study
must add fixed workload traces, pressure telemetry, candidate-selection counts,
multiple pressure regimes, length groups, and enough repetitions for confidence
intervals. PR1's unchanged behavior is a regression/overhead comparison, not a
claim that an unimplemented capacity override improves TTFT.

The status `completed_unassessed` means the paired runs completed and results
exist. It is deliberately not a performance acceptance decision. Compare the
same TTFT statistic on each arm, retain absolute milliseconds and per-run raw
values, and predeclare the allowed regression before evaluation.

### Optional model-forward diagnostic

```bash
python3 scripts/benchmarks/pd_pressure/run.py forward \
  --stage 1 --base-ref BASE_COMMIT_SHA \
  --vllm /absolute/path/to/vllm --model /absolute/path/to/model \
  --out-dir /absolute/path/to/benchmark-artifacts/pr1-forward-run1
```

This action probes `bench forward --help` before doing anything GPU-related. An
unsupported subcommand or incompatible interface produces `unsupported`, exit
3, and saved help/error output. A supported invocation explicitly runs the
installed CLI's forward defaults for the supplied model; inspect the saved help
to understand those defaults.

Forward benchmarks bypass the serving scheduler and do not traverse the router.
They cannot measure router TTFT, P/D transfer, candidate selection, queueing, or
end-to-end benefits. Use them only for model/GPU compute diagnosis. Do not
substitute `bench throughput` or `latency` and label it as a forward result.

## Evidence layout and exit codes

Each new artifact directory contains `manifest.json`, ordered raw command logs,
a source patch log, version output, resolved base/head commits and dirty status,
the runner/tool hash, and (when available) Cargo test-executable hashes. Serve
runs also contain per-arm `metrics.json` and the caller-supplied binary hashes.
The manifest records argv without shell interpretation, working directory,
start/end timestamps, elapsed seconds, command exit codes, and timeouts.
The `machine` object records CPU model, visible logical CPU count, and memory
bytes using standard-library OS observations. Linux uses `/proc/cpuinfo`, CPU
affinity, and `sysconf`; memory describes the exposed machine/VM memory, not a
container cgroup limit. Unavailable fields are `null` (including CPU model and
memory on Windows); values are not inferred. `test_binaries` retains its existing
path-to-SHA-256 format, while `test_binary_metadata` adds each file's `size_bytes`
and `mtime_ns` (Unix epoch nanoseconds) alongside the hash.
Git metadata and patch commands use invocation-local `core.autocrlf=true` so
Windows checkout line endings do not appear as full-repository changes under
WSL; no global Git setting is changed. Unimplemented stages record a status and
the supplied parameters without executing provenance or benchmark commands.
Required provenance commands must succeed. The patch contains tracked changes;
untracked files under `src`, `tests`, this runner's directory, and this document
(plus root Cargo/build files) are copied into `source-snapshot` with SHA-256
hashes. Other untracked files are listed in Git status but are not archived.

Only non-secret build environment settings are recorded. URLs with embedded
credentials or query strings are rejected. This initial runner does not accept
API keys; use a controlled benchmark deployment that does not require them.

| Exit | Meaning |
|---|---|
| 0 | CPU stress passed, both probe capabilities available, diagnostic completed, or serve completed unassessed; inspect `status`. |
| 1 | Execution, compilation, zero-test, missing artifact, incomplete request, or evidence error. |
| 2 | Invalid command-line arguments or attempted reuse of an evidence directory. |
| 3 | Stage not implemented or installed CLI capability unavailable. |

Run the runner's own tests without any GPU/model/Cargo build:

```bash
python3 -B -m unittest discover -s scripts/benchmarks/pd_pressure/tests -v
```

The tests cover all three PR1 and PR2 suites and their default two-repeat orchestration,
tiny sample output, command construction, invalid parameters,
zero/ignored tests, artifact extraction, evidence preservation, unsupported
forward handling, and unimplemented stages. The additional PR3 tests exercise
real local HTTP streaming, process cleanup, frozen workloads and strict statistics.

## PR3: fixed traces, deployment lifecycle and paired reports

PR3 validates the **measurement tool**, not the pressure-routing performance
hypothesis. `performance_accepted` is always false. Real NPU or other accelerator
validation requires a separately prepared deployment; no accelerator is started
by the CPU selftest. The Python mock simulates a P-to-D handoff in memory and
exposes two P and two D identities. It is not the production Rust router or a
real KV connector. Continue running `stress --stage 2` for the actual router's
HTTP/gRPC routing and lifecycle checks.

### Run the CPU tooling selftest

```bash
python3 -B scripts/benchmarks/pd_pressure/run.py stress \
  --stage 3 --base-ref BASE_COMMIT_SHA \
  --repeat 2 --stress-rounds 4 --stress-concurrency 4 --timeout 120 \
  --out-dir /absolute/path/to/pr3-selftest
```

With `N = stress_rounds * stress_concurrency`, each repeat executes A/A and
off/shadow experiments. Each experiment has two pairs in AB then BA order, with
N measured requests per arm per pair, and one separately accounted warmup
request per arm. Thus each repeat measures **8N requests**, plus eight warmup
requests. Every round must reconcile request, completion and mock handoff
counts; actual targets remain baseline in off/shadow; process cleanup must be
recorded. All source, trace, startup, metrics, request and report evidence is
retained. Fixture timing budgets are intentionally broad functional checks,
not deployment regression budgets. Two pairs do not establish a confidence
interval; that field is null rather than pretending the requests are independent.

To inspect and edit a complete, runnable plan instead of using the selftest:

```bash
python3 -B scripts/benchmarks/pd_pressure/mock_pd.py plan \
  --directory /absolute/path/to/new-pr3-input \
  --comparison off_shadow --count 16 --concurrency 4
python3 -B scripts/benchmarks/pd_pressure/run.py experiment \
  --stage 3 --base-ref BASE_COMMIT_SHA \
  --plan /absolute/path/to/new-pr3-input/plan.json \
  --timeout 120 --out-dir /absolute/path/to/new-pr3-run
python3 -B scripts/benchmarks/pd_pressure/run.py report \
  --stage 3 --base-ref BASE_COMMIT_SHA \
  --input-dir /absolute/path/to/new-pr3-run \
  --out-dir /absolute/path/to/new-pr3-report
```

Report regeneration reads the frozen artifacts and performs no HTTP inference.
It rejects incomplete rounds, missing/duplicated request IDs, contradictory
timestamps, changed request/metrics files and changed frozen plan/trace hashes.
An experiment with failed requests can still produce a report, but failed
requests remain in the denominator and a violated budget exits 1.

### Workload and timing contract

`workload.py` writes a fixed JSONL trace. It supports constant, Poisson and burst
arrivals with an explicit seed, and preserves the exact same trace for every
arm and round:

```bash
python3 -B scripts/benchmarks/pd_pressure/workload.py \
  --output /absolute/path/to/new-trace.jsonl --model MODEL_ID \
  --count 64 --rate 8 --seed 17 --arrival poisson \
  --short-words 32 --long-words 256 --output-tokens 64 \
  --scenario prefill-pressure
```

Each record contains `id`, `offset_s`, predeclared `scenario`, `length_group`,
`prefix_group`, and the exact streaming chat `body`. Synthetic input lengths
are **word counts**, not tokenizer lengths. Output `max_tokens` is an upper
bound, not proof of the realized generation length. Server usage, when present,
is saved per request. Use a tokenizer-verified trace for exact token-length
experiments; do not relabel requests based on the measured arm's pressure.
Only one text choice is supported; tool calls and multimodal messages are
rejected. Put only shareable synthetic prompts in public evidence.

`replay.py` schedules all requests against their original arrival times.
`concurrency` limits active HTTP calls but does not move arrival times or drop
queued requests. Each record retains planned time, client-call start, first
generated content, completion/failure, client lag and observed chunk intervals.
`sent_ms` is the start of the HTTP call including connection setup, not a wire
timestamp. Client lag is `sent_ms - planned_ms`; TTFT is
`first_content_ms - sent_ms`. The additional arrival-to-first-content value
includes client backlog. Role-only events do not count as generated content;
nonempty text/reasoning content does. `[DONE]` and generated content are both
required for completion. Chunk intervals are not token ITL.

HTTP errors, total request timeouts, malformed SSE and EOF without `[DONE]` are
failures. An interrupted replay preserves records even for unsent requests, and
the incomplete experiment cannot be reported as valid. Redirects are not
followed. The current client expects a controlled API without authentication;
it does not infer credentials or copy proxy settings from the environment.

### Deployment plan contract

The generated fixture plan is a complete schema example, not an NPU deployment
recipe. Replace its values and commands with the actual environment before
using `evidence_level: deployment`:

| Field | Required meaning |
|---|---|
| `schema_version`, `evidence_level`, `backend_transport` | Version 1; `mock` or `deployment`; HTTP backend. gRPC regression is a separate stage-2 run. |
| `comparison`, `repeats`, `seed`, `cache_state` | `aa`, `off_shadow`, or `candidate`; at least two pairs; fixed random seed; `cold` or `warm`. |
| `hardware`, `model`, `connector` | Accelerator kind/description (for example NPU), model/tokenizer revisions and template identity, actual connector name/version/transfer mode. Unknown connector is rejected for deployment evidence. |
| `trace`, `concurrency`, timeouts | Existing JSONL; active HTTP call limit; request, readiness and telemetry timing parameters. |
| `budgets` | Predeclare maximum failure fraction, maximum client lag in ms, and maximum mean paired TTFT regression in percent. |
| `arms.baseline`, `arms.candidate` | Explicit router URL, local binary and source repository, full captured RouterConfig JSON, and both declared stage modes. |
| `topology` | Stage, inference endpoint/rank, engine ID, instance epoch, layout fingerprint, metrics URL and explicit usage/identity selectors for every worker. Every metrics URL must be sampled. |
| `telemetry` | Unique source name, stage and exact URL; raw Prometheus bodies and sample times are retained. |
| `start`, `reset`, `warmup`, `stop` | Arrays of argv arrays, never shell strings; replace `{artifact_dir}`, `{arm}`, `{iteration}` where needed. |
| `ready_urls` | Health endpoints checked before reset/warmup/measurement. |
| `require_overrides`, `override_counters` | Stages whose actual override must be exercised, and explicit source/metric/label selectors for the future candidate's actual counter. Missing/ambiguous/reset/zero counters cannot establish override coverage. |

The config modes must agree with the declared modes. The captured config also
preserves pressure thresholds and mappings; omitted low/high currently mean
0.70/0.85, matching the Rust configuration. A/A requires identical binary,
configuration and source state. Baseline and candidate run sequentially in
alternating order to avoid competing accelerator workloads created by this
runner. The caller must also exclude unrelated external workloads.

Reset is mandatory every round; warm cache requires a warmup command; cold
cache rejects one. Command success does not prove the reset actually cleared
engine caches: verify the backend-specific semantics before performance work.
Managed startup/cleanup requires Linux/WSL and uses owned process groups only.
Remote or externally managed services require explicit stop hooks. Runtime
source/binary/config changes invalidate the experiment. Captured URL/binary
mappings and live identity claims are not remote-process attestation.

### Statistics and evidence

Each experiment retains `plan.json`, `topology.json`, `trace.jsonl`, source
patches and untracked contents under `sources/`, binary fingerprints,
`commands.txt`, per-round process logs, `workers/*.prom` plus sample metadata,
`requests.jsonl`, `rounds.csv`, `report.json` and `report.md`.

The report groups by predeclared scenario/input group, with a shared cache
condition. It reports every arm/round's planned, completed, failed and TTFT
sample counts; completion rate and completed requests/s; p50/p95 TTFT and
completion latency; and client lag/backlog. A request that emitted first content
then failed contributes a TTFT sample **and** a failure. A request with no first
content contributes a failure but no fabricated zero-latency sample.

Percentiles use linear interpolation. For each pair, absolute improvement is
`baseline_p95_ms - candidate_p95_ms`; relative improvement is that difference
divided by **baseline** p95, multiplied by 100. Reports retain all raw pair
values, AB/BA order means and the mean paired improvement. A 95% percentile
bootstrap resamples whole paired rounds (2,000 draws, fixed seed), requires at
least three pairs, and allows negative improvements. Independence still depends
on the deployment's reset protocol. It never treats correlated requests as
independent experiment repetitions.

`functional_valid` means the recorded experiment is complete and its declared
budgets/coverage checks passed. It is **not** a production performance verdict.
`invalid_measurement` exits 1 and includes the violated checks. Required override
coverage is opt-in: zero overrides are expected for off/shadow and are allowed
for explicitly declared no-op controls. An on candidate with a required stage
and no observed overrides is marked unexercised. PR3 does not implement on.

The legacy official `serve` adapter remains useful for cross-checking client
measurements. It probes the installed help and requests detailed results; it
does not claim the strict fixed-trace/backlog contract above. `forward` remains
compute diagnosis only. No performance result is inferred from unavailable CLI
capabilities. Ruff and the CPU tooling unit tests now cover this script
directory through `.pre-commit-config.yaml` as well as the documented commands.

References:

- [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [vLLM forward benchmark](https://docs.vllm.ai/en/latest/cli/bench/forward/)
