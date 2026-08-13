# Air-Gapped Supply Chain: Harbor, skopeo, cosign, zarf, and the Bundle Pipeline

**Deep dive #11 — companion to the [deployment SOP](../SOP-production-ai-model-deployment.md) §4 and the [PRIMER](../PRIMER.md) §7.**
**Status: August 2026.** Versions cited: Harbor v2.15.x (current stable, v2.15.0 released March 2026), cosign v2.x, Zarf v0.83.x, ORAS CLI v1.3, Kyverno v1.15, huggingface_hub v1.x (`hf` CLI). All of these move; re-verify before copying commands into runbooks.

---

## Key takeaways

- The air gap does not remove supply-chain risk; it **moves all of the risk to the moment of transfer**. Everything in this document exists to make that one moment verifiable: content-addressing (digests), cryptographic signing, and frozen, recorded snapshots.
- **skopeo** is the workhorse for moving container images across the gap: `registry -> oci-archive -> registry`, no Docker daemon needed. Use `--all` or you will silently ship only one CPU architecture.
- **Harbor** is the enclave's registry of record: projects with RBAC, robot accounts for machines, replication rules, Trivy scanning (which works offline with a hand-carried database), tag immutability, and garbage collection. Since v2.13 it also has first-class support for AI models as OCI artifacts.
- **Keyless cosign signing depends on two internet services (Fulcio and Rekor).** The practical air-gap answer is boring: key-pair signing with `--tlog-upload=false`, verification with `--insecure-ignore-tlog`, keys held offline or in a KMS/HSM. A private sigstore stack is possible but is a whole extra service to operate.
- Enforce signatures **at admission** (Kyverno `verifyImages` / `ImageValidatingPolicy`, or sigstore policy-controller) on Kubernetes, and **manually in the transfer runbook** everywhere else.
- For Python, pick your tier: a **wheelhouse directory + `pip install --no-index --require-hashes`** for production images (smallest, most verifiable), **devpi** as the enclave's caching index for tooling, **bandersnatch** only if you truly need all of PyPI (tens of terabytes).
- **Model weights are files, not images** (SOP stance). Download with `hf download --revision <commit>`, generate a `SHA256SUMS` manifest on the staging side, re-verify inside the enclave before first load. Pushing weights to Harbor via ORAS is a real 2026 option (CNAI/ModelPack annotations, Harbor model UI) but still loses to plain NFS on serving-path simplicity.
- **Zarf** solves a different problem than Harbor + Compose: it packages an *entire Kubernetes workload* (images, charts, manifests, files) into one signed tarball and bootstraps its own in-cluster registry. It becomes relevant at SOP Tier 3, not before.
- One **bundle = one manifest.yaml** listing every artifact with its digest, who approved it, and what evals it passed. The manifest is the deployment record; if it is not in the manifest, it does not cross the gap.
- The gap does not pause CVEs. Plan a **monthly patch train** (scan data + rebuilt images), a quarterly base-refresh, and rehearse registry and model-store disaster recovery like any other production datastore.

---

## 1. Why this discipline exists

### 1.1 The threat model in plain terms

A modern software deployment is assembled from thousands of parts you did not write: container base layers, Python wheels, OS packages, Helm charts, and now multi-hundred-gigabyte model checkpoints. A **supply-chain attack** compromises one of those parts upstream so that *you* install the attacker's code yourself, with your own credentials. The famous public examples — trojaned build systems, typosquatted PyPI packages, compromised registry accounts, malicious "updated" model checkpoints with pickle payloads — all share one shape: the artifact you received was not the artifact you thought you were receiving.

Two controls break that shape:

1. **Content-addressing.** Refer to every artifact by its cryptographic hash (an image *digest* like `@sha256:ab12...`, a file's SHA-256, a git commit hash). A hash pins *what the bytes are*, so a swapped artifact fails verification no matter where it came from.
2. **Signing.** A hash proves integrity but not origin. A signature (cosign for images and OCI artifacts, GPG for git tags/RPMs/debs) proves *who vouched for* those bytes — in our case, the staging-zone pipeline that scanned and eval-gated them.

The air gap adds a third control — **no network path for an attacker to phone home or for software to silently self-update** — but it also removes your safety nets: no `pip install` to fix a missing dependency, no registry fallback, no automatic CVE feeds. That trade is why every section below ends in "freeze it, hash it, sign it, record it."

### 1.2 Compliance framing (generic)

Disconnected operation is mandated by several regimes — classified enclaves, defense and government programs, some critical-infrastructure and financial rules. The specifics vary, but auditors consistently ask the same four questions, and the bundle pipeline is designed to answer them:

| Auditor question | Our answer |
|---|---|
| What exactly is running? | Digest-pinned images + hashed weights, recorded in `manifest.yaml` |
| Where did it come from? | Staging pipeline provenance; cosign signatures |
| Who approved it? | Approval block in the bundle manifest |
| What was it tested against? | Eval + scan evidence shipped inside the bundle |

---

## 2. Zone architecture recap

The SOP's two-zone model, with the standing services this document details:

```
[Connected staging zone]                          [Enclave / production]
                                                  runs only what arrived in bundles
  pull from internet:                             standing services:
    hf download (models)          one-way           Harbor         (images + OCI artifacts)
    skopeo copy (images)          transfer          model store    (NFS / PVC, plain files)
    pip download (wheels)        ─────────►         devpi          (frozen PyPI index)
    apt-mirror / reposync         media,            git server     (configs, mirrored recipes)
    git clone --mirror            data diode,       Trivy DB copy  (offline CVE scanning)
  then: scan, eval, SIGN,         or CDS            Prometheus / Grafana / PKI / NTP
        build bundle, hash
```

Roles, one sentence each: the **staging zone** is the only place downloading happens and the only place signing keys are used; the **transfer** is a controlled, logged, one-way crossing; the **enclave** verifies everything on arrival and thereafter treats the bundle manifest as ground truth. The unit of change for the enclave is the **bundle** (§11) — never an individual file someone carried over "real quick."

---

## 3. Moving container images: skopeo (and crane)

### 3.1 What skopeo is

**skopeo** is a daemonless CLI from the containers project for inspecting and copying container images between *transports* — different places an image can live. No Docker daemon, no local image store required, which makes it ideal for a staging pipeline. The transports you will use:

| Transport | Meaning |
|---|---|
| `docker://` | A remote registry (Docker Hub, NGC, Harbor, ghcr.io) |
| `oci:` | An OCI-layout directory on disk (the open standard on-disk format) |
| `oci-archive:` | The same OCI layout, tarred into one file |
| `docker-archive:` | A `docker save`-style tarball (legacy; single-image only) |
| `dir:` | Raw blobs + manifest in a directory (skopeo-internal, fastest) |

**Prefer `oci-archive:` (or `oci:`) for gap crossings.** `docker-archive:` cannot hold a multi-architecture manifest list and exists mainly for compatibility with `docker load`.

### 3.2 The three-step gap crossing

```bash
# 1. Staging: registry -> archive file  (--all = every architecture, not just the host's)
skopeo copy --all \
  docker://docker.io/vllm/vllm-openai:v0.26.1 \
  oci-archive:vllm-openai-v0.26.1.tar

# 2. Record the digest you intend to run (goes into the bundle manifest)
skopeo inspect --format '{{.Digest}}' docker://docker.io/vllm/vllm-openai:v0.26.1

# 3. Enclave: archive file -> Harbor
skopeo copy --all \
  oci-archive:vllm-openai-v0.26.1.tar \
  docker://harbor.enclave.local/infra/vllm-openai:v0.26.1

# 4. Verify what Harbor now serves matches the recorded digest
skopeo inspect --format '{{.Digest}}' \
  docker://harbor.enclave.local/infra/vllm-openai:v0.26.1
```

Useful flags: `--preserve-digests` (fail rather than rewrite anything that would change the digest), `--dest-creds robot$bundle-loader:TOKEN` (authenticate to Harbor), `--src-tls-verify` / `--dest-tls-verify` (internal PKI trust; fix your CA bundle rather than setting these to false).

### 3.3 Bulk transfers: `skopeo sync`

For a list of images (a whole release), `skopeo sync` copies many repositories in one command, driven by a YAML file:

```yaml
# images.yaml — one stanza per source registry
docker.io:
  images:
    vllm/vllm-openai: ["v0.26.1"]
    grafana/grafana:  ["11.6.0"]
nvcr.io:
  images:
    nvidia/k8s/dcgm-exporter: ["4.2.0-4.1.0-ubuntu22.04"]
```

```bash
# Staging: everything to one directory tree (--scoped keeps full source paths, avoiding name collisions)
skopeo sync --all --scoped --src yaml --dest dir images.yaml /media/transfer/images/

# Enclave: directory tree into Harbor
skopeo sync --all --scoped --src dir --dest docker /media/transfer/images/ harbor.enclave.local/mirror
```

### 3.4 crane, and when to use it instead

**crane** (from Google's `go-containerregistry`) overlaps with skopeo for copying but adds image *mutation*: append a layer, change an entrypoint or labels, flatten — directly against a registry, no build daemon. Rules of thumb:

- **Copy/inspect/sync across the gap:** skopeo. More transports, better packaged, copies signatures alongside images.
- **Writing Go tooling, or tweaking an image without a Dockerfile:** crane (`crane mutate`, `crane append`, `crane digest`).
- Either one replaces the anti-pattern of `docker pull && docker save` on a build box.

> **Common pitfalls**
> - **Single-arch surprise:** `skopeo copy` without `--all` copies only the manifest matching the *staging host's* architecture. If staging is x86 and works fine, you will not notice until an arm64 node appears. Always `--all`.
> - **Tag drift:** a tag (`:latest`, even `:v0.26.1`) is a movable pointer. The digest recorded at staging time is truth; the enclave verifies against the digest, and deployment manifests pin `image@sha256:...`.
> - **Signatures do not ride along automatically** in every path. Cosign signatures are separate OCI objects; use `cosign copy` (copies image + signatures/attestations) or explicitly copy the signature tags. Verify inside the enclave to prove you did it right.

---

## 4. Harbor deep dive

**Harbor** is a CNCF-graduated, self-hosted registry: the CNCF Distribution registry underneath, plus a Postgres database, Redis, a web UI, RBAC, replication, scanning, and policy. It is the enclave's single source of truth for images and (optionally) other OCI artifacts. Current stable as of August 2026: **v2.15.x**. For air-gapped installation, Harbor publishes an **offline installer tarball** (`harbor-offline-installer-vX.Y.Z.tgz`) containing all component images — carry that across the gap; do not try to pull Harbor's own images through Harbor.

### 4.1 Projects and RBAC

A **project** is Harbor's namespace and permission boundary (`harbor.enclave.local/<project>/<repo>:<tag>`). Sensible enclave layout:

| Project | Contents | Write access |
|---|---|---|
| `infra` | vLLM, gateway, exporters, monitoring | bundle-loader robot only |
| `mirror` | third-party mirrored images (scoped paths) | bundle-loader robot only |
| `models` | ORAS model artifacts (if used, §9.5) | bundle-loader robot only |
| `sandbox` | scratch/testing | engineers |

Roles per project run from limited guest through developer to project admin; production projects should have *no* human with push rights — humans approve bundles, robots push them.

### 4.2 Robot accounts

**Robot accounts** are non-human credentials for CI, nodes, and the bundle-import job. Facts that matter operationally (Harbor 2.13+ docs):

- Two scopes: **project robots** (one project) and **system robots** (span multiple projects). Prefer project robots; least privilege.
- Names get the prefix `robot$` — you log in as `robot$bundle-loader`.
- Push permission requires pull permission alongside it.
- Default token expiry is **30 days** (system-configurable). Decide deliberately: short-lived tokens are safer but an air-gapped cluster whose pull secret silently expires is a 2 a.m. outage. Common enclave pattern: long-lived pull-only robots for nodes, short-lived push robots for the import job.
- **Harbor does not store robot secrets** — copy the secret at creation time or recreate the robot.

```bash
# Nodes authenticate with a pull-only robot
docker login harbor.enclave.local -u 'robot$infra+pull-only'   # paste secret
```

### 4.3 Replication rules

Replication copies artifacts between Harbor and other registries by **policy**: push-based (Harbor → remote) or pull-based (remote → Harbor), with filters on repository name, tag, and label (glob patterns: `*`, `**`, `?`, `{a,b}`), and three trigger modes — manual, cron-scheduled, or event-based (fires on push/retag). You can cap bandwidth per task (KB/s) and, between two Harbors, enable copy-by-chunk.

Air-gap uses:

- **Staging Harbor → export:** in the staging zone, an event-based replication rule can fan every approved push into the "to be bundled" project automatically.
- **Enclave-internal:** replicate `infra` to a second Harbor in another security zone or site over an approved internal link.
- What replication is **not**: a way across the gap itself — there is no network path. skopeo archives cross the gap; replication organizes either shore.

### 4.4 Proxy-cache projects (staging side only)

A **proxy-cache project** is a read-only pull-through cache: you pull `harbor-staging/dockerhub-proxy/library/python:3.12` and Harbor fetches and caches from Docker Hub on first use (dodging Hub rate limits via HEAD checks), serving the cached copy if upstream is unreachable. Default retention on cached content is 7 days. Useful on the **staging** Harbor so builds never touch the internet directly; meaningless inside the enclave (there is no upstream to proxy). You cannot push to a proxy-cache project.

### 4.5 Trivy scanning inside Harbor — offline

**Trivy** is Harbor's default vulnerability scanner (since Harbor v2.2). Trivy compares an image's package inventory against a vulnerability database (`trivy-db`) that it normally auto-downloads — which fails in the enclave. The air-gap pattern:

1. In `harbor.yml`, configure the scanner for offline mode: `trivy: { skip_update: true, offline_scan: true }`. `skip_update` stops DB download attempts; `offline_scan` stops network lookups during scans (e.g., Maven Central for Java).
2. On staging, pull the databases as OCI artifacts with ORAS:

```bash
oras pull ghcr.io/aquasecurity/trivy-db:2         # -> db.tar.gz
oras pull ghcr.io/aquasecurity/trivy-java-db:1    # Java ecosystem DB
```

3. Ship them in each bundle and unpack into the trivy-adapter's cache volume (path is typically `/home/scanner/.cache/trivy/db/` inside the adapter container — verify against your Harbor version), then restart the adapter.

Alternative that scales better: run a small internal OCI registry (or a Harbor project) hosting the mirrored `trivy-db`, and point standalone Trivy at it with `--db-repository harbor.enclave.local/mirror/trivy-db` (plus `--java-db-repository`, `--checks-bundle-repository`). Note the DB artifacts use custom media types (`application/vnd.aquasec.trivy.db.layer.v1.tar+gzip`), which any OCI-artifact-capable registry (Harbor, zot) handles fine.

Scan-time policy: set per-project "**prevent vulnerable images from running**" with a severity threshold, and schedule "scan all" after each DB refresh so old images get re-judged against new CVE knowledge.

### 4.6 Tag immutability

Per-project **tag immutability rules** (pattern-matched) make matching tags write-once: no re-push, no retag over, no delete, no overwrite via replication. Set a rule like `release-*` / `v*` immutable in `infra` and `models`. This is the registry-side enforcement of the same idea as digest pinning — nobody can quietly change what `v0.26.1` means.

### 4.7 Garbage collection

Deleting a tag or repo in Harbor does **not** free disk; blobs are shared and only a **garbage collection (GC)** run removes unreferenced ones. Schedule GC (e.g., weekly, off-peak) in the admin UI; modern Harbor performs GC online (early versions forced the whole registry read-only for the duration — one more reason not to run ancient Harbor). Harbor v2.15 added options around tag deletion during GC and upstream-connection limits. With multi-hundred-GB model artifacts in the registry (§9.5), GC discipline stops being optional housekeeping and becomes capacity management.

### 4.8 High availability and backup

Harbor's state lives in exactly three places, which defines both HA and backup:

1. **Postgres** (projects, users, robots, policies, scan results) — for HA run an external replicated Postgres (Patroni or CloudNativePG on K8s); back up with regular dumps/WAL archiving.
2. **Redis** (job queues, sessions) — replaceable cache; HA via Sentinel if you run multiple Harbor cores.
3. **Blob storage** (the actual image bytes) — filesystem or S3-compatible object store (MinIO is the common enclave choice); back up at the storage layer, or via Velero on K8s.

HA shape: ≥2 stateless core/portal/jobservice replicas behind a load balancer, all pointing at the shared external Postgres/Redis/object store (the Harbor Helm chart supports this directly). For a Tier-2 (Compose) enclave, a single well-backed-up Harbor VM plus a rehearsed restore procedure is honestly acceptable — measure your real recovery time before buying HA complexity.

**Restore rehearsal is the control that matters:** restore Postgres + blob storage to a scratch VM quarterly and prove a node can pull a digest-pinned image from it.

### 4.9 Lightweight alternatives

| Option | What it is | When it fits |
|---|---|---|
| **zot** | CNCF OCI-native registry, single Go binary, no Postgres/Redis; OCI 1.1 referrers, cosign support, online GC + retention | Edge/forward sites; staging scratch registries; anywhere Harbor is overkill |
| **Distribution `registry:2`/`registry:3`** | The bare CNCF registry (Harbor's own core) | Throwaway local registries, CI; no UI/RBAC/scanning — not an enclave system of record |
| **Quay `mirror-registry`** | Red Hat's single-node minimal Quay (`mirror-registry install`, port 8443) | OpenShift-based enclaves bootstrapping with `oc-mirror` |

A good composite pattern: Harbor at the enclave core, zot at small forward locations, replicated one-way from Harbor.

---

## 5. Signing and verification: sigstore/cosign offline

### 5.1 The sigstore idea, and why "keyless" breaks offline

**sigstore** is an open standard for signing artifacts. Its flagship mode, **keyless signing**, removes long-lived private keys: you authenticate with **OIDC** (OpenID Connect — the "log in with your identity provider" protocol), **Fulcio** (sigstore's certificate authority) issues a certificate valid for ~10 minutes binding your identity to a one-time key, you sign, and the signature is logged in **Rekor**, a public append-only transparency log. Verifiers check the certificate chain and the Rekor inclusion proof.

Every step of that assumes reachable services: an OIDC issuer, Fulcio, Rekor, and TUF-distributed root material for verifiers. Inside an enclave none of them exist. Two honest options:

1. **Key-pair signing (recommended; boring is good).** Generate a key pair, guard the private key in the staging zone (file with passphrase, or better a KMS/HSM — cosign supports Vault, AWS/GCP/Azure KMS, and hardware tokens), distribute only the *public* key into the enclave.
2. **Private sigstore stack.** Run your own Fulcio + Rekor (+ internal OIDC + TUF root) inside your network using the sigstore Helm charts ("scaffolding"). You regain keyless ergonomics and a transparency log, at the cost of operating a CA and a log as production services. Justifiable for large multi-team enclaves; overkill for one platform team.

### 5.2 Key-based signing and verification, end to end

```bash
# Staging, once: create the signing key pair (passphrase-protected)
cosign generate-key-pair            # -> cosign.key (SECRET, stays in staging), cosign.pub

# Staging, per release: sign by DIGEST, skip the (unreachable/public) transparency log
cosign sign --key cosign.key --tlog-upload=false \
  harbor-staging.local/infra/vllm-openai@sha256:ab12...

# Move image + signatures together between registries
cosign copy harbor-staging.local/infra/vllm-openai:v0.26.1 \
            harbor.enclave.local/infra/vllm-openai:v0.26.1

# Enclave: verify with the public key; tell cosign not to expect a Rekor entry
cosign verify --key cosign.pub --insecure-ignore-tlog \
  harbor.enclave.local/infra/vllm-openai@sha256:ab12...
```

Notes: always sign the **digest**, not a tag (cosign warns if you don't). `--insecure-ignore-tlog` is alarmingly named but is precisely the correct flag for key-based verification with no transparency log — the "insecure" part is the absence of the *public log*, which you deliberately opted out of; `--private-infrastructure` and `--offline` serve adjacent purposes (private sigstore deployments / forcing bundled offline verification). Signatures live in the registry as OCI objects next to the image (tag `sha256-<digest>.sig`), so they cross the gap inside the same OCI archive if you copy them (hence `cosign copy`). `COSIGN_REPOSITORY` can redirect signature storage to a separate repo if policy requires.

Attestations (signed statements *about* an artifact — e.g., "this SBOM describes it", "these evals passed") use the same machinery:

```bash
cosign attest --key cosign.key --tlog-upload=false \
  --type cyclonedx --predicate sbom.cdx.json \
  harbor.enclave.local/infra/vllm-openai@sha256:ab12...
cosign verify-attestation --key cosign.pub --insecure-ignore-tlog --type cyclonedx \
  harbor.enclave.local/infra/vllm-openai@sha256:ab12...
```

### 5.3 Verification at admission (Kubernetes) and by hand

On a Tier-3 (Kubernetes) enclave, make the cluster itself refuse unsigned images. Two mainstream enforcers:

- **Kyverno** (policy engine): the classic `verifyImages` rule, and since Kyverno 1.15 the dedicated `ImageValidatingPolicy` resource. Key-based, no sigstore services needed:

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata: { name: require-signed-images }
spec:
  webhookConfiguration: { failurePolicy: Fail }
  background: false            # verify at admission only; no cluster-wide rescans
  rules:
    - name: verify-enclave-sig
      match: { any: [ { resources: { kinds: [Pod] } } ] }
      verifyImages:
        - imageReferences: ["harbor.enclave.local/infra/*", "harbor.enclave.local/models/*"]
          mutateDigest: true   # rewrite tag -> digest at admission
          attestors:
            - entries:
                - keys:
                    publicKeys: |-
                      -----BEGIN PUBLIC KEY-----
                      ...cosign.pub contents...
                      -----END PUBLIC KEY-----
                    rekor: { ignoreTlog: true }   # air gap: no transparency log
```

- **sigstore policy-controller**: the sigstore project's own admission controller with a `ClusterImagePolicy` CRD; same key-based configuration is possible. Pick one enforcer, not both.

On Tier 1/2 (no Kubernetes admission path), verification is a **runbook step**: the bundle-import script runs `cosign verify` against every image digest in the manifest *before* pushing to Harbor, and the deploy script re-verifies before `docker compose up`. Docker Engine has no native cosign gate, so the script *is* the gate — keep it short, auditable, and exit-on-failure.

### 5.4 SBOMs and offline vulnerability scanning

An **SBOM (Software Bill of Materials)** is a machine-readable inventory of every package inside an artifact, in **SPDX** or **CycloneDX** format. Generate them at staging with **syft** (Anchore's generator — runs fully offline, no phone-home):

```bash
syft harbor-staging.local/infra/vllm-openai:v0.26.1 -o cyclonedx-json > sbom.cdx.json
```

Why bother in an enclave: when the next big CVE lands, you grep SBOMs on a laptop instead of re-scanning every image ("which of our images contain `libwhatever` < 3.2?"). Ship the SBOM in the bundle and attach it as a cosign attestation (§5.2).

Scanning offline, two tools, same pattern (hand-carried DB):

- **Trivy**: standalone use mirrors the Harbor setup — carry `trivy-db`/`trivy-java-db` (§4.5), then `trivy image --skip-db-update --skip-java-db-update --offline-scan <image>`.
- **grype** (Anchore's scanner, pairs with syft): supports importing a vulnerability DB archive offline (`grype db import <archive>`), then scans SBOM files directly: `grype sbom:./sbom.cdx.json`.

> **Common pitfalls**
> - Signing a **tag** instead of a digest — the signature then blesses whatever the tag points to later.
> - Verifying in staging but not in the enclave. The enclave-side check is the one that counters tampering *in transit*; staging-side checks only counter upstream compromise.
> - Letting the Trivy DB go stale silently. A scanner with a six-month-old DB reports a comforting green that means nothing. Alert on DB age (it ships in every bundle; if bundles stop, the alert fires).
> - Losing the staging private key = unable to sign new releases; leaking it = attacker can sign. Passphrase + restricted host at minimum; KMS/HSM if available; and document key rotation (new key pair, dual-trust window in verifiers, retire old).

---

## 6. Python packages offline

Production containers should never pip-install at runtime (SOP rule) — this section is about *tooling* (eval harnesses, scripts, notebook users) and about building images reproducibly. Three tiers, smallest first:

### 6.1 Wheelhouse directory + `--require-hashes` (default for anything that ships)

A **wheelhouse** is just a directory of `.whl` files plus a hashed requirements lockfile.

```bash
# Staging: resolve, pin, and hash the full dependency tree
uv pip compile requirements.in --generate-hashes -o requirements.txt
#   (pip-tools' pip-compile --generate-hashes is the older equivalent)

# Staging: download wheels FOR THE ENCLAVE'S platform/python, not the laptop's
pip download -r requirements.txt -d wheelhouse/ \
  --platform manylinux2014_x86_64 --python-version 312 --only-binary=:all:

# Enclave: install with no index access and hard hash enforcement
pip install --no-index --find-links ./wheelhouse --require-hashes -r requirements.txt
```

`--require-hashes` makes pip refuse any package whose SHA-256 doesn't match the lockfile — the pip-world equivalent of digest pinning, and it also forces *every* transitive dependency to be pinned (`==`) or fail. A **constraints file** (`-c constraints.txt`) is the complementary tool: it pins versions *if* a package is installed without forcing installation — use one org-wide constraints file to keep every team's `torch`/`numpy`/`transformers` aligned with the frozen snapshot.

### 6.2 devpi: the enclave's caching index (for humans and CI)

**devpi** is a self-hosted PyPI-compatible index server (actively maintained; devpi-server 6.x, releases through July 2026). On staging it acts as a caching mirror of PyPI; the cached state can be transferred (or a replica synced while connected, then detached) so the enclave gets a working `pip install` for everything that was ever cached — and a clean failure for everything that wasn't, which is exactly the behavior you want. It also hosts *private* indexes for your own internal wheels. Point clients at it via `pip config set global.index-url https://devpi.enclave.local/root/frozen/+simple/`.

### 6.3 bandersnatch: the full mirror (rarely justified)

**bandersnatch** (PyPA's PEP 381/503/691 mirror client) replicates PyPI wholesale: `bandersnatch mirror` against a config file, re-run on schedule, `bandersnatch verify` to repair. A *full* mirror is tens of terabytes as of 2026 (check pypi.org/stats for the current figure) and mostly packages you will never install. Its allowlist/blocklist plugins can cut this drastically — an allowlisted bandersnatch mirror of your ~500 actually-used projects is a legitimate middle ground when policy demands "a real static mirror, no caching server."

### 6.4 uv in an air gap

**uv** (Astral's fast package manager) is now common on engineer workstations. Offline behaviors worth knowing:

- Global `--offline` flag: use only the local cache; fail rather than touch the network.
- It speaks the same standards, so it works against devpi (`UV_DEFAULT_INDEX=...`) or a wheelhouse (`uv pip install --no-index --find-links ./wheelhouse ...`), and `uv pip compile --generate-hashes` produces the hashed lockfiles for §6.1.
- Known gap: git-URL dependencies in `pyproject.toml` still want to reach the git host — mirror those repos internally (§8) and point the dependency at the internal URL.
- uv also manages Python interpreters themselves; in an enclave, host the interpreter archives internally and set `UV_PYTHON_INSTALL_MIRROR` to that URL.

**Decision table:**

| Need | Tool |
|---|---|
| Reproducible production image builds | wheelhouse + `--require-hashes` |
| `pip install` for engineers/CI inside the enclave | devpi (frozen/cached index) |
| Policy demands a static standards-based mirror | bandersnatch with an allowlist |
| Entire PyPI, no exceptions | bandersnatch full (budget tens of TB) |

---

## 7. OS packages: mirrors and the frozen-snapshot discipline

The enclave's Ubuntu/RHEL nodes need `apt`/`dnf` to work against internal mirrors — for the NVIDIA driver stack above all (driver, fabric manager, DCGM, container toolkit — mirror NVIDIA's CUDA/driver repo alongside the distro repos; NVIDIA's DGX OS docs describe the same air-gapped pattern).

- **Debian/Ubuntu:** `apt-mirror` (simple config in `mirror.list`, rsync-style tree you serve over HTTP) or — better — **aptly**, whose *snapshot* model is built for this discipline: `aptly mirror update`, `aptly snapshot create nodes-2026-08-12 from mirror ...`, `aptly publish snapshot ...`. Snapshots are immutable and atomically switchable, so "what was installable on 2026-08-12" is a named, permanent thing.
- **RHEL-family:** `dnf reposync --download-metadata` into a dated directory, `createrepo_c` if needed, serve over HTTP, and *keep* old dated trees (never `--delete` your history).

**The frozen-snapshot-date discipline:** every bundle release names the exact repo snapshot the fleet is on (e.g., `os-snapshot: 2026-08-01`). All nodes upgrade against that snapshot and nothing else; the next snapshot arrives as part of a planned patch train (§12), staged on the canary node first. This gives you the two properties an internet-connected `apt upgrade` destroys: any two nodes patched in the same train are byte-identical, and any node state can be reproduced later for debugging. Pin kernel and NVIDIA driver packages (`apt-mark hold` / `dnf versionlock`) independently of the snapshot switch — driver bumps ride the *engine* upgrade train with GPU node acceptance tests (DCGM diagnostics), not the routine OS train.

---

## 8. Git across the gap: `git bundle`

A **git bundle** is a single file containing git objects and refs that `git clone`/`fetch` can treat as a read-only remote — the designed-in mechanism for sneakernet git. Workflow for mirroring `vllm-project/recipes` (which the SOP requires in the enclave):

```bash
# Staging: mirror-clone once, refresh thereafter
git clone --mirror https://github.com/vllm-project/recipes.git recipes.git
cd recipes.git && git remote update

# Full bundle (first transfer): every branch and tag
git bundle create recipes-full-2026-08-12.bundle --all

# Incremental bundle (subsequent transfers): only commits since the last shipped state
git bundle create recipes-inc-2026-08-12.bundle last-shipped..main
git tag -f last-shipped main        # record the new baseline

# Enclave: verify BEFORE using — checks integrity and that prerequisites exist locally
git bundle verify recipes-inc-2026-08-12.bundle

# Enclave: first time -> clone; afterwards -> fetch into the internal mirror
git clone recipes-full-2026-08-12.bundle recipes.git       # first time
cd /srv/git/recipes.git && git fetch ../bundles/recipes-inc-2026-08-12.bundle 'refs/heads/*:refs/heads/*'
```

`git bundle verify` on an *incremental* bundle fails if the enclave repo lacks the prerequisite commits — a feature: it catches out-of-order or skipped transfers before they corrupt the mirror. Host the enclave-side mirrors on any internal git server (Gitea is the lightweight standard; Zarf's init package can even deploy one, §10). Same mechanism for your own config repo, agent code, and any git-URL Python dependencies from §6.4.

---

## 9. Model weights across the gap

### 9.1 Download with a pinned revision (staging)

A Hugging Face repo is a git repo; a model "version" is a **commit hash**. Pin it — a tag or `main` can move after your eval gate ran.

```bash
# Resolve and record the commit first (from the model page or API), then:
hf download nvidia/Llama-3.3-70B-Instruct-NVFP4 \
  --revision 3f5a9c1e...<full-commit-sha> \
  --local-dir /staging/models/nvidia/llama-3.3-70b-instruct-3f5a9c1-nvfp4

# Equivalent single-URI form:
hf download "hf://nvidia/Llama-3.3-70B-Instruct-NVFP4@3f5a9c1e..." --local-dir ...
```

Notes as of huggingface_hub v1.x (2026): the CLI is `hf` (the old `huggingface-cli` is deprecated); `--local-dir` gives you a plain directory (with a small `.cache/huggingface/` metadata folder inside — exclude it from your hash manifest); downloads are accelerated by **hf_xet** automatically — Xet is Hugging Face's chunk-based, content-addressed storage backend, and **it replaced `hf_transfer`** (the `HF_HUB_ENABLE_HF_TRANSFER=1` variable is now ignored; older docs and the SOP's shorthand "hf_transfer/Xet" predate this). Xet acceleration matters only on the connected side; inside the enclave every process runs with `HF_HUB_OFFLINE=1` and loads from local paths.

### 9.2 Hash manifest: generate at staging, verify in the enclave

```bash
# Staging: deterministic, sorted, recursive manifest
cd /staging/models/nvidia/llama-3.3-70b-instruct-3f5a9c1-nvfp4
find . -type f -not -path './.cache/*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum SHA256SUMS   # record this one line in the bundle manifest — one hash covers all

# Enclave, after transfer, BEFORE first load:
cd /models/nvidia/llama-3.3-70b-instruct-3f5a9c1-nvfp4
sha256sum -c SHA256SUMS --quiet && echo "WEIGHTS VERIFIED"
```

Two-level trick: the bundle `manifest.yaml` stores only the hash *of `SHA256SUMS` itself*; that file in turn covers every shard. Safetensors files also embed internal integrity metadata, but `sha256sum -c` is the check your auditor and your transfer runbook agree on. Budget real time: hashing ~400 GB at NVMe speeds is minutes, at USB/spinning-disk speeds much longer — this belongs in the transfer-window estimate.

### 9.3 Storage layout (SOP recap)

Weights live as **plain files** on the model store (NFS export or K8s PVC), laid out `/models/<org>/<name>-<revision>-<quant>/`, revision = short commit hash. vLLM serves from the path (`vllm serve /models/...`), never a hub ID. Weights are *not* baked into container images: images stay small and generic, weights version independently, and one copy on NFS serves N replicas with zero pull time.

### 9.4 The ORAS option: weights as OCI artifacts in Harbor

**ORAS** ("OCI Registry As Storage") pushes arbitrary files to any OCI registry with custom media types. Since Harbor v2.13 there is first-class AI-model support: a model-aware UI, and the **CNAI model spec** (CNCF's Cloud Native AI artifact conventions, also referred to as ModelPack) defining annotations like `org.cnai.model.name/.format/.param.size`. **KitOps** (`kit pack` / `kit push`, driven by a `Kitfile`) is the higher-level tool that produces spec-conformant model artifacts without hand-rolling ORAS flags.

```bash
oras push harbor.enclave.local/models/llama-3.3-70b:3f5a9c1-nvfp4 \
  --artifact-type application/vnd.cnai.model \
  --annotation "org.cnai.model.format=safetensors" \
  ./model-00001-of-00030.safetensors:application/vnd.cnai.model.layer.v1 \
  ... \
  ./config.json:application/vnd.cnai.model.layer.v1
```

**Tradeoffs vs plain NFS files, honestly (as of mid-2026):**

| Dimension | NFS files (SOP default) | ORAS/OCI in Harbor |
|---|---|---|
| Serving path | vLLM reads directly; zero extra copies | Pull + unpack to local disk first (init container) — extra time and 2× transient storage |
| Dedup | None (dir per revision) | Layer-level only: unchanged *files* dedup across versions; a changed shard re-stores whole (no sub-file chunking à la Xet) |
| Integrity/versioning | Your SHA256SUMS discipline | Digests, tags, immutability, cosign signing — built in and uniform with images |
| Governance | Filesystem perms, ad-hoc audit | Harbor RBAC, audit log, replication, retention |
| Rough edges | None new | No resumable uploads in ORAS; multi-hundred-GB layers stress timeouts, GC, and backups |

**Verdict consistent with the SOP:** keep NFS files as the serving default; the serving-path cost of pull-and-unpack is real and buys nothing at inference time. Adopt ORAS/Harbor for weights *as a catalog and distribution layer* when you have many models, many sites (replication!), or an auditor who wants one signed system of record for every artifact type — and even then, materialize to the model store before serving. This corner of the ecosystem is moving fast (Harbor model UI, CNAI/ModelPack, KServe's OCI-model support); re-verify tooling maturity before standardizing.

---

## 10. Zarf deep dive

### 10.1 What Zarf is

**Zarf** (OpenSSF/Defense Unicorns lineage; v0.83.x as of August 2026) is a declarative air-gap packaging tool for Kubernetes. One YAML (`zarf.yaml`) declares a workload's *complete* closure — container images, Helm charts, K8s manifests, arbitrary files — and `zarf package create` produces a single (signable) tarball. On the disconnected side, `zarf package deploy` installs it with **no external registry at all**, because Zarf brings its own.

### 10.2 How the embedded registry works (`zarf init`)

The clever part is bootstrap — you need a registry to deploy a registry. The **init package** (`zarf init`) runs four ordered components:

1. **zarf-injector** — a tiny static Rust binary that reassembles a registry image from chunks and serves it temporarily;
2. **zarf-seed-registry** — the Distribution `registry:3` image (~18 MB compressed) is split into ~1 MB chunks, injected into the cluster *as ConfigMaps* (under etcd's default size limit), reassembled, and run (NodePort by default);
3. **zarf-registry** — the long-lived internal registry (docker-registry Helm chart) is then pulled *from the seed* and becomes the cluster's permanent image source;
4. **zarf-agent** — a mutating admission webhook that **rewrites every pod's image references on the fly** to point at the internal registry (it also rewrites Flux/ArgoCD sources), so upstream charts and manifests work unmodified in the enclave.

Optional init components: `k3s` (bring your own lightweight Kubernetes) and `git-server` (Gitea, for GitOps repos).

### 10.3 Package anatomy and commands

```yaml
kind: ZarfPackageConfig
metadata: { name: monitoring-stack, version: 1.4.0 }
components:
  - name: prometheus
    required: true
    charts:
      - name: kube-prometheus-stack
        url: https://prometheus-community.github.io/helm-charts
        version: 62.3.0
        namespace: monitoring
    images:
      - quay.io/prometheus/prometheus:v2.54.0
      - docker.io/grafana/grafana:11.6.0
```

```bash
zarf package create .                       # staging: pulls everything, emits zarf-package-*.tar.zst
zarf package sign / zarf package inspect    # cosign-based signing; audit contents
zarf package deploy zarf-package-monitoring-stack-amd64-1.4.0.tar.zst   # enclave
zarf package publish ... oci://harbor.enclave.local/zarf                # optional: store packages in Harbor
```

### 10.4 When Zarf fits vs plain Harbor + Helm

| Situation | Better fit |
|---|---|
| SOP Tier 1/2 (Compose, no K8s) | Harbor + skopeo bundles — Zarf is K8s-only |
| Tier 3 you built yourself: Harbor, PKI, git already standing | Harbor + Helm + your bundle pipeline; Zarf's registry duplicates Harbor |
| Greenfield/edge disconnected K8s with nothing standing | Zarf shines — `zarf init` bootstraps registry + git from zero |
| Shipping one complex third-party K8s app across the gap intact | Zarf package, even alongside Harbor |
| Many small disconnected sites fed from one staging zone | Zarf packages (possibly published into central Harbor) |

Honest framing for our environment: with the SOP's compose-then-k8s trajectory and a standing Harbor, Zarf is not the day-1 tool. It becomes attractive at Tier 3 for *whole-stack* deliveries (llm-d + gateway + monitoring as one signed, versioned, testable unit) and for any satellite enclave without standing services. The image-reference-mutating agent is the feature you cannot cheaply replicate with scripts.

---

## 11. Bundle design: manifest, transfer, arrival

### 11.1 The manifest is the product

The SOP defines the bundle layout (`manifest.yaml`, `images/`, `models/`, `recipes/`, `evals/`, `signatures/`). Here is a complete worked `manifest.yaml` — every enclave action is driven by, and checked against, this file:

```yaml
schema: enclave-bundle/v1
bundle:
  id: bundle-2026-08-12-qwen4-72b
  created: "2026-08-12T09:30:00Z"
  staging_pipeline: stage-ci/run/4812        # provenance pointer (staging-side record)
  transfer: { method: encrypted-ssd, ticket: SEC-2026-0812-03 }

approvals:                                    # who signed off, on what evidence
  - { role: platform-eng, name: A. Steele,  date: 2026-08-12 }
  - { role: security,     name: R. Okafor,  date: 2026-08-12 }

images:                                       # every image: tag AND digest AND archive hash
  - name: harbor.enclave.local/infra/vllm-openai:v0.26.1
    digest: sha256:ab12cd34...
    archive: images/vllm-openai-v0.26.1.tar
    archive_sha256: 7f3e9a...
    signed_by: cosign-key-2026a               # key id -> enclave trust store
  - name: harbor.enclave.local/infra/agentic-api:0.9.2
    digest: sha256:ef56ab78...
    archive: images/agentic-api-0.9.2.tar
    archive_sha256: 22c1d0...
    signed_by: cosign-key-2026a

models:
  - path: models/qwen/qwen4-72b-instruct-8c1f2ab-nvfp4/
    hf_repo: Qwen/Qwen4-72B-Instruct-NVFP4    # informational; enclave never resolves it
    hf_revision: 8c1f2ab9...<full commit sha>
    quant: nvfp4
    manifest_file: SHA256SUMS
    manifest_sha256: 91b4e7...                # the one hash that covers all shards
    size_bytes: 41876322304

git:
  - bundle: recipes/recipes-inc-2026-08-12.bundle
    repo: vllm-project/recipes
    head: f00dbeef...
    sha256: 5d1c22...
  - bundle: recipes/config-repo-2026-08-12.bundle
    repo: internal/config
    head: 0ddba11...
    sha256: 83aa10...

evidence:                                     # evals are evidence, not promises (SOP §5)
  evals:
    - { file: evals/agentic-gate-qwen4-72b.json, sha256: 44f0c9..., result: PASS,
        suite: agentic-eval-suite@2.3, baseline: prod-agent-v3 }
  scans:
    - { file: evals/trivy-vllm-openai-v0.26.1.json, sha256: cc27b1...,
        db_date: "2026-08-11", critical: 0, high: 2, waivers: [CVE-2026-XXXXX] }
  sboms:
    - { file: evals/sbom-vllm-openai.cdx.json, sha256: 1e9f77... }

data_updates:                                 # standing-service refreshes ride every bundle
  trivy_db: { file: security/trivy-db.tar.gz, sha256: ab99e2..., db_built: "2026-08-11" }
  pypi:     { snapshot: "2026-08-01", delta: pypi/devpi-delta-2026-08-12.tar }
  os_repos: { snapshot: "2026-08-01", note: "no change this bundle" }

verification:
  bundle_sha256sums: SHA256SUMS               # top level: covers every file above
  detached_signature: signatures/SHA256SUMS.sig   # cosign sign-blob of that file
```

Design rules: digests everywhere (tags are labels, digests are identity); the top-level `SHA256SUMS` + one detached signature (`cosign sign-blob --key cosign.key SHA256SUMS`) authenticates the whole bundle with a single check; evidence travels *with* the release so the enclave record is self-contained; and the manifest is append-only history — it gets committed to the enclave config repo as the deployment record.

### 11.2 Transfer mechanics, practically

- **Removable media** (encrypted SSD/NVMe is the realistic choice at model scale — a 400 GB model on USB-2-class media is a day, on NVMe minutes to tens of minutes): dedicated, labeled, media pool; wiped and re-imaged between uses; never leaves the facility pair; chain-of-custody log per crossing (who, when, media serial, bundle id).
- **Data diode** (hardware-enforced one-way optical link): there is no return channel, so no TCP and no retransmit-on-error — diode transfer software uses forward error correction and typically sends redundant passes. Consequence for you: *expect occasional corruption and design for it* — the per-file hash manifest isn't paranoia, it's the diode's error detection; failed files get re-sent on the next pass.
- **CDS (cross-domain solution)** — an accredited guarded gateway that inspects and filters content between security domains: expect enforced file-type allowlists, archive-inspection limits, and maximum file sizes. Two practical accommodations: keep bundles in plain, inspectable formats (tar, not exotic compression), and be ready to **split large weight shards** (`split -b 20G`, with per-part hashes, reassemble and re-verify inside) if the CDS caps file size.
- Whatever the mechanism: the **security-office ticket and transfer window** are usually the schedule bottleneck (SOP Phase 0 pre-authorizes an expedited lane for model bundles). Book the window when the eval gate *starts*, not when it passes.

### 11.3 Verification-on-arrival runbook (enclave side)

1. Mount media read-only (or receive diode drop directory); record media serial + ticket in the transfer log.
2. `sha256sum -c SHA256SUMS` at bundle root — every file, no skips. Any failure: quarantine the bundle, request re-send; never "fix" a single file by hand.
3. `cosign verify-blob --key /etc/enclave/keys/cosign.pub --signature signatures/SHA256SUMS.sig SHA256SUMS` — authenticate the manifest itself.
4. Diff `manifest.yaml` against reality: every listed file present, no *extra* files present.
5. Images: `skopeo copy --all oci-archive:... docker://harbor.enclave.local/...` per manifest entry; then `skopeo inspect` and compare digests to the manifest; then `cosign verify --key ... --insecure-ignore-tlog` each digest in Harbor.
6. Weights: rsync into the model store under the manifest path; `sha256sum -c SHA256SUMS` *at the destination* (verifies the copy, not just the media); set read-only perms.
7. Git: `git bundle verify` each bundle against its mirror, then fetch.
8. Data updates: install Trivy DB, apply devpi/OS-repo deltas; kick a Harbor "scan all" against the fresh DB.
9. Commit `manifest.yaml` to the enclave config repo; sign off the arrival checklist (two-person rule if policy requires).
10. Only now does the deployment runbook (canary → eval gate → alias promotion, SOP §5) begin.

---

## 12. Ongoing operations

### 12.1 CVE triage without internet

The enclave learns about CVEs only when you carry the knowledge in. Cadence that works:

- **Every bundle** carries the freshest Trivy/grype DBs (they are small — tens to hundreds of MB) plus a staging-side scan report of everything currently deployed. Even a bundle with no new model should ship monthly for exactly this reason.
- **On arrival:** re-scan all of Harbor against the new DB (scheduled "scan all"); diff against last month; triage new CRITICAL/HIGH findings into: fix now (emergency train), fix next train, or waive with written justification and expiry. SBOMs (§5.4) answer "are we even affected?" in minutes.
- Reality check: most image CVEs in a non-internet-facing enclave are *not* emergencies — the air gap removes most exploitation paths. The discipline is knowing your exposure and choosing, on the record, rather than not knowing.

### 12.2 Upgrade trains

Run fixed, boring trains rather than ad-hoc updates — each is a bundle type with its own test gate:

| Train | Cadence | Contents / gate |
|---|---|---|
| Security data | monthly (with any bundle) | scanner DBs, scan reports / auto |
| Engine | monthly-ish (SOP Phase 0) | new vLLM image / canary + full tool-call parser re-test |
| Model | on model drop (SOP §5) | weights + recipe / agentic eval gate |
| Platform | quarterly | Harbor, gateway, monitoring, OS snapshot / staged, canary node first |
| Driver/firmware | as required | NVIDIA driver, fabric / DCGM diagnostics + node acceptance |

Never combine driver and engine changes in one train — when latency regresses you want one variable.

### 12.3 Rollback strategy

- **Models:** alias flip (SOP §5 Phase 3); previous model stays warm one week.
- **Images:** previous digests remain in Harbor (retention policy: keep the last N releases per repo *exempt from GC*; tag immutability guarantees they still mean the same bytes). Rollback = redeploy the prior manifest's digests — which is why manifests are kept forever in the config repo.
- **OS:** previous aptly/reposync snapshot remains published; repoint and reinstall the held packages.
- The universal precondition: **rollback must never need the internet.** If reverting requires re-downloading anything, retention was wrong.

### 12.4 Disaster recovery of the registry and model store

- **Harbor:** back up Postgres (dump/WAL) + blob storage (filesystem/S3 snapshot) + `harbor.yml`; quarterly restore rehearsal to scratch, proving a node can pull a known digest. Worst case, Harbor is *reconstructable from bundles* — every image that matters arrived as an archive; keep the last N bundle media/archives in an enclave vault as the recovery source of last resort.
- **Model store:** NFS-level snapshots/replication if the filer supports it; otherwise the bundle archives again are the backstop. After any restore, run `sha256sum -c` per model directory before returning it to service — a restored-but-corrupt shard fails in the worst possible way (a model that loads and is subtly wrong is rarer than a crash, but a truncated shard *will* crash at load; verify either way).
- **Gateway state DB (Responses API) and Prometheus data:** in scope for the same backup regime (SOP §6.3); listed here because registry-and-models DR plans habitually forget them.

---

## 13. End-to-end worked example: bundle a new model release

Scenario: `Qwen/Qwen4-72B-Instruct-NVFP4` dropped; staging eval gate passed; ship it. (Names illustrative; flags per your mirrored recipe.)

**Staging zone:**

```bash
# 1. Pin the revision and download
REV=8c1f2ab9d0e4...            # full commit sha from the model page
hf download Qwen/Qwen4-72B-Instruct-NVFP4 --revision $REV \
  --local-dir /staging/models/qwen/qwen4-72b-instruct-8c1f2ab-nvfp4

# 2. Hash the weights
cd /staging/models/qwen/qwen4-72b-instruct-8c1f2ab-nvfp4
find . -type f -not -path './.cache/*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS

# 3. Refresh the engine image if the recipe requires a newer vLLM; archive + record digest
skopeo copy --all docker://docker.io/vllm/vllm-openai:v0.26.1 \
  oci-archive:/staging/bundle/images/vllm-openai-v0.26.1.tar
skopeo inspect --format '{{.Digest}}' docker://docker.io/vllm/vllm-openai:v0.26.1

# 4. Sign the image (key-based, no transparency log)
cosign sign --key cosign.key --tlog-upload=false \
  harbor-staging.local/infra/vllm-openai@sha256:ab12cd34...

# 5. Scan + SBOM (evidence for the bundle)
trivy image --format json -o /staging/bundle/evals/trivy-vllm.json \
  harbor-staging.local/infra/vllm-openai:v0.26.1
syft harbor-staging.local/infra/vllm-openai:v0.26.1 -o cyclonedx-json \
  > /staging/bundle/evals/sbom-vllm.cdx.json

# 6. Run the agentic eval gate on staging GPUs (tool-call parser test included); save results
cp eval-results/agentic-gate-qwen4-72b.json /staging/bundle/evals/

# 7. Refresh git bundles and the Trivy DB
cd /staging/mirrors/recipes.git && git remote update && \
  git bundle create /staging/bundle/recipes/recipes-inc-$(date +%F).bundle last-shipped..main
oras pull ghcr.io/aquasecurity/trivy-db:2 && mv db.tar.gz /staging/bundle/security/trivy-db.tar.gz

# 8. Assemble: weights -> models/, write manifest.yaml (per §11.1), then seal
cd /staging/bundle
find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
cosign sign-blob --key cosign.key --output-signature signatures/SHA256SUMS.sig SHA256SUMS

# 9. Copy to transfer media; hand off under ticket SEC-2026-0812-03
```

**Enclave:**

```bash
# 10. Verify everything (the §11.3 runbook, steps 1–4)
sha256sum -c SHA256SUMS
cosign verify-blob --key /etc/enclave/keys/cosign.pub \
  --signature signatures/SHA256SUMS.sig SHA256SUMS

# 11. Load images into Harbor; verify digest + signature
skopeo copy --all oci-archive:images/vllm-openai-v0.26.1.tar \
  docker://harbor.enclave.local/infra/vllm-openai:v0.26.1 \
  --dest-creds 'robot$bundle-loader:...'
skopeo inspect --format '{{.Digest}}' docker://harbor.enclave.local/infra/vllm-openai:v0.26.1
cosign verify --key /etc/enclave/keys/cosign.pub --insecure-ignore-tlog \
  harbor.enclave.local/infra/vllm-openai@sha256:ab12cd34...

# 12. Weights to the model store; verify AT DESTINATION; lock read-only
rsync -a models/qwen/qwen4-72b-instruct-8c1f2ab-nvfp4/ \
  /models/qwen/qwen4-72b-instruct-8c1f2ab-nvfp4/
cd /models/qwen/qwen4-72b-instruct-8c1f2ab-nvfp4 && sha256sum -c SHA256SUMS --quiet
chmod -R a-w .

# 13. Git + security data
git -C /srv/git/recipes.git bundle verify /media/bundle/recipes/recipes-inc-2026-08-12.bundle && \
  git -C /srv/git/recipes.git fetch /media/bundle/recipes/recipes-inc-2026-08-12.bundle 'refs/heads/*:refs/heads/*'
# install trivy-db into the Harbor trivy-adapter cache; trigger "scan all"

# 14. Record: commit manifest.yaml to the enclave config repo; sign the arrival checklist

# 15. Deploy to the canary slice (recipe flags; HF_HUB_OFFLINE=1; local path; digest-pinned image)
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 vllm serve \
  /models/qwen/qwen4-72b-instruct-8c1f2ab-nvfp4 \
  --served-model-name candidate-qwen4-72b ...   # per mirrored recipe

# 16. Enclave eval gate + guidellm sweep -> then alias promotion per SOP §5 Phase 3
#     (agent-default -> candidate-qwen4-72b at 5% -> 25% -> 100%; old model warm for a week)
```

---

## Study questions

1. **Why is content-addressing (digests/hashes) the foundation of the whole pipeline rather than signing?**
   Answer: A digest pins *what the bytes are*, independent of any name or location; signing then binds an identity to that digest. Signatures without digest pinning would bless movable references (tags), which an attacker can repoint.

2. **You ran `skopeo copy` without `--all` on an x86 staging box. What breaks, and when?**
   Answer: Only the x86 manifest crossed the gap. Nothing breaks until a non-x86 node (or a tool expecting the full manifest list) tries to pull — then the pull fails or falls back unexpectedly. Always `--all` for gap crossings.

3. **Why is `docker-archive:` the wrong transport for bundles?**
   Answer: It's the legacy `docker save` format and cannot represent a multi-architecture manifest list; `oci-archive:` carries the full OCI index and is the standard.

4. **What two properties do Harbor robot accounts have that trip people up operationally?**
   Answer: Secrets are shown once and never stored (no recovery — recreate instead), and tokens expire after 30 days by default, which can silently break node pulls if you don't set expiry deliberately.

5. **Why doesn't keyless cosign work in an enclave, and what is the standard substitute?**
   Answer: Keyless needs live OIDC, Fulcio (CA), and Rekor (transparency log) at signing time, plus TUF root updates for verification — all internet services. Substitute: key-pair signing with `--tlog-upload=false`, verifying with `--insecure-ignore-tlog` against a distributed public key (or, for large orgs, a self-hosted private sigstore stack).

6. **How does vulnerability scanning stay useful with no CVE feed?**
   Answer: The vulnerability databases (trivy-db/grype DB) are themselves OCI artifacts pulled with ORAS on staging and hand-carried in every bundle; Harbor runs Trivy with `skip_update`/`offline_scan`, and a "scan all" after each DB install re-judges old images against new knowledge. Alert on DB age so staleness is visible.

7. **When would you choose devpi over bandersnatch, and what protects a production image build regardless?**
   Answer: devpi when you need a working `pip install` for the packages you actually use (caching index, modest storage); bandersnatch only when policy demands a static mirror (full PyPI is tens of TB — use allowlists). Production builds are protected independently by a wheelhouse plus `pip install --no-index --require-hashes` against a hash-pinned lockfile.

8. **What does `git bundle verify` actually check on an incremental bundle?**
   Answer: Bundle integrity and that the *prerequisite* commits (the `basis` before the `..`) already exist in the local repo — so skipped or out-of-order transfers fail loudly before corrupting the mirror.

9. **Why does the SOP keep model weights on NFS instead of in Harbor via ORAS, and what would change the answer?**
   Answer: vLLM serves straight from a filesystem path; OCI weights must be pulled and unpacked first (extra time, transient 2× storage), and OCI dedup is whole-file only, with no resumable uploads for huge layers. Many models/sites, replication needs, or a single-signed-system-of-record requirement tip the balance toward Harbor as *catalog*, still materializing files before serving.

10. **What is the one clever mechanism in `zarf init` that plain Harbor + Helm cannot replicate cheaply?**
    Answer: The mutating-webhook agent that rewrites all image (and Flux/ArgoCD) references at admission to the in-cluster registry — letting unmodified upstream charts run air-gapped. (The seed trick — injecting a registry image as 1 MB ConfigMap chunks to solve registry-needs-a-registry — is the other signature move.)

11. **A file fails `sha256sum -c` on arrival after a data-diode transfer. What do you do, and why is this expected?**
    Answer: Quarantine the bundle and request re-transmission of the failed files — never hand-patch. Diodes have no return channel, so no retransmit protocol; occasional corruption is a designed-for event and the hash manifest *is* the error detection.

12. **Why must rollback never require the internet, and what three retention policies guarantee that?**
    Answer: The enclave has no internet, so anything not retained locally is unrecoverable on the rollback timeline. Guarantees: Harbor keeps the last N release digests exempt from GC (with tag immutability), the previous OS repo snapshot stays published, and previous bundle archives/media are vaulted.

---

## Sources

Primary documentation and repos (fetched/verified August 2026):

- Harbor docs (v2.13/2.15): robot accounts, replication rules, proxy cache, tag immutability, GC — https://goharbor.io/docs/ ; releases — https://github.com/goharbor/harbor/releases
- VMware/Broadcom: "Using Harbor as an AI Model Registry" (CNAI spec, KitOps, Harbor v2.13+ model support), Mar 2026 — https://blogs.vmware.com/cloud-foundation/2026/03/03/using-harbor-as-an-ai-model-registry/ ; "Making Harbor Production-Ready", Dec 2025 — https://blogs.vmware.com/cloud-foundation/2025/12/02/making-harbor-production-ready-essential-considerations-for-deployment/
- skopeo README and skopeo-sync(1) — https://github.com/containers/skopeo ; https://github.com/containers/skopeo/blob/main/docs/skopeo-sync.1.md
- crane vs skopeo background — https://eng.d2iq.com/blog/a-tale-of-two-container-image-tools-skopeo-and-crane/
- cosign README (key signing, `--tlog-upload`, verify flags, KMS) — https://github.com/sigstore/cosign ; air-gap issue #3437 — https://github.com/sigstore/cosign/issues/3437 ; offline verification write-up — https://some-natalie.dev/blog/cosign-disconnected/
- Kyverno: verify-images / ImageValidatingPolicy (v1.15) — https://kyverno.io/docs/policy-types/image-validating-policy/
- Trivy: air-gap + self-hosting DBs (ORAS pulls, `--db-repository`) — https://trivy.dev/latest/docs/advanced/self-hosting/ ; https://trivy.dev/latest/docs/advanced/air-gap/
- syft (SBOM) — https://github.com/anchore/syft ; grype — https://github.com/anchore/grype
- bandersnatch — https://github.com/pypa/bandersnatch ; PyPI mirror guidance — https://packaging.python.org/en/latest/guides/index-mirrors-and-caches/
- devpi (server 6.x, active 2026) — https://devpi.net/docs/ ; https://github.com/devpi/devpi
- uv offline / air-gap issues — https://github.com/astral-sh/uv/issues/13587 ; https://github.com/astral-sh/uv/issues/11746
- git-bundle — https://git-scm.com/docs/git-bundle
- Hugging Face `hf` CLI guide (hf download, `--revision`, hf:// URIs, local-dir) — https://huggingface.co/docs/huggingface_hub/main/en/guides/cli ; huggingface_hub v1.0 migration (hf_transfer removal, Xet default) — https://huggingface.co/docs/huggingface_hub/en/concepts/migration
- ORAS push/pull (CLI v1.3) — https://oras.land/docs/how_to_guides/pushing_and_pulling/
- Zarf docs (v0.83): init package internals — https://docs.zarf.dev/ref/init-package/ ; examples/commands — https://docs.zarf.dev/ref/examples/
- Quay mirror-registry — https://github.com/quay/mirror-registry
- zot registry — https://zotregistry.dev/ ; registry comparison — https://distr.sh/blog/container-image-registry-comparison/
- NVIDIA DGX OS air-gapped installations (OS package pattern) — https://docs.nvidia.com/dgx/dgx-os-7-user-guide/appendix_e_air_gapped_installations.html
- NVIDIA NIM air-gap deployment pattern (SOP-cited, transferable practice) — https://docs.nvidia.com/nim/large-language-models/2.0.0/deployment/air-gap-deployment.html
