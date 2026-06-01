# Template Progress

| # | Template | Status | Coverage | Local services | Validation | Remote CI | Notes |
| -: | --- | --- | --- | --- | --- | --- | --- |
| 1 | async-webhook-ledger | golden | inbound event idempotency | Postgres, LocalStack SQS, WireMock, MailHog | passed in prior repo | passed in prior repo | First backend reliability template |
| 2 | rag-retrieval-quality-lab | golden | AI retrieval quality | Qdrant, Postgres, MinIO, fake embedding API | passed in prior repo | passed in prior repo | AI backend retrieval template |
| 3 | gpu-fault-correlation-drain-scheduler | golden | ML infra cluster operations | telemetry/control-plane simulator | passed in prior repo | passed in prior repo | Hard infra flagship |
| 4 | streaming-chat-budget-tools | golden | full-stack AI streaming | fake model/tool simulator | passed in prior repo | passed in prior repo | UI state gate included |
| 5 | agent-trace-evaluator | golden | agent trajectory evals | fake trace API, Jaeger | passed in prior repo | passed in prior repo | AI evals flagship |
| 6 | payment-recovery-state-machine | golden | outbound payment recovery | Postgres, WireMock, LocalStack SQS | passed in prior repo | passed in prior repo | Distinct from inbound webhooks |
| 7 | workflow-recovery-reconciler | golden | durable workflow recovery | Postgres, WireMock, LocalStack SQS | passed in prior repo | passed in prior repo | Platform recovery template |
| 8 | session-lifecycle-reconciler | golden | SaaS auth/session drift | Keycloak-like simulator, Postgres, Redis | passed in prior repo | passed in prior repo | Auth/session template |
| 9 | tenant-permission-audit | golden | enterprise SaaS authorization | Postgres, OPA | passed in prior repo | passed in prior repo | Security/authorization template |
| 10 | distributed-training-resume | golden | ML infra checkpoint recovery | MinIO, Postgres, CPU PyTorch | passed | passed | Remote CI and Docker-backed expected failure passed |
## Machine-Readable Contract Migration

- machine_readable_manifest: present
- root_make_aliases: present
- render_context_support: present
- check_render_contract: present
- scan_safety_uses_manifest: present
- remote_ci_manifest_validation: passed
- personalization_contract: present
- validate_personalization: present
- published_repo_contract_checker: present
