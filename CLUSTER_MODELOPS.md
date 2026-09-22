# Cluster Model Ops Source Workflow

This is the LivingAgile/exo-contributions working fork, not an upstream release.
Root AGENTS.md records the owner-authorized development and publication scope.

- `main` tracks upstream; do not add fork-only policy or runtime changes there.
- `cluster-modelops/baseline` starts at upstream commit
  `21a54c5ea0230a3bec1e1a786d200126c7e34ec6` with one policy-only child.
- `cluster-modelops/integration` preserves the selected runtime work descended from
  `09cc87ff6af892b82be0b7296ed4a302a864c886`.
- Use `cluster-modelops/feature/<topic>` for scoped changes. Review the outgoing
  diff and commit history, run applicable checks, and promote by fast-forward from
  the observed integration tip. A concurrent advance requires reconciliation, not
  force-push. Verify the remote SHA after publication.

The preserved build pins LivingAgile/mlx-lm at
`7c4a3669d87d0cc9bdc63433030932850221f40e` in pyproject.toml and uv.lock.
Keep those identities consistent. A future dependency change needs its own validation;
branch publication does not change the installed cluster runtime.

Before adoption, record exact EXO/MLX-LM SHAs, immutable build identity, qualified
model coverage and the prior runtime pair. Recover by selecting that retained pair
through the authorized deployment procedure, not by rewriting published branches.
The historical source branch `work/plan-0041/integration` remains preserved. Its
existence is not a claim that it is the current live runtime or includes later models.

For upstream work, start `upstream/<topic>` from a recorded upstream commit and
port only selected source, tests and public reproduction material. Inspect every
outgoing commit as well as the final diff. Neither this file nor the fork-only
AGENTS.md scope belongs in upstream history. A tip revert is insufficient. Preserve
attribution and follow the target's current submission rules and separate operator
publication/readiness decisions. Never publish secrets or private experimental ancestry.