# Required checks for main

The active [Protect main ruleset](https://github.com/atankeev/kmeans/rules/25317516)
(rule ID `25317516`) targets exactly `refs/heads/main`. Changes reach `main`
through a pull request. Force pushes and branch deletion are blocked. Zero
approving reviews are required for this single-maintainer repository; code-owner,
last-push, and unattributed-change approvals are disabled. Updating a PR branch
to the latest `main` before merging is not required.

No bypass actors are configured, including administrators. Repository
administrators can edit the ruleset, but have no merge bypass. Do not weaken
protection to deliver a task.

## Required checks

The required status contexts and their GitHub App integration IDs are:

| Exact check context | Publisher | Integration ID |
| --- | --- | --- |
| `lint` | GitHub Actions | 15368 |
| `build` | GitHub Actions | 15368 |
| `Linux x64 / Go minimum` | GitHub Actions | 15368 |
| `Linux x64 / Go stable` | GitHub Actions | 15368 |
| `Linux x64 / Go stable race and coverage` | GitHub Actions | 15368 |
| `Windows x64 / Go stable smoke` | GitHub Actions | 15368 |
| `macOS ARM64 / Go stable smoke` | GitHub Actions | 15368 |
| `govulncheck` | GitHub Actions | 15368 |
| `codecov/patch` | Codecov | 254 |

The [Go workflow](../.github/workflows/go.yml) and
[Vulnerability scan workflow](../.github/workflows/vulnerability.yml) run for
PRs targeting `main`, without path filters or author exclusions. Documentation
and Dependabot changes must emit the same required checks. The `build` check
also requires every compatibility, smoke, and race/coverage job to succeed.
Codecov must publish `codecov/patch`; a successful coverage upload alone cannot
satisfy that requirement. See the [coverage policy](../README.md#coverage-reporting-and-patch-policy)
for the 85% patch target, upload setup, and failure investigation.

Scheduled benchmarks and extended fuzz mutation are not required PR checks.
Ordinary tests execute the committed fuzz seed corpus.

## Verify and maintain

Inspect the live rules and current PR checks with read-only commands from a
checkout authenticated with `gh`:

```bash
gh api repos/atankeev/kmeans/rulesets/25317516
gh api repos/atankeev/kmeans/rules/branches/main
gh pr checks PR_NUMBER
gh pr view PR_NUMBER --json headRefOid,mergeable,mergeStateStatus,statusCheckRollup
```

Confirm exact check names and publisher identities on the PR's current head
before changing the ruleset. To inspect the check runs for that commit:

```bash
gh api repos/atankeev/kmeans/commits/HEAD_SHA/check-runs \
  --jq '.check_runs[] | {name, conclusion, app_id: .app.id}'
```

Replace `PR_NUMBER` and `HEAD_SHA` with the PR number and its current head SHA.
Workflow titles are not required status contexts; matrix-generated job names
must match the table exactly. When renaming a required job, coordinate the
workflow change and ruleset update after confirming the new context and App on
a real PR. Preserve the other required checks, pull-request requirement, and
bypass restrictions; avoid leaving a retired context pending indefinitely.

If a check is missing, inspect the workflow trigger, path filters, job
conditions, PR head SHA, and any pending workflow approval. Inspect Codecov
processing and App access when its upload succeeded but its check is absent.
Resolve the cause and rerun the affected workflow, then verify all required
checks on the current head. Do not remove requirements or substitute a manually
reported success to unblock a merge.

## Verification evidence

[Documentation-only PR #60](https://github.com/atankeev/kmeans/pull/60) and
[Dependabot PR #62](https://github.com/atankeev/kmeans/pull/62) emitted all nine
contexts successfully, including `codecov/patch`, and merged. This establishes
that these PR types emit the checks; the expanded required-check policy is
tracked in [issue #34](https://github.com/atankeev/kmeans/issues/34), including
verification of merge eligibility and blocking under the final ruleset.

Earlier closed probes [#58](https://github.com/atankeev/kmeans/pull/58) and
[#59](https://github.com/atankeev/kmeans/pull/59) verified that the required
Codecov patch check permits 100% patch coverage and blocks 60% patch coverage.
Neither dummy change was merged.
