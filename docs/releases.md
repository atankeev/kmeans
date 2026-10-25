# Releases

Merging a reviewed Release Please PR authorizes a release. Ordinary changes to
`main`, including installation of this automation, do not publish a version.
The first intended version is **v1.0.0**. The library is distributed as a Go
module; releases do not include executable binaries.

## Version and compatibility policy

The documented exported API, input validation, errors, ownership, and numerical
contracts are the stable v1 interface. Compatible fixes use patch versions
(`v1.0.1`); compatible additions use minor versions (`v1.1.0`). An incompatible
API change requires a deliberate v2 migration, including the `/v2` module and
import path. A breaking-change commit alone cannot perform that migration.

Seeded reproducibility applies to identical ordered data and configuration in
the same library version and execution environment. A fixed seed does not promise
identical results across library versions, Go versions, architectures, or other
environment changes. See [the README](../README.md#reproducibility) for the
complete guarantee.

Tags use plain `vMAJOR.MINOR.PATCH` names. A published tag is the Go module release
point and is immutable: never move, delete, or reuse it for different source.
Code defects require a new version, even if indexing has not completed.

## One-time owner setup

Create a fine-grained personal access token restricted to this repository, with
**Contents: read and write** and **Pull requests: read and write**. Release Please
also manages PR labels through the Issues API; grant **Issues: read and write**
for that operation. Set a sensible expiration and arrange renewal. Store the
value only as the Actions repository secret **`RELEASE_PLEASE_TOKEN`** under
**Settings → Secrets and variables → Actions**. Never put it in a file or issue.
A suitably scoped GitHub App installation token is an alternative, but generating
that token must be wired into the maintenance workflow before using it.

The external token lets Release Please PR events trigger ordinary PR checks;
events created with `GITHUB_TOKEN` generally do not start new workflow runs.
The maintenance workflow fails visibly if the secret is absent. Repository and
organization policy must permit that token to create PRs and update its release
branch. Do not bypass protection or reduce required checks to accommodate it.

Publication uses the job's `GITHUB_TOKEN` with Contents and Pull requests write
permissions plus Actions read access. Validation has read access only. Tag events
from that token are not relied on to start validation or publication workflows.
Keep Codecov configured as described in the README: the release PR must still
pass the existing PR `codecov/patch` protection check.

## Prepare and approve a release

1. Merge ordinary reviewed changes to `main` using Conventional Commit titles in
   the resulting history. Prefer squash merges with titles such as
   `fix: reject an invalid configuration` or `feat: add a compatible option`.
   `fix` proposes a patch; `feat` proposes a minor release. Review breaking-change
   markers (`feat!:` or `BREAKING CHANGE:`) especially carefully. Historical
   commits need not be rewritten.
2. **Actions → Release PR** maintains a PR named
   `chore(main): release 1.0.0` (with the proposed version). You can select
   **Run workflow → main** to retry maintenance. It prepares `CHANGELOG.md` and
   `.release-please-manifest.json`; it never publishes a release. With no
   releasable commit, Release Please may produce no PR. For the first release,
   merge a reviewed releasable change; `initial-version` sets its proposal to
   1.0.0. If a later proposal needs an intentional version override, use
   [Release Please's documented Release-As footer](https://github.com/googleapis/release-please#how-do-i-change-the-version-number).
3. Review the release PR's version, generated notes, complete diff, and required
   checks. Edit or correct the source commits/notes before approval as necessary.
   Keep the generated release PR title, branch, body structure, manifest, and
   `autorelease: pending` label intact: publication validates them. Merge only
   when ready to publish that version. Merging the setup PR is not this approval.
4. **Actions → Publish release** waits for both **Go** and **Vulnerability scan**
   push workflows on the release PR's exact merge SHA. The Go workflow includes
   lint, minimum/stable tests, platform smoke tests, and race/coverage upload.
   These checks differ from the PR-only Codecov patch evaluation, which remains
   required before merging the release PR. A newer successful `main` run or PR
   head result cannot substitute for this SHA's push results.
5. After successful validation, publication rechecks the pending candidate and
   CI, creates an annotated version tag at that SHA, then asks Release Please to
   create the GitHub Release with the reviewed notes and finish its labels.
   Review the resulting tag SHA, release notes, and Actions logs.

PR maintenance uses the repository's manifest configuration. Publication uses
fixed single-module Go settings from its workflow instead, so later changes to
the manifest configuration on `main` cannot change the candidate's tag format
or introduce additional packages during publication.

Maintenance and publication share a serialized concurrency group. While a merged
release PR remains pending, maintenance does not prepare another one. The gate
rejects multiple merged pending release PRs, malformed candidates, changed
versions without release metadata, and a SHA outside `main`. Missing or running
CI leaves publication pending; later workflow completion retries the gate.
Each push SHA has its own CI concurrency group so later merges cannot cancel a
release candidate's checks. Required PR job names and protection stay unchanged.

## Recovery

For failed CI, fix the cause and rerun the failed **Go** or **Vulnerability scan**
run on the candidate SHA. Successful completion retries publication automatically.
For API failures, absent completion events, or a pending run that needs another
look, use **Actions → Publish release → Run workflow → main**. This command takes
no version or SHA input: it always rediscovers and revalidates the sole merged
pending release PR. An ordinary merge with no candidate does nothing.

If an annotated tag already exists at the intended SHA, the workflow preserves
it and resumes GitHub Release creation. An existing lightweight tag is rejected
and preserved; investigate and use a new version rather than replacing it. A tag
at another SHA is a hard error; investigate without retagging. A failure after
tag creation means the module version already
exists at the Git level. Resume the same version and SHA, never remove the tag.
If GitHub Release creation succeeded but label updates failed, verify that the
existing release has the intended tag, that the annotated tag peels to the
verified merge SHA, and that its notes match the reviewed release PR. If
`autorelease: pending` remains, retry publication: Release Please can reconcile
labels and still fail that run with a duplicate-release error. Verify the final
release and labels instead of interpreting that error as a new publication.
If pending was already removed but `autorelease: tagged` is absent, the workflow
has no pending candidate to rediscover. After the same release/SHA/notes checks,
manually apply `autorelease: tagged` to the original merged release PR. Do not
restore pending to an already completed release, retag, or create a replacement
release. Do not create a new release PR until recovery finishes.

Do not manually add pending labels to ordinary PRs or manufacture release PRs.
If more than one pending merged PR is discovered, investigate their tags/releases
and reconcile their state before retrying. Automatic publication intentionally
refuses to choose between them. Never skip the candidate checks to clear a queue.

## Verify the published module and documentation

After each release, verify its actual version (substitute the released tag):

```bash
GOPROXY=https://proxy.golang.org go mod download -json github.com/atankeev/kmeans@v1.0.0
```

Use a temporary directory outside this checkout for any consumer smoke test.
Confirm the download succeeded and the reported module version and origin hash
match the published tag. Open
`https://pkg.go.dev/github.com/atankeev/kmeans@v1.0.0` and verify that the actual
API documentation renders, including `Cluster`, `Result`, and options. A landing
page alone does not establish that API indexing completed. If necessary, request
indexing through pkg.go.dev's UI and retry later. Delayed indexing remains pending
verification and is never a reason to move or recreate a tag.

Record the verified version, module download result, and exact rendered API URL
in [issue #24](https://github.com/atankeev/kmeans/issues/24), retaining that issue's
other acceptance criteria. Installing these workflows does not publish the first
release or perform indexing requests.

## Sources

- [Release Please action: credentials and inputs](https://github.com/googleapis/release-please-action)
- [Manifest releaser configuration](https://github.com/googleapis/release-please/blob/main/docs/manifest-releaser.md)
- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
- [GitHub token event behavior](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-when-your-workflow-runs/triggering-a-workflow#triggering-a-workflow-from-a-workflow)
