# Git workflow

## Prepare the branch

Before editing, inspect the working tree and current branch. Preserve unrelated
changes. For a new task, fetch `origin` and create a descriptive task branch from
`origin/main`; use an isolated worktree when needed. Continue an existing task on
its unmerged task branch. Make changes on the task branch rather than `main`.

When a GitHub issue exists, read it using the issue-tracker instructions and
include its number in the branch name. GitHub issue numbers are this project's
task identifiers; a JIT key is not required.

## Implement and verify

Run checks appropriate to the completed changes. For Go or lint-tooling changes,
run `make lint` and `go test ./...`; run `go test -race ./...` for concurrency
changes. For documentation-only changes, verify the content, referenced paths,
and `git diff --check`; the required CI checks still run on the PR.

Obtain independent agent review for substantial behavior, contract, security, or
concurrency changes. Address findings and recheck affected behavior after fixes.
This development review does not require a GitHub approving review from another
maintainer.

## Commit and open the PR

Stage only files belonging to the task and inspect the staged diff before
committing. Push the task branch and use `gh` to open a PR targeting `main`.
Describe the final change, checks performed, and remaining limitations. Link the
related issue; use a closing reference only when the PR resolves its scope.

## Verify CI and deliver

Wait for the PR's required GitHub checks on its latest head commit. Fix failures
and repeat affected local checks and review before pushing updates. Report
blocked or unrun checks explicitly rather than treating a local pass as CI
success. Read the current required checks from GitHub instead of assuming a
fixed list.

When merging is within the requested scope, verify that the PR is mergeable and
all required checks pass, then merge through GitHub against the verified head
commit. Confirm the merge and related issue state. Otherwise, return the PR URL
and its check status at the requested delivery point.

Keep repository protection settings unchanged during task delivery. Resolve the
cause of a blocked merge or report the limitation; do not bypass checks, force
push to `main`, or weaken protection to complete a task. Changes to protection
settings are a separate task when requested by the user.
