# Go style

## Adopted standard

Follow the [Uber Go Style Guide at commit
`1d60a91aa5e87d443002e23c21903c49489dbde5`](https://github.com/uber-go/guide/blob/1d60a91aa5e87d443002e23c21903c49489dbde5/style.md)
for Go production code and tests. Read the full pinned guide before changing
Go code or performing the Standards axis of a Go review. This document adds
project context and review prompts; it does not replace the guide.

The [upstream master guide](https://github.com/uber-go/guide/blob/master/style.md)
is a reference for evaluating updates. Adopt a new revision deliberately in a
separate reviewed change, accounting for changed advice and automated coverage.

Explicit project requirements and accepted [ADRs](../adr/) take precedence over
general guide advice. Preserve public contracts and numerical behavior during
style cleanup, including the arithmetic and run-selection policy in
[ADR-0001](../adr/0001-finite-arithmetic-policy.md) and the option and convergence
contracts in [ADR-0002](../adr/0002-in-place-api-redesign.md). Use language features
supported by `go.mod`; guide examples do not authorize changing Go, dependencies,
or tool versions. Prefer typed standard-library atomics when supported rather
than adding the guide's suggested atomic dependency.

## Workflow and review

Before handoff, review the complete task diff against the pinned guide and the
checklist below, including tests and changes made after an earlier review.
Explain justified deviations and their reasons in the PR. A passing linter is
evidence for its checks, not full guide compliance.

Use this guidance as project standards in the existing Standards review axis.
Follow [the git workflow](git-workflow.md) for required local checks, independent
review, and latest-head CI verification.
Style-only work does not add a maintainer-approval requirement beyond that
workflow.

Check each applicable area in the complete diff:

- **Names:** use clear package, function, receiver, and variable names; check
  aliases, built-in shadowing, and the guide's global and error naming rules.
- **Keyed literals:** name struct fields, retaining the guide's small test-table
  exception; omit zero fields only when their names add no useful context.
- **Grouping:** group related declarations, follow the guide's adjacent-local
  declaration rule, group methods by receiver, and keep rough call order.
- **Nesting:** handle errors and special cases early; keep the main path readable
  with early returns and loop continuations.
- **Errors:** handle each error once, add useful concise context, and preserve
  documented sentinel/type matching and nil-result guarantees.
- **Interfaces:** pass interfaces as values, verify required implementations,
  and keep receiver and embedding choices consistent with public contracts.
- **Concurrency:** account for ownership of shared slices/maps, mutex copying,
  channel capacity, and a stop and wait path for every goroutine.
- **Wrapping:** wrap long expressions and calls for readability. The guide's
  99-character recommendation is a soft target, not a hard limit.

## Automated coverage

The current [configuration](../../.golangci.yml) provides partial coverage:

| Check | Coverage and limits |
| --- | --- |
| `goimports` | Formatting and standard-library/other import grouping. |
| `errcheck` | Unchecked returned errors and unchecked type assertions. |
| `revive` | Only `error-naming`, `error-strings`, `indent-error-flow`, and `redundant-import-alias`; other default rules remain disabled. |
| `predeclared` | Shadowing of built-in identifiers. |
| `nolintlint` | Suppressions must name a specific linter, explain the reason, and suppress an actual finding. |
| `errorlint` | Error comparisons and type assertions that bypass wrapping; `%w` wrapping is not required. |
| `govet` | Suspicious constructs, including some unkeyed literals; not all local struct literals. |
| `staticcheck` | Correctness and simplification; not every style convention. |
| `ineffassign`, `unused` | Ineffective assignments and unused code. |
| `paralleltest`, `testifylint` | Project test parallelism and assertion-use policy. |

The added checks target unchecked assertions, recognizable error names,
lowercase error messages without trailing punctuation, flat error paths,
unambiguous built-in names, meaningful suppressions, wrapped-error matching,
and imports without redundant aliases. They were verified with representative
violating and valid examples; they do not establish full guide compliance.

The checklist and full-guide review cover judgment and gaps left by these tools.
Review control-flow suggestions before applying them: rewrites can change
variable scope. Preserve intentional `%v` abstraction permitted by the guide;
blanket error-wrapping rules are not enabled. Line wrapping remains a review
judgment without a hard 99-character check. Before enabling further checks,
evaluate overlap and false positives, demonstrate representative failing and
passing examples, explain scope and exclusions, and pass the required project
checks.
