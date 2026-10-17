GOLANGCI_LINT_VERSION := 2.15.0
LINT_GO_VERSION := 1.27.2
GOLANGCI_LINT := $(CURDIR)/.bin/golangci-lint
GOVULNCHECK := $(CURDIR)/.bin/govulncheck

.PHONY: tools lint fmt check-lint-tool print-lint-go-version vuln-tools vuln

tools:
	@sh scripts/lint-tool.sh install $(GOLANGCI_LINT_VERSION) "$(GOLANGCI_LINT)"

check-lint-tool:
	@sh scripts/lint-tool.sh check $(GOLANGCI_LINT_VERSION) "$(GOLANGCI_LINT)"

lint: check-lint-tool
	GOTOOLCHAIN=go$(LINT_GO_VERSION) "$(GOLANGCI_LINT)" run ./...
	GOTOOLCHAIN=go$(LINT_GO_VERSION) "$(GOLANGCI_LINT)" fmt --diff ./...

fmt: check-lint-tool
	GOTOOLCHAIN=go$(LINT_GO_VERSION) "$(GOLANGCI_LINT)" fmt ./...

print-lint-go-version:
	@printf '%s\n' '$(LINT_GO_VERSION)'

vuln-tools:
	GOBIN="$(CURDIR)/.bin" go -C tools/govulncheck install -mod=readonly golang.org/x/vuln/cmd/govulncheck

vuln: vuln-tools
	"$(GOVULNCHECK)" ./...
