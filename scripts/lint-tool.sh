#!/bin/sh
# Install a verified official release, or check the repository-local binary.
set -eu

action=$1
version=$2
binary=$3

matches_version() {
    [ -x "$binary" ] || return 1
    installed_version=$("$binary" version --short 2>/dev/null) || return 1
    [ "$installed_version" = "$version" ]
}

case "$action" in
    check)
        if matches_version; then
            exit 0
        fi
        printf 'golangci-lint %s is required at %s; run make tools\n' "$version" "$binary" >&2
        exit 1
        ;;
    install) ;;
    *) printf 'unknown action: %s\n' "$action" >&2; exit 1 ;;
esac

if matches_version; then
    printf 'golangci-lint %s is already installed\n' "$version"
    exit 0
fi

case "$(uname -s)" in
    Darwin) os=darwin ;;
    Linux) os=linux ;;
    *) printf 'unsupported OS; use Linux, macOS, or WSL\n' >&2; exit 1 ;;
esac
case "$(uname -m)" in
    arm64|aarch64) arch=arm64 ;;
    x86_64|amd64) arch=amd64 ;;
    *) printf 'unsupported architecture; use amd64 or arm64\n' >&2; exit 1 ;;
esac
for prerequisite in curl tar mktemp awk; do
    if ! command -v "$prerequisite" >/dev/null 2>&1; then
        printf 'missing prerequisite: %s\n' "$prerequisite" >&2
        exit 1
    fi
done
if command -v sha256sum >/dev/null 2>&1; then
    checksum_command=sha256sum
elif command -v shasum >/dev/null 2>&1; then
    checksum_command=shasum
else
    printf 'missing prerequisite: sha256sum or shasum\n' >&2
    exit 1
fi

temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT HUP INT TERM
archive="golangci-lint-${version}-${os}-${arch}.tar.gz"
release_url="https://github.com/golangci/golangci-lint/releases/download/v${version}"
curl --fail --location --silent --show-error --connect-timeout 15 --max-time 300 \
    "$release_url/$archive" --output "$temporary_dir/$archive"
curl --fail --location --silent --show-error --connect-timeout 15 --max-time 60 \
    "$release_url/golangci-lint-${version}-checksums.txt" --output "$temporary_dir/checksums.txt"
expected=$(awk -v archive="$archive" '$2 == archive {print $1}' "$temporary_dir/checksums.txt")
if [ "${#expected}" -ne 64 ]; then
    printf 'missing or invalid release checksum for %s\n' "$archive" >&2
    exit 1
fi
if [ "$checksum_command" = sha256sum ]; then
    actual=$(sha256sum "$temporary_dir/$archive" | awk '{print $1}')
else
    actual=$(shasum -a 256 "$temporary_dir/$archive" | awk '{print $1}')
fi
if [ "$actual" != "$expected" ]; then
    printf 'release checksum mismatch for %s\n' "$archive" >&2
    exit 1
fi
tar -xzf "$temporary_dir/$archive" -C "$temporary_dir" "golangci-lint-${version}-${os}-${arch}/golangci-lint"
candidate="$temporary_dir/golangci-lint-${version}-${os}-${arch}/golangci-lint"
if [ "$("$candidate" version --short)" != "$version" ]; then
    printf 'downloaded binary version does not match %s\n' "$version" >&2
    exit 1
fi
mkdir -p "$(dirname "$binary")"
cp "$candidate" "$binary"
printf 'installed golangci-lint %s at %s\n' "$version" "$binary"
