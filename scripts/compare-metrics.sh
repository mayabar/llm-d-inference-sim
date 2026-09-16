#!/usr/bin/env bash
# Scrape /metrics and /metrics_new and print the differences.
#
# Usage:
#   ./compare-metrics.sh                  # against http://localhost:8000
#   ./compare-metrics.sh http://host:port
#   HOST=http://host:port ./compare-metrics.sh
#
# What it does:
#   - Fetches both endpoints.
#   - Drops HELP/TYPE comment lines.
#   - Sorts the sample lines.
#   - Prints a unified diff between the two, plus a per-metric-family
#     summary of which families appear on one surface only or with
#     differing samples.
#
# Notes:
#   - `vllm:lora_requests_info` samples carry a Unix timestamp as their
#     VALUE (set to time.Now().Unix() by each pipeline independently).
#     Those lines will nearly always diff by 0-1 seconds. The summary
#     filters them out; the raw diff still shows them.

set -euo pipefail

HOST="${1:-${HOST:-http://localhost:8000}}"
LEGACY_URL="${HOST%/}/metrics"
NEW_URL="${HOST%/}/metrics_new"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

LEGACY_RAW="$TMPDIR/metrics.raw"
NEW_RAW="$TMPDIR/metrics_new.raw"
LEGACY_NORM="$TMPDIR/metrics.norm"
NEW_NORM="$TMPDIR/metrics_new.norm"

fetch() {
    local url="$1" out="$2"
    local code
    code=$(curl -sS -o "$out" -w '%{http_code}' "$url" || echo "000")
    if [[ "$code" != "200" ]]; then
        echo "error: GET $url returned HTTP $code" >&2
        exit 1
    fi
}

# Keep sample lines only; drop HELP/TYPE comments and blank lines.
# Sort so ordering differences don't show up as diffs.
normalize() {
    grep -Ev '^(#|$)' "$1" | LC_ALL=C sort > "$2"
}

echo "Scraping:"
echo "  legacy : $LEGACY_URL"
echo "  new    : $NEW_URL"
echo

fetch "$LEGACY_URL" "$LEGACY_RAW"
fetch "$NEW_URL"    "$NEW_RAW"

normalize "$LEGACY_RAW" "$LEGACY_NORM"
normalize "$NEW_RAW"    "$NEW_NORM"

# --- Unified diff ----------------------------------------------------------
echo "=== unified diff (< legacy /metrics, > new /metrics_new) ==="
if diff -u "$LEGACY_NORM" "$NEW_NORM"; then
    echo "(no differences)"
fi
echo

# --- Family-level summary --------------------------------------------------
# Extract the metric family name from each sample line: everything before
# the first `{` or ` ` (whichever comes first).
family() { sed -E 's/[ {].*$//' "$1" | LC_ALL=C sort -u; }

LEGACY_FAMILIES="$TMPDIR/legacy.families"
NEW_FAMILIES="$TMPDIR/new.families"
family "$LEGACY_NORM" > "$LEGACY_FAMILIES"
family "$NEW_NORM"    > "$NEW_FAMILIES"

echo "=== family-level summary ==="
echo
echo "Families only in /metrics (legacy):"
comm -23 "$LEGACY_FAMILIES" "$NEW_FAMILIES" | sed 's/^/  /' || true
echo
echo "Families only in /metrics_new:"
comm -13 "$LEGACY_FAMILIES" "$NEW_FAMILIES" | sed 's/^/  /' || true
echo

# For families present on both sides, report those whose sample sets differ.
# Skip lora_requests_info: its value is a wall-clock timestamp set
# independently by each pipeline, so a 0-1s drift is expected noise.
echo "Families present on both sides but with differing samples:"
comm -12 "$LEGACY_FAMILIES" "$NEW_FAMILIES" | while read -r fam; do
    [[ "$fam" == "vllm:lora_requests_info" ]] && continue
    l="$TMPDIR/l.$fam"; n="$TMPDIR/n.$fam"
    grep -E "^${fam}([ {])" "$LEGACY_NORM" > "$l" || true
    grep -E "^${fam}([ {])" "$NEW_NORM"    > "$n" || true
    if ! diff -q "$l" "$n" >/dev/null; then
        echo "  $fam"
        diff -u "$l" "$n" | sed -n 's/^\([<>]\) /    \1 /p'
    fi
done
