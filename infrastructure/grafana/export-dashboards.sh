#!/bin/bash
# Export all dashboards from the running Grafana instance to the repo.
# Usage: bash infrastructure/grafana/export-dashboards.sh
#
# - Filenames use UID (immutable) with slug suffix for readability: <uid>__<slug>.json
# - Stale files (dashboards deleted in UI) are removed automatically.
# - Requires: Grafana running, curl, python3

set -euo pipefail

GRAFANA_URL="${GRAFANA_URL:-http://127.0.0.1:3000/grafana}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"
DASHBOARD_DIR="$(cd "$(dirname "$0")" && pwd)/dashboards"

AUTH="${GRAFANA_USER}:${GRAFANA_PASSWORD}"

mkdir -p "$DASHBOARD_DIR"

echo "Exporting dashboards from ${GRAFANA_URL} ..."

# Track exported files to detect stale ones
exported_files=()

# List all dashboards
uids=$(curl -sf -u "$AUTH" "${GRAFANA_URL}/api/search?type=dash-db" \
    | python3 -c "import sys,json; [print(d['uid']) for d in json.load(sys.stdin)]")

count=0
for uid in $uids; do
    response=$(curl -sf -u "$AUTH" "${GRAFANA_URL}/api/dashboards/uid/${uid}")

    slug=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['meta']['slug'])")
    filename="${uid}__${slug}.json"

    echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)['dashboard']
# Remove transient fields that change on every save
data.pop('id', None)
print(json.dumps(data, indent=2))
" > "${DASHBOARD_DIR}/${filename}"

    exported_files+=("$filename")
    echo "  exported: ${filename}"
    count=$((count + 1))
done

# Remove stale JSON files (dashboards deleted in UI)
removed=0
for existing in "$DASHBOARD_DIR"/*.json; do
    [ -f "$existing" ] || continue
    basename=$(basename "$existing")
    found=false
    for f in "${exported_files[@]}"; do
        if [ "$f" = "$basename" ]; then
            found=true
            break
        fi
    done
    if [ "$found" = false ]; then
        echo "  removed stale: ${basename}"
        rm "$existing"
        removed=$((removed + 1))
    fi
done

echo "Done. ${count} exported, ${removed} stale removed."
