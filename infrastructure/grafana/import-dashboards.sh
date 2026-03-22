#!/bin/bash
# Import all dashboard JSON files from the repo into a running Grafana instance.
# Usage: bash infrastructure/grafana/import-dashboards.sh
#
# Use case: disaster recovery, cold-start on a new machine, or restoring from backup.
# - Creates or updates dashboards (matched by UID in the JSON).
# - Requires: Grafana running, curl, python3

set -euo pipefail

GRAFANA_URL="${GRAFANA_URL:-http://127.0.0.1:3000/grafana}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"
DASHBOARD_DIR="$(cd "$(dirname "$0")" && pwd)/dashboards"

AUTH="${GRAFANA_USER}:${GRAFANA_PASSWORD}"

echo "Importing dashboards from ${DASHBOARD_DIR} into ${GRAFANA_URL} ..."

count=0
for file in "$DASHBOARD_DIR"/*.json; do
    [ -f "$file" ] || continue
    basename=$(basename "$file")

    # Wrap the dashboard JSON in the import envelope
    payload=$(python3 -c "
import sys, json
with open('$file') as f:
    dashboard = json.load(f)
# Ensure id is null so Grafana treats it as create-or-update by UID
dashboard['id'] = None
envelope = {
    'dashboard': dashboard,
    'overwrite': True,
    'message': 'Imported from repo by import-dashboards.sh'
}
print(json.dumps(envelope))
")

    http_code=$(curl -sf -o /dev/null -w "%{http_code}" \
        -u "$AUTH" \
        -H "Content-Type: application/json" \
        -X POST "${GRAFANA_URL}/api/dashboards/db" \
        -d "$payload")

    if [ "$http_code" = "200" ]; then
        echo "  imported: ${basename}"
        count=$((count + 1))
    else
        echo "  FAILED (HTTP ${http_code}): ${basename}" >&2
    fi
done

echo "Done. ${count} dashboard(s) imported."
