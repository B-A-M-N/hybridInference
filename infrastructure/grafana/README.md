# Grafana Dashboards

Dashboards are managed via the Grafana UI. Changes persist in the `grafana_data` Docker volume.

The JSON files in `dashboards/` are version-controlled backups, **not** auto-loaded by Grafana.

## Sync: Grafana UI → Repo

After editing dashboards in the UI, export them to the repo:

```bash
bash infrastructure/grafana/export-dashboards.sh
git add infrastructure/grafana/dashboards/
git commit -m "chore: sync Grafana dashboards"
```

## Restore: Repo → Grafana

To recover dashboards after a volume loss or cold-start on a new machine:

```bash
bash infrastructure/grafana/import-dashboards.sh
```

## File Naming

Exported files are named `<uid>__<slug>.json` (e.g., `d3bf03c0-1126-4e65-b8a3-03bdb4c6c82c__hybrid-inference.json`). UID is immutable; slug is for readability.

## Provisioning

- `provisioning/datasources/` — auto-configures Prometheus and PostgreSQL datasources on container start (UIDs match the migrated `grafana.db`).
- `provisioning/dashboards/` — intentionally empty (`providers: []`). Dashboards live in the DB, not files.
- `grafana.ini` — reference copy of host Grafana config (not mounted into Docker; env vars are used instead).
