# Scripts Directory

This directory contains helper scripts used to provision and run the fully
Dockerized staging environment on a fresh temporary server.

The staging servers used for experiments are **ephemeral**. When the node reboots or
expires, the disk resets to the base OS image. Any local state (Docker volumes, database
files, Grafana data, etc.) is lost.
These scripts make it easy to recreate the environment quickly on a new machine.

# Overview

Typical workflow when starting a new staging server:

1.  SSH into the server
2.  Clone this repository
3.  Run the bootstrap script to install system dependencies
4.  Start the staging services
5.  Optionally restore staging data

Example:

```bash
git clone <repo-url>
cd hybridInference/scripts/staging

chmod +x *.sh

./bootstrap_server.sh
newgrp docker

./start_staging.sh
```

# Scripts

## bootstrap-server.sh

Prepares a fresh Ubuntu server for running the staging stack.

Responsibilities:
-   installs required system packages
-   installs Docker Engine
-   installs Docker Compose
-   adds the current user to the `docker` group

This script only needs to run **once per server**.
After running it you must either:

```bash
newgrp docker
```
or log out and SSH back in.

## start-staging.sh

Starts the staging environment entirely through Docker Compose.

Responsibilities:
- changes to the repository root
- pulls infrastructure images using the staging compose file
- builds the backend and frontend images from the current worktree
- starts staging containers in detached mode
- prints container status

### Verify installation

```bash
docker --version
docker compose version
```

## Services Started by the Stack
After everything starts, the server runs several services:
- Staging Frontend: Next.js dashboard UI (3002 by default)
- FastAPI Gateway: Main API server for LLM requests (8000)
- PostgreSQL: Stores logs, users, API keys (5433)
- Prometheus: Collects application metrics (9091)
- Grafana: Visual dashboard for metrics (3001)
