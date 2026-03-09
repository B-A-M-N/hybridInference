# Scripts Directory

This directory contains helper scripts used to **provision and run the staging environment
on a fresh temporary server**.

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
-   installs `uv` (Python dependency tool)
-   adds the current user to the `docker` group
-   configures the shell so `uv` is available

This script only needs to run **once per server**.
After running it you must either:

```bash
newgrp docker
```
or log out and SSH back in.

## start-staging.sh

Starts the staging environment and refreshes the application service.

Responsibilities:
- changes to the repository root
- loads `uv` into the shell environment if available
- installs or syncs Python dependencies with `uv sync`
- pulls infrastructure images using the staging compose file
- starts infrastructure containers in detached mode
- prints container status
- restarts the `hybrid_inference.staging` systemd service
- shows the current service status

### Verify installation

```bash
docker --version
docker compose version
uv --version
```
