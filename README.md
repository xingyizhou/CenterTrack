# CenterTrack (WATonomous version)

This repo is derived from the original [CenterTrack repo](https://github.com/xingyizhou/CenterTrack). The original README can be found in [README.original.md](README.original.md).

## Getting Started

1. Make sure `/etc/docker/daemon.json` contains `"default-runtime": "nvidia"` as a top-level property. For example:

```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

If changes are made, restart the docker daemon:

```bash
sudo service docker restart
```

2. Start the container. Some assumptions have been made in `docker-compose.yml` (e.g. where the data is stored). Edit the file if needed.

```bash
touch .env && echo "COMPOSE_PROJECT_NAME=centertrack_$USER" >> .env
docker-compose up -d
```

3. Open a shell

```bash
docker-compose exec dev /bin/bash
```

4. [Only needed to run once] Copy DCNv2 into the mounted directory

```bash
# in the shell in docker
cp -r /CenterTrack/src/lib/model/networks/DCNv2 src/lib/model/networks/
```

