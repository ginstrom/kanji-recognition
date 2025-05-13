# Development Rules

## Terminal Output Bug
- There is a bug preventing reading terminal output directly
- When running shell commands, use `tee` to both display and save output to files in /tmp:
  ```bash
  docker-compose up | tee /tmp/docker-compose-out.txt
  ```
- For commands that might produce errors, capture both stdout and stderr:
  ```bash
  docker-compose run --rm handwriting python train.py 2>&1 | tee /tmp/train-output.txt
  ```

## Running Code
- Always use docker-compose to run code
- Never run Python scripts directly on the host machine
- Remember that the working directory in the container is `/app/src`, so use script names without the `src/` prefix
- For foreground processes:
  ```bash
  docker-compose run --rm handwriting python train.py 2>&1 | tee /tmp/train-output.txt
  ```
- For detached mode (background processes):
  ```bash
  docker-compose up -d
  # Then check logs with:
  docker-compose logs | tee /tmp/docker-logs.txt
  ```
- For interactive sessions:
  ```bash
  docker-compose run --rm handwriting bash
  ```

## Checking Logs
- For containers running in detached mode (-d), use:
  ```bash
  docker-compose logs | tee /tmp/docker-logs.txt
  ```
- To follow logs in real-time:
  ```bash
  docker-compose logs -f | tee /tmp/docker-logs-live.txt
  ```
- For specific services:
  ```bash
  docker-compose logs handwriting | tee /tmp/handwriting-logs.txt
  ```

Remember to always follow these rules to ensure consistent development practices and to work around the terminal output bug.
