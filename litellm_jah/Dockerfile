# JevAgentHarness decision points for the LiteLLM proxy: the plugin package on the official LiteLLM image.
# Build from the repository root:
#   docker build -f litellm_jah/Dockerfile -t litellm-jah:latest .
FROM docker.litellm.ai/berriai/litellm:1.106.6

# The plugin package; the redis extra makes the shared RedisSessionStore available for multi-replica use.
# The image's /app/.venv is uv-managed and ships without pip: bootstrap it first.
COPY litellm_jah/ /tmp/litellm_jah/
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir "/tmp/litellm_jah[redis]" \
    && rm -rf /tmp/litellm_jah

# The proxy configs (override with --config or a volume mount for production).
# The master key is read from LITELLM_MASTER_KEY; the entrypoint generates a random one when it is unset.
# The JAH config is neutral: rules decision backend, no vendor endpoints.
COPY litellm_jah/docker/litellm_config.yaml /app/litellm_config.yaml
COPY litellm_jah/docker/jah-config.yaml /app/jah-config.yaml
COPY litellm_jah/docker/jah_entrypoint.sh /app/jah_entrypoint.sh
RUN chmod +x /app/jah_entrypoint.sh
ENV LITELLM_JAH_CONFIG=/app/jah-config.yaml
ENTRYPOINT ["/app/jah_entrypoint.sh"]
