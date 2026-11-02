#!/bin/sh
# litellm-jah wrapper around the base image entrypoint:
# the shipped litellm_config.yaml reads its master key from LITELLM_MASTER_KEY;
# generate a random one when the operator did not set it.
if [ -z "$LITELLM_MASTER_KEY" ]; then
    LITELLM_MASTER_KEY="sk-$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
    export LITELLM_MASTER_KEY
    echo "[litellm-jah] LITELLM_MASTER_KEY not set: generated a random master key: $LITELLM_MASTER_KEY"
    echo "[litellm-jah] set LITELLM_MASTER_KEY to use a fixed key instead"
fi
exec /app/docker/prod_entrypoint.sh "$@"
