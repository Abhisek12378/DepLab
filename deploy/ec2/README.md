# EC2 deployment contract

The GitHub workflow deploys only source code and a compiled React frontend. It
never uploads the trained model, feature table, experiment results, `.env`, or
SSH keys.

EC2 keeps private runtime material outside each source release:

```text
/opt/deplab/
  current -> /opt/deplab/releases/<release-id>
  releases/
  shared/
    .env
    outputs/
```

`shared/models/current/` must point to an immutable, validated model bundle
containing:

```text
outputs/deplab-large-features-v3.0.0/development-features.csv
outputs/deplab-large-features-v3.0.0/validation-inputs.csv
outputs/deplab-large-candidate-freeze-v3.0.0/candidate-structured_weighted_logistic.json
outputs/deplab-large-candidate-freeze-v3.0.0/candidate-modernbert_stage_aware_hybrid.json
outputs/large-release-modernbert-v3.0.0.jsonl
```

The API uses the structured model only to rank candidate environments. It asks
`uv pip compile` to verify dependency resolution for a bounded number of top
candidates, then uses the frozen ModernBERT post-install heads to predict import
and smoke-test risk. Request-time checks never install packages or execute their
code.

The shared environment must contain:

```text
DEPLAB_MODEL_ROOT=/opt/deplab/shared/models/current
DEPLAB_UV_COMMAND=/usr/local/bin/uv
DEPLAB_UV_CACHE_DIR=/opt/deplab/shared/uv-cache
DEPLAB_RESOLVER_TIMEOUT_SECONDS=15
DEPLAB_RESOLVER_MAXIMUM_CONCURRENCY=2
```

The GitHub `production` environment needs these secrets:

- `EC2_HOST`
- `EC2_USER`
- `EC2_SSH_PRIVATE_KEY`
- `EC2_KNOWN_HOSTS`

The repository variable `EC2_DEPLOY_ENABLED` must be `true` before deployment
runs. Until then, pushes to `main` run CI only. This prevents an incomplete EC2
setup from causing failed or unsafe deployments.

## HTTPS hostname

The hackathon deployment uses this free IP-based hostname:

```text
deplab.13-234-114-139.sslip.io
```

The hostname resolves to the current EC2 public IP without a separate DNS
account. Nginx starts with the HTTP configuration in `nginx-deplab.conf`, then
Certbot's Nginx integration obtains a Let's Encrypt certificate, enables the
HTTPS listener, and redirects HTTP requests to HTTPS. Certbot also installs its
automatic renewal timer.

The EC2 security group must allow inbound TCP ports 80 and 443. Because this
hostname contains the public IP, it must be updated and the certificate reissued
if the instance's public IP changes. An Elastic IP or a purchased domain avoids
that limitation.
