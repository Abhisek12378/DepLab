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

`shared/outputs/` must contain these private files before deployment:

```text
deplab-expanded-development-v2.0.0/features.csv
deplab-expanded-weighted-logistic-v2.0.0/model.json
deplab-advanced-model-comparison-v3.0.0/model.json
deplab-hybrid-validation-v3.0.0/metrics.json
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
