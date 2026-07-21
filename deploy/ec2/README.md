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
