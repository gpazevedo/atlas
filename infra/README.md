# ATLAS Counsel — AWS Infrastructure

Terraform configuration for deploying ATLAS Counsel on ECS Fargate.

## Prerequisites

- AWS account with `AdministratorAccess` (or equivalent)
- Terraform >= 1.5
- Qdrant Cloud cluster (free tier works) — get the URL from the dashboard
- Docker image pushed to ECR (CI/CD handles this; for manual first deploy, run
  `docker build -t atlas-counsel .` and push to the ECR repo)

## Quickstart

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your Qdrant Cloud URL
terraform init
terraform plan
terraform apply
```

## What it creates

- VPC with 2 public subnets, Internet Gateway
- ECS Fargate cluster + service (1 task, 256 CPU / 512 MB)
- Application Load Balancer (HTTP on port 80, health check at /health)
- EFS file system for per-tenant SQLite checkpoints (encrypted at rest)
- ECR repository for Docker images
- CloudWatch log group (30-day retention)
- Security groups: ALB public, app from ALB only, EFS from app only

## Required secrets

- `qdrant_url` — Qdrant Cloud cluster URL (set in terraform.tfvars, not committed)

## Cleaning up

```bash
terraform destroy
```
