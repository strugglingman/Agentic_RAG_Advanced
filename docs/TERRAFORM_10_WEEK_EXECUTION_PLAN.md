# Terraform Learning + Implementation Plan (10 Weeks)

Audience: developer new to Terraform, using this repository as a real migration case.

## Success Criteria

By the end of Week 10, you can:
1. Explain Terraform state, plan/apply workflow, and import strategy.
2. Safely migrate existing AWS resources into Terraform state.
3. Run PR-based infra changes with reviewable plans.
4. Present the project in resume/interviews with measurable outcomes.

## Week 1 (Now): Workflow Basics In This Repo

1. Install Terraform CLI.
2. Run in `infra/terraform/envs/dev`:
- `terraform init`
- `terraform fmt -recursive`
- `terraform validate`
- `terraform plan -var-file=terraform.tfvars.example`
3. Explain each variable in `envs/dev/variables.tf` in plain language.

Definition of done:
1. You can run plan without confusion.
2. You can describe why `enable_monitoring=false` is safer for learning.

## Week 2: Remote State + First Real Resources

1. Build bootstrap state backend (S3 + lock mechanism).
2. Configure dev root to use remote backend.
3. Enable monitoring module in dev and apply:
- SNS topic
- RDS CPU alarm
- Redis evictions alarm

Definition of done:
1. CloudWatch shows Terraform-managed alarms.
2. You can produce and explain a clean `plan`.

## Week 3: Import Existing Monitoring Resources

1. Import existing SNS/alarm resources instead of recreating.
2. Fix drift until plan is no-op.

Definition of done:
1. `terraform plan` shows no unintended changes for monitoring stack.

## Week 4: IAM Foundations

1. Create IAM module conventions.
2. Manage one role + policy attachment used by EKS add-ons.
3. Add tags and naming standards.

Definition of done:
1. Role/policy is fully code-managed.

## Week 5: EKS Add-ons + Pod Identity

1. Terraformize EKS add-on resources.
2. Terraformize Pod Identity associations.
3. Validate add-ons remain healthy.

Definition of done:
1. No add-on regression after apply.

## Week 6: Core Data Services (RDS / ElastiCache)

1. Import current RDS and ElastiCache definitions.
2. Add lifecycle safeguards for critical resources.
3. Validate no accidental replacement risk.

Definition of done:
1. Plan is stable and does not propose destructive changes.

## Week 7: EKS Core + Node Group

1. Import EKS cluster and node group carefully.
2. Split module ownership for cluster vs workloads.

Definition of done:
1. Cluster remains stable and drift is controlled.

## Week 8: Edge (ACM + Route53 + ALB Side)

1. Manage ACM + DNS resources with Terraform.
2. Keep K8s ingress YAML as-is initially.

Definition of done:
1. TLS/domain infra can be reproduced from code.

## Week 9: CI/CD And Governance

1. Add CI jobs for `fmt`, `validate`, and gated `plan`.
2. Add policy/security scan tool.

Definition of done:
1. PR cannot merge without Terraform checks.

## Week 10: Resume + Interview Packaging

1. Build one-page case study:
- Problem
- Migration strategy
- Risk controls
- Outcomes
2. Prepare 6 STAR interview stories:
- import strategy
- drift incident
- rollback decision
- security hardening
- cost control
- production change review

Definition of done:
1. You can explain this project end-to-end in 10 minutes.

