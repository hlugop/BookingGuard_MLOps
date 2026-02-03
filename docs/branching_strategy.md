# Branching Strategy

## Overview

This document describes the Git branching strategy for the MLOps project, following GitFlow adapted for ML projects.

## Branch Types

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BRANCH STRUCTURE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  main (production)                                                               │
│    │                                                                             │
│    ├──────────────────────────────────────────────────────────────────────┐     │
│    │                                                                       │     │
│    │  develop                                                              │     │
│    │    │                                                                  │     │
│    │    ├─── feature/add-batch-prediction ─────────────────────────┐      │     │
│    │    │                                                           │      │     │
│    │    ├─── feature/improve-feature-engineering ──────────┐       │      │     │
│    │    │                                                   │       │      │     │
│    │    ├─── experiment/xgboost-tuning ────────┐           │       │      │     │
│    │    │                                       │           │       │      │     │
│    │    ├─── data/update-training-data ──┐     │           │       │      │     │
│    │    │                                 │     │           │       │      │     │
│    │    └─────────────────────────────────┴─────┴───────────┴───────┴──────┤     │
│    │                                                                       │     │
│    ├──────────────────────────────────────────────────────────────────────┘     │
│    │                                                                             │
│    ├─── release/v1.0.0 ──────────────────────────────────────────────┐          │
│    │                                                                  │          │
│    ├─── hotfix/fix-prediction-bug ───────────────────────────────────│──┐       │
│    │                                                                  │  │       │
│    └──────────────────────────────────────────────────────────────────┴──┴──────│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Branch Naming Convention

| Branch Type | Pattern | Example | Purpose |
|-------------|---------|---------|---------|
| Main | `main` | `main` | Production-ready code |
| Develop | `develop` | `develop` | Integration branch |
| Feature | `feature/<description>` | `feature/add-batch-api` | New features |
| Experiment | `experiment/<description>` | `experiment/lightgbm-comparison` | ML experiments |
| Data | `data/<description>` | `data/add-2024-reservations` | Data updates |
| Release | `release/v<version>` | `release/v1.2.0` | Release preparation |
| Hotfix | `hotfix/<description>` | `hotfix/fix-memory-leak` | Production fixes |

## Workflow

### 1. Feature Development

```bash
# Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/add-new-endpoint

# Work on feature
# ... make commits ...

# Push and create PR
git push origin feature/add-new-endpoint
# Create PR to develop via GitHub
```

### 2. ML Experiment

```bash
# Create experiment branch
git checkout develop
git checkout -b experiment/hyperparameter-tuning

# Run experiments, track in MLflow
# ... make commits with experiment results ...

# If successful, merge to develop
git push origin experiment/hyperparameter-tuning
# Create PR to develop
```

### 3. Release Process

```bash
# Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v1.2.0

# Update version numbers, final testing
# ... make commits ...

# Merge to main
git checkout main
git merge release/v1.2.0
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin main --tags

# Back-merge to develop
git checkout develop
git merge release/v1.2.0
git push origin develop
```

### 4. Hotfix Process

```bash
# Create hotfix from main
git checkout main
git checkout -b hotfix/fix-critical-bug

# Fix the issue
# ... make commits ...

# Merge to main
git checkout main
git merge hotfix/fix-critical-bug
git tag -a v1.2.1 -m "Hotfix v1.2.1"
git push origin main --tags

# Back-merge to develop
git checkout develop
git merge hotfix/fix-critical-bug
git push origin develop
```

## CI/CD Pipeline Integration

### Branch-Specific Actions

| Branch | CI Actions | CD Actions |
|--------|------------|------------|
| `feature/*` | Lint, Test | - |
| `experiment/*` | Lint, Test, Run Experiments | - |
| `develop` | Lint, Test, Build | Deploy to Dev |
| `release/*` | Lint, Test, Build, Integration Tests | Deploy to Staging |
| `main` | Lint, Test, Build, Integration Tests | Deploy to Production |
| `hotfix/*` | Lint, Test | Deploy to Staging |

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml (example structure)
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop, 'feature/**', 'release/**']
  pull_request:
    branches: [main, develop]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run linting
        run: |
          pip install black flake8 isort
          black --check src/
          flake8 src/
          isort --check src/

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -e ".[dev]"
          pytest tests/ --cov=src

  build:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t hotel-prediction:${{ github.sha }} .

  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: echo "Deploy to production"
```

## ML-Specific Considerations

### Model Versioning

- Each model training run creates a new version in MLflow
- Model versions are linked to Git commits via tags
- Production models are promoted through MLflow stages: `None` → `Staging` → `Production`

### Experiment Branches

- Used for trying new algorithms, features, or hyperparameters
- Must include MLflow experiment tracking
- Successful experiments are merged to develop
- Failed experiments are documented and archived

### Data Branches

- Used when training data is updated
- Triggers full retraining pipeline
- Includes data validation checks
- Documents data changes in commit messages

## Best Practices

1. **Atomic Commits**: Each commit should represent a single logical change
2. **Meaningful Messages**: Use conventional commit format
   ```
   feat: add batch prediction endpoint
   fix: resolve memory leak in preprocessing
   docs: update API documentation
   experiment: test LightGBM vs XGBoost
   data: add Q4 2024 reservations
   ```
3. **Code Review**: All PRs require at least one review
4. **Protected Branches**: `main` and `develop` are protected
5. **Linear History**: Use rebase for feature branches
6. **Tag Releases**: All releases are tagged with semantic versioning

## Branch Lifecycle

```
Create Branch → Develop → PR → Review → Merge → Delete Branch
```

- Feature branches are deleted after merging
- Experiment branches are archived (not deleted) for reference
- Release branches are deleted after merging to main and develop
