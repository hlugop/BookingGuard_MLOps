# ML Architecture - Hotel Cancellation Prediction

## Overview

This document describes the end-to-end ML architecture for deploying, monitoring, and maintaining the hotel cancellation prediction model in a production environment.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PRODUCTION ENVIRONMENT                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Data       │    │   Feature    │    │   Model      │    │   Model      │       │
│  │   Sources    │───▶│   Store      │───▶│   Training   │───▶│   Registry   │       │
│  │   (Hotels)   │    │              │    │   (MLflow)   │    │   (MLflow)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                   │               │
│         │                   │                   │                   │               │
│         ▼                   ▼                   ▼                   ▼               │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                         CI/CD PIPELINE (GitHub Actions)                   │      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │      │
│  │  │  Lint   │─▶│  Test   │─▶│  Build  │─▶│ Deploy  │─▶│ Monitor │        │      │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                      │                                              │
│                                      ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                         KUBERNETES CLUSTER                                │      │
│  │                                                                           │      │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │      │
│  │  │   API       │    │   API       │    │   API       │   Auto-scaling    │      │
│  │  │   Pod 1     │    │   Pod 2     │    │   Pod N     │   based on load   │      │
│  │  └─────────────┘    └─────────────┘    └─────────────┘                   │      │
│  │         │                 │                 │                             │      │
│  │         └─────────────────┼─────────────────┘                             │      │
│  │                           │                                               │      │
│  │                           ▼                                               │      │
│  │                  ┌─────────────────┐                                      │      │
│  │                  │  Load Balancer  │                                      │      │
│  │                  │   (Ingress)     │                                      │      │
│  │                  └─────────────────┘                                      │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                      │                                              │
│                                      ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                         MONITORING & OBSERVABILITY                        │      │
│  │                                                                           │      │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │      │
│  │  │ Prometheus  │    │  Grafana    │    │   Alerts    │                   │      │
│  │  │  Metrics    │───▶│ Dashboards  │───▶│  (PagerDuty)│                   │      │
│  │  └─────────────┘    └─────────────┘    └─────────────┘                   │      │
│  │                                                                           │      │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │      │
│  │  │  Model      │    │  Data       │    │   ELK       │                   │      │
│  │  │  Monitoring │    │  Drift      │    │   Logs      │                   │      │
│  │  └─────────────┘    └─────────────┘    └─────────────┘                   │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                      │
                                      ▼
                         ┌──────────────────────┐
                         │     External API     │
                         │    (Hotel Systems)   │
                         └──────────────────────┘
```

## Components

### 1. Data Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Sources | Hotel PMS, Booking Engines | Raw booking data |
| Feature Store | Feast / Redis | Consistent feature serving |
| Data Validation | Great Expectations | Data quality checks |

### 2. Training Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| Experiment Tracking | MLflow | Track experiments, metrics, artifacts |
| Model Registry | MLflow | Version and stage models |
| Training Orchestration | Airflow / Kubeflow | Schedule and manage training pipelines |

### 3. Serving Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Framework | FastAPI | REST API for predictions |
| Container Runtime | Docker | Consistent deployment |
| Orchestration | Kubernetes | Scaling and orchestration |
| Load Balancer | NGINX Ingress | Traffic distribution |

### 4. Monitoring Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| Metrics | Prometheus | System and business metrics |
| Visualization | Grafana | Dashboards and alerts |
| Logging | ELK Stack | Centralized logging |
| Model Monitoring | Evidently AI / Alibi Detect | Drift detection |

## Data Flow

1. **Training Flow**:
   ```
   Raw Data → Validation → Feature Engineering → Model Training → Model Registry
   ```

2. **Inference Flow**:
   ```
   API Request → Feature Transform → Model Prediction → API Response
   ```

3. **Monitoring Flow**:
   ```
   Predictions → Metrics Collection → Drift Detection → Alert/Retrain
   ```

## Deployment Strategy

### Blue-Green Deployment

```
┌─────────────────┐     ┌─────────────────┐
│   Blue (v1.0)   │     │  Green (v1.1)   │
│   100% Traffic  │────▶│   0% Traffic    │
└─────────────────┘     └─────────────────┘
         │                      │
         │   After Validation   │
         ▼                      ▼
┌─────────────────┐     ┌─────────────────┐
│   Blue (v1.0)   │     │  Green (v1.1)   │
│   0% Traffic    │◀────│   100% Traffic  │
└─────────────────┘     └─────────────────┘
```

### Canary Release

1. Deploy new version to small subset (5-10%)
2. Monitor metrics and errors
3. Gradually increase traffic
4. Full rollout or rollback

## Model Monitoring

### Key Metrics

| Metric | Threshold | Action |
|--------|-----------|--------|
| Prediction Latency (p99) | < 100ms | Scale pods |
| Error Rate | < 0.1% | Alert & investigate |
| Data Drift (PSI) | < 0.2 | Retrain model |
| Model Performance (F1) | > 0.75 | Retrain model |

### Drift Detection

- **Input Drift**: Monitor feature distributions using Population Stability Index (PSI)
- **Concept Drift**: Monitor prediction distribution changes
- **Performance Drift**: Compare predictions with actual outcomes (delayed labels)

## Retraining Strategy

```
┌─────────────────┐
│  Trigger Event  │
│  - Schedule     │
│  - Drift Alert  │
│  - Performance  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Pipeline  │
│  - Extract      │
│  - Validate     │
│  - Transform    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Training       │
│  - Train Model  │
│  - Evaluate     │
│  - Compare      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deployment     │
│  - Stage Model  │
│  - A/B Test     │
│  - Promote      │
└─────────────────┘
```

## Security Considerations

1. **API Security**
   - Authentication (API Keys / OAuth2)
   - Rate limiting
   - Input validation

2. **Data Security**
   - Encryption at rest and in transit
   - PII handling compliance
   - Access control (RBAC)

3. **Infrastructure Security**
   - Network policies
   - Secret management (Vault)
   - Container scanning

## Cost Optimization

- Spot/Preemptible instances for training
- Auto-scaling based on traffic patterns
- Model optimization (quantization, pruning)
- Efficient batch inference for bulk predictions
