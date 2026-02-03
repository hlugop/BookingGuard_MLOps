# Hotel Reservation Cancellation Prediction - MLOps Project

[![CI/CD Pipeline](https://github.com/hlugop/HotelCancellationPredictor/actions/workflows/ci.yml/badge.svg)](https://github.com/hlugop/HotelCancellationPredictor/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hlugop/HotelCancellationPredictor/branch/main/graph/badge.svg)](https://codecov.io/gh/hlugop/HotelCancellationPredictor)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About This Project

Este proyecto implementa un pipeline MLOps completo para predecir cancelaciones de reservas de hotel. Utiliza XGBoost como modelo principal, MLflow para tracking de experimentos, y FastAPI para servir predicciones.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Quick Start - Docker (Recomendado)](#quick-start---docker-recomendado)
4. [Quick Start - Local](#quick-start---local)
5. [Training Pipeline](#training-pipeline)
6. [Inference Pipeline](#inference-pipeline)
7. [API Documentation](#api-documentation)
8. [Docker Deployment](#docker-deployment)
9. [Feature Store](#feature-store)
10. [Testing](#testing)
11. [MLOps Concepts](#mlops-concepts)
12. [CI/CD Pipeline](#cicd-pipeline)
13. [Technical Requirements Checklist](#technical-requirements-checklist)

---

## Project Overview

This project demonstrates MLOps best practices for productionizing a machine learning model that predicts hotel reservation cancellations. The model uses XGBoost for classification and achieves an ROC-AUC of ~0.90.

### Business Problem
Hotels lose significant revenue from last-minute cancellations. By predicting which reservations are likely to be canceled, hotels can:
- Implement targeted retention strategies
- Optimize overbooking policies
- Improve revenue management

### Dataset
- **Source**: Hotel Reservations Dataset
- **Size**: 36,275 records
- **Target**: `booking_status` (Canceled / Not_Canceled)
- **Cancellation Rate**: ~33%

---

## Project Structure

```
hotel_cancellation_predictor/
├── src/                          # Source code
│   ├── __init__.py
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py          # Pydantic Settings (centralized config)
│   ├── container.py             # Dependency Injection Container
│   ├── exceptions.py            # Custom exception classes
│   ├── data/                     # Data loading and validation
│   │   ├── __init__.py
│   │   ├── loader.py            # DataLoader class
│   │   └── validator.py         # DataValidator class
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py
│   │   ├── engineer.py          # FeatureEngineer class
│   │   ├── preprocessor.py      # Preprocessor class (sklearn pipeline)
│   │   └── store.py             # FeatureStore for feature versioning
│   ├── models/                   # Model training and prediction
│   │   ├── __init__.py
│   │   ├── trainer.py           # ModelTrainer with MLflow
│   │   └── predictor.py         # ModelPredictor for inference
│   ├── pipelines/                # Orchestration pipelines
│   │   ├── __init__.py
│   │   ├── training.py          # TrainingPipeline (with DI)
│   │   └── inference.py         # InferencePipeline (with DI)
│   ├── api/                      # FastAPI application
│   │   ├── __init__.py
│   │   ├── app.py               # FastAPI app (with DI)
│   │   └── schemas.py           # Pydantic models
│   └── utils/                    # Utilities
│       ├── __init__.py
│       └── logging_config.py
├── tests/                        # Test suite
│   ├── __init__.py
│   └── unit/
│       ├── __init__.py
│       ├── test_feature_engineer.py
│       ├── test_data_validator.py
│       ├── test_api.py
│       ├── test_config.py       # Configuration tests
│       ├── test_container.py    # DI Container tests
│       ├── test_exceptions.py   # Exception tests
│       └── test_feature_store.py # Feature Store tests
├── data/                         # Data directory
│   ├── raw/                     # Raw data files
│   │   └── Hotel Reservations.csv
│   └── processed/               # Processed data
├── configs/                      # Configuration files
│   └── config.yaml
├── artifacts/                    # Model artifacts
│   └── models/
├── docs/                         # Documentation
│   ├── architecture.md          # ML Architecture diagram
│   └── branching_strategy.md    # Git branching strategy
├── Dockerfile                    # Docker configuration (API)
├── Dockerfile.mlflow            # Docker configuration (MLflow server)
├── docker-compose.yml           # Docker Compose for full stack
├── .dockerignore
├── pyproject.toml               # Project configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Architecture Highlights

The project follows enterprise-grade MLOps patterns:

| Pattern | Implementation | Benefit |
|---------|----------------|---------|
| **Centralized Configuration** | `src/config/settings.py` | Single source of truth, environment-specific configs |
| **Dependency Injection** | `src/container.py` | Testability, loose coupling, flexible component swapping |
| **Custom Exceptions** | `src/exceptions.py` | Clear error hierarchy, structured error responses |
| **Factory Pattern** | `Container.training_pipeline()` | Consistent object creation with dependencies |
| **Repository Pattern** | `DataLoader`, `DataValidator` | Separation of data access from business logic |
| **Feature Store** | `src/features/store.py` | Feature versioning, reuse between training and inference |

---

## Quick Start - Docker (Recomendado)

Esta es la forma más rápida de levantar todo el proyecto. Solo necesitas Docker instalado.

### Prerrequisitos

- Docker Desktop instalado y corriendo
- Git (para clonar el repo)

### Paso 1: Clonar y entrar al proyecto

```bash
git clone https://github.com/hlugop/HotelCancellationPredictor.git
cd HotelCancellationPredictor
```

### Paso 2: Levantar los servicios

```bash
docker-compose up --build -d
```

> **Nota:** La primera vez toma varios minutos porque construye las imágenes desde cero. Ve por un café mientras esperas.

### Paso 3: Verificar que todo esté corriendo

```bash
docker ps
```

Deberías ver dos contenedores con status `(healthy)`:
- `hotel-cancellation-api` en puerto 8000
- `mlflow-server` en puerto 5000

### Paso 4: Entrenar el modelo

```bash
docker exec -it hotel-cancellation-api python src/pipelines/training.py
```

Esto entrena el modelo XGBoost y registra todo en MLflow. Puedes ver los experimentos en http://localhost:5000

### Paso 5: Probar la API

Abre http://localhost:8000/docs para ver la documentación interactiva, o usa curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "no_of_adults": 2,
    "no_of_children": 0,
    "no_of_weekend_nights": 1,
    "no_of_week_nights": 2,
    "type_of_meal_plan": "Meal Plan 1",
    "required_car_parking_space": 0,
    "room_type_reserved": "Room_Type 1",
    "lead_time": 224,
    "arrival_year": 2018,
    "arrival_month": 10,
    "arrival_date": 2,
    "market_segment_type": "Online",
    "repeated_guest": 0,
    "no_of_previous_cancellations": 0,
    "no_of_previous_bookings_not_canceled": 0,
    "avg_price_per_room": 65.0,
    "no_of_special_requests": 0
  }'
```

### Para detener todo

```bash
docker-compose down
```

---

## Quick Start - Local

Si prefieres correr el proyecto sin Docker, aquí están los pasos.

### Prerrequisitos

- Python 3.9 o superior
- pip

### Paso 1: Clonar y configurar el entorno

```bash
git clone https://github.com/hlugop/HotelCancellationPredictor.git
cd HotelCancellationPredictor

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -e ".[dev]"
```

### Paso 2: Entrenar el modelo

```bash
python -m src.pipelines.training --data-path "data/raw/Hotel Reservations.csv"
```

### Paso 3: Iniciar MLflow UI (opcional)

En otra terminal:
```bash
mlflow ui --port 5000
```

Abre http://localhost:5000 para ver los experimentos.

### Paso 4: Iniciar la API

```bash
export MODEL_URI="models:/hotel_cancellation_model/latest"
export ARTIFACTS_DIR="artifacts/models"

uvicorn src.api.app:app --reload --port 8000
```

Abre http://localhost:8000/docs para la documentación interactiva.

---

## Installation

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)

### Install from Source

```bash
# Install in development mode
pip install -e ".[dev]"

# Or install production dependencies only
pip install -r requirements.txt
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_URI` | MLflow model URI | `models:/hotel_cancellation_model/latest` |
| `ARTIFACTS_DIR` | Path to artifacts | `artifacts/models` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server | Local file store |

---

## Training Pipeline

**Location**: `src/pipelines/training.py`

The training pipeline orchestrates the complete model training workflow using **dependency injection**:

```python
# Recommended: Using Container (Dependency Injection)
from src.container import Container

container = Container.from_settings()
pipeline = container.training_pipeline()
results = pipeline.run(run_name="training_run_1")

# Alternative: Direct instantiation (for testing)
from src.pipelines.training import TrainingPipeline

pipeline = TrainingPipeline.from_settings()
results = pipeline.run(run_name="training_run_1")
```

### Pipeline Steps

1. **Load Data** → `DataLoader`
2. **Validate Data** → `DataValidator`
3. **Feature Engineering** → `FeatureEngineer`
   - Creates `total_nights` (weekend + weekday nights)
   - Creates `total_guests` (adults + children)
   - Encodes target variable
4. **Preprocessing** → `Preprocessor`
   - Numerical: Imputation + StandardScaler
   - Categorical: Imputation + OneHotEncoder
5. **Train Model** → `ModelTrainer` with MLflow tracking
6. **Save Artifacts** → Preprocessor, Feature Engineer

### MLflow Tracking

After training, view experiments in MLflow UI:

```bash
mlflow ui --port 5000
```

**Logged Artifacts**:
- Model parameters
- Metrics (ROC-AUC, F1, Precision, Recall, Accuracy)
- Confusion matrix plot
- Feature importance plot
- Trained model

---

## Inference Pipeline

**Location**: `src/pipelines/inference.py`

```python
# Recommended: Using Container (Dependency Injection)
from src.container import Container

container = Container.from_settings()
pipeline = container.inference_pipeline()
# Pipeline is automatically loaded with artifacts

# Single prediction
result = pipeline.predict_single({
    "no_of_adults": 2,
    "lead_time": 224,
    # ... other features
})
print(result)  # {"prediction": 1, "probability": 0.85, "label": "Canceled"}

# Batch prediction
results = pipeline.predict_batch([record1, record2, ...])
```

---

## API Documentation

**Location**: `src/api/app.py`

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |

### Swagger UI

Access interactive API docs at: `http://localhost:8000/docs`

### Request/Response Examples

#### Single Prediction

**Request**:
```json
POST /predict
{
  "no_of_adults": 2,
  "no_of_children": 0,
  "no_of_weekend_nights": 1,
  "no_of_week_nights": 2,
  "type_of_meal_plan": "Meal Plan 1",
  "required_car_parking_space": 0,
  "room_type_reserved": "Room_Type 1",
  "lead_time": 224,
  "arrival_year": 2018,
  "arrival_month": 10,
  "arrival_date": 2,
  "market_segment_type": "Online",
  "repeated_guest": 0,
  "no_of_previous_cancellations": 0,
  "no_of_previous_bookings_not_canceled": 0,
  "avg_price_per_room": 65.0,
  "no_of_special_requests": 0
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.847,
  "label": "Canceled"
}
```

#### Batch Prediction

**Request**:
```json
POST /predict/batch
{
  "reservations": [
    { /* reservation 1 */ },
    { /* reservation 2 */ }
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {"prediction": 1, "probability": 0.847, "label": "Canceled"},
    {"prediction": 0, "probability": 0.123, "label": "Not Canceled"}
  ],
  "count": 2
}
```

---

## Docker Deployment

El proyecto incluye dos Dockerfiles:
- `Dockerfile` - Para la API de FastAPI
- `Dockerfile.mlflow` - Para el servidor de MLflow (imagen personalizada con MLflow pre-instalado)

### Opción 1: Docker Compose (Recomendado)

Esta es la forma más sencilla. Levanta la API y MLflow con un solo comando:

```bash
# Construir y levantar todo
docker-compose up --build -d

# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f api
docker-compose logs -f mlflow

# Detener todo
docker-compose down

# Detener y eliminar volúmenes (borra datos de MLflow)
docker-compose down -v
```

### Opción 2: Construir imágenes manualmente

Si necesitas más control, puedes construir cada imagen por separado:

```bash
# Construir imagen de la API
docker build -t hotel-cancellation-api:latest .

# Construir imagen de MLflow
docker build -f Dockerfile.mlflow -t mlflow-server:latest .

# Correr MLflow primero
docker run -d \
  --name mlflow-server \
  -p 5000:5000 \
  -v mlflow-data:/mlflow \
  mlflow-server:latest

# Correr la API
docker run -d \
  --name hotel-api \
  -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts:ro \
  -e MLFLOW_TRACKING_URI="http://mlflow-server:5000" \
  --link mlflow-server \
  hotel-cancellation-api:latest

# Verificar que estén corriendo
docker ps

# Verificar health
curl http://localhost:8000/health
```

### Entrenar el modelo dentro de Docker

Una vez que los contenedores estén corriendo:

```bash
docker exec -it hotel-cancellation-api python src/pipelines/training.py
```

### Arquitectura de contenedores

| Contenedor | Puerto | Descripción |
|------------|--------|-------------|
| `hotel-cancellation-api` | 8000 | API FastAPI para predicciones |
| `mlflow-server` | 5000 | MLflow tracking server |

### Detalles técnicos

- **Base Image**: `python:3.11-slim`
- **Multi-stage Build**: Optimizado para producción (imagen API ~500MB)
- **Non-root User**: El contenedor de la API corre como `appuser` por seguridad
- **Health Checks**: Ambos contenedores tienen health checks configurados
- **Volúmenes**: Los datos de MLflow persisten en un volumen Docker

---

## Persistent Storage with Supabase

Por defecto, MLflow usa SQLite local que se pierde cuando el volumen de Docker se elimina. Para persistencia en la nube, el proyecto soporta **Supabase** (PostgreSQL gratuito).

### ¿Por qué Supabase?

| Característica | Free Tier |
|----------------|-----------|
| Base de datos PostgreSQL | 500 MB |
| Conexiones | Ilimitadas (pooling) |
| Uptime | 99.9% SLA |
| Dashboard | Incluido |

### Configuración de Supabase

#### Paso 1: Crear cuenta y proyecto

1. Ve a [supabase.com](https://supabase.com) y crea una cuenta
2. Crea un nuevo proyecto (elige la región más cercana)
3. Espera ~2 minutos a que el proyecto se inicialice

#### Paso 2: Obtener connection string

1. En el dashboard, ve a **Settings** > **Database**
2. En la sección "Connection string", selecciona **URI**
3. Copia el string que se ve así:
   ```
   postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
   ```

#### Paso 3: Configurar el proyecto

```bash
# Copiar el archivo de ejemplo
cp .env.example .env

# Editar .env y agregar tu connection string
nano .env  # o usa tu editor preferido
```

En `.env`, cambia esta línea:
```bash
SUPABASE_DB_URL=postgresql://postgres.[TU_PROJECT_REF]:[TU_PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
```

#### Paso 4: Levantar con persistencia

```bash
docker-compose up --build -d
```

Ahora todos los experimentos de MLflow se guardan en Supabase.

### Verificar conexión

```bash
# Ver logs de MLflow para confirmar conexión
docker-compose logs mlflow | grep -i postgres
```

Deberías ver algo como:
```
INFO:mlflow.store.db.utils:Creating initial MLflow database tables...
```

### Modo local vs Producción

| Modo | Backend Store | Persistencia |
|------|--------------|--------------|
| Local (sin .env) | SQLite | Solo en volumen Docker |
| Producción (con .env) | PostgreSQL (Supabase) | Nube, permanente |

### Variables de entorno

| Variable | Descripción | Requerida |
|----------|-------------|-----------|
| `SUPABASE_DB_URL` | Connection string de Supabase | Solo producción |
| `MLFLOW_ARTIFACT_ROOT` | Ruta para artefactos | No (default: /mlflow/artifacts) |
| `ENVIRONMENT` | development/production | No (default: development) |

---

## Feature Store

El proyecto incluye un **Feature Store** que permite almacenar y reutilizar features computadas, con soporte para versionado y múltiples backends (SQLite/PostgreSQL).

### ¿Qué es un Feature Store?

Un Feature Store es un componente central en MLOps que:
- **Almacena features computadas** para evitar recalcularlas
- **Versiona features** para reproducibilidad de experimentos
- **Comparte features** entre entrenamiento e inferencia
- **Reduce latencia** en inferencia al tener features pre-computadas

### Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                      Feature Store                               │
├─────────────────────────────────────────────────────────────────┤
│  feature_metadata (versiones)     │  feature_vectors (datos)    │
│  ┌─────────────────────────────┐  │  ┌───────────────────────┐  │
│  │ version: v1.0.0             │  │  │ entity_id: INN001     │  │
│  │ feature_names: [a, b, c]    │  │  │ version: v1.0.0       │  │
│  │ experiment_id: exp_001      │  │  │ feature_values: [...]  │  │
│  │ is_active: true             │  │  │ created_at: ...        │  │
│  └─────────────────────────────┘  │  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Uso Básico

```python
from src.features.store import FeatureStore

# Crear e inicializar el store
store = FeatureStore.from_settings()
store.initialize()

# Registrar una versión de features
store.register_version(
    version="v1.0.0",
    feature_names=["total_nights", "total_guests", "lead_time"],
    experiment_id="hotel_cancellation_exp",
    model_name="xgboost_v1"
)

# Almacenar features
store.store_features(
    entity_ids=["INN001", "INN002", "INN003"],
    feature_vectors=[[1.0, 2.0, 30], [2.0, 3.0, 45], [1.0, 1.0, 10]],
    version="v1.0.0"
)

# Recuperar features (online - single)
features = store.get_features("INN001", version="v1.0.0")
# {'entity_id': 'INN001', 'version': 'v1.0.0', 'feature_values': [1.0, 2.0, 30], ...}

# Recuperar features (offline - batch)
df = store.get_features_batch(["INN001", "INN002"], version="v1.0.0")
#   entity_id  total_nights  total_guests  lead_time
# 0    INN001           1.0           2.0       30.0
# 1    INN002           2.0           3.0       45.0
```

### Configuración

En `configs/config.yaml`:

```yaml
feature_store:
  enabled: true
  db_path: "feature_store.db"    # SQLite local
  auto_store: true               # Almacenar features durante entrenamiento
  auto_retrieve: true            # Usar features almacenadas en inferencia
```

### Operaciones Disponibles

| Método | Descripción |
|--------|-------------|
| `initialize()` | Crea las tablas en la base de datos |
| `register_version()` | Registra una nueva versión de features |
| `store_features()` | Almacena vectores de features |
| `get_features()` | Recupera features para un entity (online) |
| `get_features_batch()` | Recupera features para múltiples entities (offline) |
| `get_missing_entities()` | Encuentra entities sin features almacenadas |
| `list_versions()` | Lista todas las versiones registradas |
| `set_active_version()` | Marca una versión como activa |
| `delete_version()` | Elimina una versión y sus features |
| `get_stats()` | Obtiene estadísticas del store |

### Con Dependency Injection

```python
from src.container import Container

container = Container.from_settings()
store = container.feature_store()
store.initialize()
```

---

## Testing

### Run All Tests

```bash
# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_feature_engineer.py -v
```

### Test Coverage

| Module | Coverage |
|--------|----------|
| `src/features/engineer.py` | Tests for feature creation, target encoding |
| `src/features/store.py` | Tests for Feature Store operations |
| `src/data/validator.py` | Tests for data validation logic |
| `src/api/app.py` | Tests for API endpoints |
| `src/config/settings.py` | Tests for configuration management |
| `src/container.py` | Tests for dependency injection |
| `src/exceptions.py` | Tests for custom exceptions |

### Unit Test Examples

```python
# test_container.py - Testing with Dependency Injection
def test_training_pipeline_with_mocks():
    """Test pipeline with injected mock dependencies."""
    mock_loader = MagicMock(spec=DataLoader)
    container = Container.for_testing(data_loader=mock_loader)
    pipeline = container.training_pipeline()
    assert pipeline.data_loader is mock_loader
```

---

## Software Architecture

### Design Patterns & Principles

This project implements several well-established software engineering patterns. For detailed documentation with academic references, see [`docs/architectural_decisions.md`](docs/architectural_decisions.md).

| Pattern | Implementation | Reference |
|---------|----------------|-----------|
| **Dependency Injection** | `src/container.py` | Fowler (2004), Martin (2017) |
| **Repository Pattern** | `src/data/loader.py` | Fowler (2002) |
| **Service Layer** | `src/pipelines/*.py` | Fowler (2002) |
| **Factory Method** | `Container.training_pipeline()` | GoF (1994) |
| **Configuration as Code** | `src/config/settings.py` | 12-Factor App |

### Key References

- **Martin, R.C.** (2017). *Clean Architecture*. Prentice Hall. ISBN: 978-0134494166
- **Percival, H. & Gregory, B.** (2020). *Architecture Patterns with Python*. O'Reilly. ISBN: 978-1492052203
- **Fowler, M.** (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley. ISBN: 978-0321127426
- **The Twelve-Factor App** - https://12factor.net/

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  FastAPI    │──│  Schemas    │──│  Exception Handlers     │ │
│  │  (app.py)   │  │  (Pydantic) │  │  (Custom Exceptions)    │ │
│  └──────┬──────┘  └─────────────┘  └─────────────────────────┘ │
└─────────┼───────────────────────────────────────────────────────┘
          │ Depends()
┌─────────▼───────────────────────────────────────────────────────┐
│                    Container (DI)                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Settings ─── DataLoader ─── Validator ─── Trainer      │   │
│  │      │            │             │            │          │   │
│  │      └────────────┴─────────────┴────────────┘          │   │
│  │                         │                                │   │
│  │              ┌──────────▼──────────┐                    │   │
│  │              │  TrainingPipeline   │                    │   │
│  │              │  InferencePipeline  │                    │   │
│  │              └─────────────────────┘                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## MLOps Concepts

### Architecture Diagram

See: [`docs/architecture.md`](docs/architecture.md)

Key components:
- **Data Layer**: Data sources, Feature Store, Validation
- **Training Layer**: MLflow Experiment Tracking, Model Registry
- **Serving Layer**: FastAPI, Docker, Kubernetes
- **Monitoring Layer**: Prometheus, Grafana, Drift Detection

### Branching Strategy

See: [`docs/branching_strategy.md`](docs/branching_strategy.md)

Branch types:
- `main` - Production-ready code
- `develop` - Integration branch
- `feature/*` - New features
- `experiment/*` - ML experiments
- `release/*` - Release preparation
- `hotfix/*` - Production fixes

### Model Monitoring (Production)

For production deployment, monitor:
1. **Prediction Latency**: < 100ms (p99)
2. **Error Rate**: < 0.1%
3. **Data Drift**: PSI < 0.2
4. **Model Performance**: F1 > 0.75

---

## CI/CD Pipeline

El proyecto incluye un pipeline de CI/CD automatizado con GitHub Actions que se ejecuta en cada push.

### Pipeline Stages

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Lint     │───▶│  Security   │───▶│    Test     │───▶│    Build    │
│   (black,   │    │  (Bandit)   │    │  (pytest)   │    │  (Docker)   │
│ flake8,isort│    │             │    │ + coverage  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Características

| Feature | Descripción |
|---------|-------------|
| **Linting** | Verifica formato (black), imports (isort), y estilo (flake8) |
| **Security Scan** | Bandit analiza el código buscando vulnerabilidades |
| **Tests + Coverage** | pytest con reporte de cobertura a Codecov |
| **Docker Build** | Construye y valida la imagen en main/develop |
| **Dependency Cache** | Cache de pip para builds más rápidos |

### Triggers

- `push` a: main, develop, feature/*, release/*, hotfix/*
- `pull_request` a: main, develop

### Archivos

```
.github/
└── workflows/
    └── ci.yml    # Pipeline principal
```

---

## Technical Requirements Checklist

| Requirement | Location | Status |
|-------------|----------|--------|
| **Training Pipeline** | `src/pipelines/training.py` | ✅ |
| **Inference Pipeline** | `src/pipelines/inference.py` | ✅ |
| **Unit Tests** | `tests/unit/test_*.py` | ✅ |
| **Model Packaging** | MLflow model registry | ✅ |
| **FastAPI - Single Prediction** | `POST /predict` | ✅ |
| **FastAPI - Batch Prediction** | `POST /predict/batch` | ✅ |
| **Dockerfile** | `Dockerfile` | ✅ |
| **Docker Build Instructions** | See [Docker Deployment](#docker-deployment) | ✅ |
| **Branching Strategy** | `docs/branching_strategy.md` | ✅ |
| **ML Architecture Diagram** | `docs/architecture.md` | ✅ |
| **CI/CD Pipeline** | `.github/workflows/ci.yml` | ✅ |
| **Feature Store** | `src/features/store.py` | ✅ |

---

## Troubleshooting

### El contenedor de MLflow dice "unhealthy"

Espera un poco. MLflow tarda en iniciar porque tiene que crear las tablas de la base de datos. Después de 30-60 segundos debería cambiar a "healthy".

### Error 403 "Invalid Host header - DNS rebinding attack"

Esto ocurre si usas una versión vieja de la imagen de MLflow. Reconstruye sin cache:

```bash
docker-compose down
docker rmi hotel-cancellation-predictor-mlflow
docker-compose up --build -d
```

### El puerto 5000 ya está en uso

En Mac, AirPlay usa el puerto 5000. Puedes desactivarlo en Preferencias del Sistema > General > AirDrop & Handoff, o cambiar el puerto en `docker-compose.yml`.

### Los tests fallan con "Model not found"

Asegúrate de entrenar el modelo primero:
```bash
python -m src.pipelines.training --data-path "data/raw/Hotel Reservations.csv"
```

---

## Author

Harold Lugo - MLOps Portfolio Project

## License

MIT License
