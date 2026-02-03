# Architectural Decision Records (ADRs)

Este documento describe las decisiones arquitectónicas del proyecto y las fuentes que las respaldan.

---

## ADR-001: Dependency Injection Pattern

### Contexto
Los pipelines de ML necesitan múltiples componentes (DataLoader, Preprocessor, Trainer) que deben ser testeables y reemplazables.

### Decisión
Implementar **Dependency Injection (DI)** mediante constructor injection y un Container centralizado.

### Implementación
```python
# src/container.py
class Container:
    def training_pipeline(self):
        return TrainingPipeline(
            data_loader=self.data_loader(),
            data_validator=self.data_validator(),
            # ... dependencies injected
        )

# src/pipelines/training.py
class TrainingPipeline:
    def __init__(
        self,
        data_loader: DataLoader,      # Injected
        data_validator: DataValidator, # Injected
        ...
    ):
        self.data_loader = data_loader
```

### Fuentes

1. **Martin Fowler - "Inversion of Control Containers and the Dependency Injection pattern" (2004)**
   - URL: https://martinfowler.com/articles/injection.html
   - Conceptos aplicados:
     - *Constructor Injection*: "Constructor injection is a way of ensuring that a component gets all its dependencies at construction time."
     - *Service Locator vs DI*: Optamos por DI puro sobre Service Locator para mayor claridad.

2. **Robert C. Martin - "Clean Architecture" (2017), Capítulo 11: The Dependency Inversion Principle**
   - ISBN: 978-0134494166
   - Principio aplicado: "High-level modules should not depend on low-level modules. Both should depend on abstractions."
   - Nuestra implementación: `TrainingPipeline` (high-level) depende de interfaces abstractas, no de implementaciones concretas.

3. **Harry Percival & Bob Gregory - "Architecture Patterns with Python" (2020), Capítulo 12: Dependency Injection**
   - ISBN: 978-1492052203
   - URL: https://www.cosmicpython.com/book/chapter_13_dependency_injection.html
   - Cita: "Dependency injection is a technique whereby one object supplies the dependencies of another object."
   - Patrón aplicado: Bootstrap/Container pattern para crear el grafo de dependencias.

### Beneficios Obtenidos
- **Testabilidad**: Inyección de mocks sin modificar código de producción
- **Desacoplamiento**: Componentes independientes y reemplazables
- **Configurabilidad**: Fácil cambio de implementaciones por ambiente

---

## ADR-002: Centralized Configuration with Pydantic Settings

### Contexto
La configuración estaba dispersa en múltiples lugares (environment variables, hardcoded values, argumentos CLI).

### Decisión
Implementar configuración centralizada usando **Pydantic Settings** con validación de tipos.

### Implementación
```python
# src/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    environment: str = "development"
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
    )
```

### Fuentes

1. **The Twelve-Factor App - Factor III: Config**
   - URL: https://12factor.net/config
   - Principio: "Store config in the environment"
   - Cita: "The twelve-factor app stores config in environment variables... Env vars are easy to change between deploys without changing any code."
   - Nuestra implementación soporta: env vars, `.env` files, y YAML configs.

2. **Pydantic Documentation - Settings Management**
   - URL: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
   - Patrón aplicado: Typed settings with validation
   - Beneficio: "Settings management where values can come from environment variables, secrets files, or other sources."

3. **Harry Percival & Bob Gregory - "Architecture Patterns with Python" (2020), Capítulo 13: Bootstrap**
   - ISBN: 978-1492052203
   - Concepto aplicado: Separación de configuración del código de aplicación
   - Cita: "Configuration should be loaded once at application startup and made available to all components that need it."

### Beneficios Obtenidos
- **Type Safety**: Validación automática de configuración
- **Single Source of Truth**: Una única fuente de configuración
- **Environment Flexibility**: Fácil override por ambiente (dev/staging/prod)

---

## ADR-003: Custom Exception Hierarchy

### Contexto
El manejo de errores usaba excepciones genéricas (`Exception`, `ValueError`) sin distinción de tipos de error.

### Decisión
Crear una jerarquía de excepciones personalizadas específicas del dominio.

### Implementación
```python
# src/exceptions.py
class BaseMLOpsException(Exception):
    def __init__(self, message: str, details: dict = None, original_error: Exception = None):
        self.message = message
        self.details = details or {}
        self.original_error = original_error

    def to_dict(self) -> dict:
        return {"error": self.__class__.__name__, "message": self.message, ...}

class ModelNotLoadedError(ModelError): ...
class PredictionError(ModelError): ...
class DataValidationError(DataError): ...
```

### Fuentes

1. **Robert C. Martin - "Clean Code" (2008), Capítulo 7: Error Handling**
   - ISBN: 978-0132350884
   - Principios aplicados:
     - "Use Exceptions Rather Than Return Codes"
     - "Define Exception Classes in Terms of a Caller's Needs"
     - "Don't Return Null" - usamos excepciones específicas en su lugar

2. **Python Documentation - User-defined Exceptions**
   - URL: https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions
   - Guía: "Exception classes can be defined which do anything any other class can do, but are usually kept simple."
   - Nuestra implementación: Excepciones con `to_dict()` para serialización API.

3. **Microsoft REST API Guidelines - Error Handling**
   - URL: https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#7102-error-condition-responses
   - Patrón aplicado: Structured error responses con `error`, `message`, y `details`.

### Beneficios Obtenidos
- **Claridad**: Errores específicos del dominio (`ModelNotLoadedError` vs genérico `Exception`)
- **API Responses**: Método `to_dict()` para respuestas estructuradas
- **Debugging**: Preservación de `original_error` para stack traces completos

---

## ADR-004: Repository Pattern for Data Access

### Contexto
La carga y validación de datos estaba mezclada con la lógica de negocio.

### Decisión
Implementar **Repository Pattern** con clases dedicadas: `DataLoader` y `DataValidator`.

### Implementación
```python
# src/data/loader.py
class DataLoader:
    """Responsible for loading data from various sources."""

    def load_csv(self, path: str) -> pd.DataFrame: ...
    def load_from_dict(self, data: dict) -> pd.DataFrame: ...

# src/data/validator.py
class DataValidator:
    """Validates data quality and integrity."""

    def validate(self, df: pd.DataFrame) -> ValidationResult: ...
```

### Fuentes

1. **Martin Fowler - "Patterns of Enterprise Application Architecture" (2002), Repository Pattern**
   - ISBN: 978-0321127426
   - URL: https://martinfowler.com/eaaCatalog/repository.html
   - Definición: "A Repository mediates between the domain and data mapping layers, acting like an in-memory domain object collection."
   - Nuestra implementación: `DataLoader` encapsula el acceso a datos CSV/dict.

2. **Harry Percival & Bob Gregory - "Architecture Patterns with Python" (2020), Capítulo 2: Repository Pattern**
   - ISBN: 978-1492052203
   - URL: https://www.cosmicpython.com/book/chapter_02_repository.html
   - Cita: "The Repository pattern is an abstraction over persistent storage."
   - Aplicación: Abstraemos la fuente de datos (CSV, dict, futuro: database).

### Beneficios Obtenidos
- **Single Responsibility**: Cada clase tiene una única responsabilidad
- **Testabilidad**: Fácil mock del DataLoader en tests
- **Extensibilidad**: Fácil agregar nuevas fuentes de datos

---

## ADR-005: Service Layer Pattern for Pipelines

### Contexto
La lógica de entrenamiento e inferencia necesita orquestar múltiples componentes.

### Decisión
Implementar **Service Layer Pattern** con `TrainingPipeline` e `InferencePipeline`.

### Implementación
```python
# src/pipelines/training.py
class TrainingPipeline:
    """Orchestrates the complete training workflow."""

    def run(self) -> Dict[str, Any]:
        df = self.data_loader.load_csv()
        validation = self.data_validator.validate(df)
        df = self.feature_engineer.fit_transform(df)
        X = self.preprocessor.fit_transform(X_df)
        metrics = self.trainer.train(X_train, y_train, ...)
        return {"metrics": metrics, ...}
```

### Fuentes

1. **Martin Fowler - "Patterns of Enterprise Application Architecture" (2002), Service Layer**
   - ISBN: 978-0321127426
   - URL: https://martinfowler.com/eaaCatalog/serviceLayer.html
   - Definición: "Defines an application's boundary with a layer of services that establishes a set of available operations."
   - Nuestra implementación: `TrainingPipeline.run()` es el único punto de entrada para entrenamiento.

2. **Harry Percival & Bob Gregory - "Architecture Patterns with Python" (2020), Capítulo 4: Service Layer**
   - ISBN: 978-1492052203
   - URL: https://www.cosmicpython.com/book/chapter_04_service_layer.html
   - Cita: "The service layer's job is to handle requests from the outside world and to orchestrate an operation."
   - Aplicación: Pipelines orquestan DataLoader → Validator → FeatureEngineer → Trainer.

### Beneficios Obtenidos
- **Orchestration**: Un único punto de entrada para workflows complejos
- **Transaction Boundary**: El pipeline maneja el flujo completo
- **API Simplicity**: La API solo llama `pipeline.predict_single(data)`

---

## ADR-006: Factory Pattern via Container

### Contexto
La creación de objetos con sus dependencias era compleja y repetitiva.

### Decisión
Implementar **Factory Pattern** mediante el `Container` que actúa como factory de pipelines.

### Implementación
```python
# src/container.py
class Container:
    def training_pipeline(self) -> TrainingPipeline:
        """Factory method for TrainingPipeline."""
        return TrainingPipeline(
            data_loader=self.data_loader(),
            data_validator=self.data_validator(),
            feature_engineer=self.feature_engineer(),
            preprocessor=self.preprocessor(),
            model_trainer=self.model_trainer(),
            settings=self.settings,
        )
```

### Fuentes

1. **Gang of Four - "Design Patterns" (1994), Factory Method**
   - ISBN: 978-0201633610
   - Definición: "Define an interface for creating an object, but let subclasses decide which class to instantiate."
   - Aplicación: `Container.training_pipeline()` encapsula la creación compleja.

2. **Harry Percival & Bob Gregory - "Architecture Patterns with Python" (2020), Capítulo 13: Bootstrap**
   - ISBN: 978-1492052203
   - Concepto: "Composition Root" - un lugar donde se ensamblan todas las dependencias.
   - Cita: "The bootstrap script is responsible for creating the object graph."
   - Nuestra implementación: `Container.from_settings()` es nuestro composition root.

### Beneficios Obtenidos
- **Encapsulation**: Lógica de creación centralizada
- **Consistency**: Todas las instancias se crean de la misma manera
- **Configuration**: El factory usa settings para configurar componentes

---

## ADR-007: Pydantic for API Schema Validation

### Contexto
Las APIs necesitan validación robusta de input/output.

### Decisión
Usar **Pydantic Models** para validación de schemas en FastAPI.

### Implementación
```python
# src/api/schemas.py
class ReservationInput(BaseModel):
    no_of_adults: int = Field(..., ge=0)
    lead_time: int = Field(..., ge=0)
    arrival_month: int = Field(..., ge=1, le=12)

    class Config:
        json_schema_extra = {"example": {...}}
```

### Fuentes

1. **FastAPI Documentation - Request Body**
   - URL: https://fastapi.tiangolo.com/tutorial/body/
   - Patrón: "Pydantic models for request/response validation"
   - Beneficio: "Automatic data validation, serialization, and documentation."

2. **Pydantic Documentation - Field Types**
   - URL: https://docs.pydantic.dev/latest/concepts/fields/
   - Aplicación: Field constraints (`ge=0`, `le=12`) para validación de rangos.

### Beneficios Obtenidos
- **Automatic Validation**: Errores 422 automáticos para input inválido
- **Documentation**: OpenAPI schema auto-generado
- **Type Safety**: Conversión automática de tipos

---

## Referencias Bibliográficas Completas

### Libros

| Título | Autor(es) | Año | ISBN |
|--------|-----------|-----|------|
| Clean Architecture | Robert C. Martin | 2017 | 978-0134494166 |
| Clean Code | Robert C. Martin | 2008 | 978-0132350884 |
| Architecture Patterns with Python | Harry Percival & Bob Gregory | 2020 | 978-1492052203 |
| Patterns of Enterprise Application Architecture | Martin Fowler | 2002 | 978-0321127426 |
| Design Patterns | Gang of Four | 1994 | 978-0201633610 |

### Recursos Online

| Recurso | URL |
|---------|-----|
| Martin Fowler - Dependency Injection | https://martinfowler.com/articles/injection.html |
| Martin Fowler - Repository Pattern | https://martinfowler.com/eaaCatalog/repository.html |
| Martin Fowler - Service Layer | https://martinfowler.com/eaaCatalog/serviceLayer.html |
| The Twelve-Factor App | https://12factor.net/ |
| Cosmic Python (libro online) | https://www.cosmicpython.com/book/preface.html |
| Pydantic Settings | https://docs.pydantic.dev/latest/concepts/pydantic_settings/ |
| FastAPI Documentation | https://fastapi.tiangolo.com/ |

---

## Mapeo de Patrones a Código

| Patrón | Archivo | Clase/Función |
|--------|---------|---------------|
| Dependency Injection | `src/container.py` | `Container` |
| Factory Method | `src/container.py` | `Container.training_pipeline()` |
| Repository | `src/data/loader.py` | `DataLoader` |
| Service Layer | `src/pipelines/training.py` | `TrainingPipeline` |
| Settings/Config | `src/config/settings.py` | `Settings` |
| Custom Exceptions | `src/exceptions.py` | `BaseMLOpsException` hierarchy |
