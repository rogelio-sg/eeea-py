# Guía de Contribución para EEEA_PY

Este proyecto busca centralizar algoritmos de metaheurísticas de alta calidad. Para mantener la consistencia y profesionalismo del código, pedimos a todos los colaboradores seguir estas directrices.


## 1. Estándares de Código (PEP 8)

Todo el código debe seguir la guía de estilo oficial de Python: **[PEP 8](https://peps.python.org/pep-0008/)**.

* **Nombres:** * Funciones y variables: `snake_case` (ej. `mi_algoritmo_genetico`).
    * Clases: `PascalCase` (ej. `ParticleSwarmOptimization`).
    * Constantes: `UPPER_CASE` (ej. `MAX_ITERATIONS`).
* **Longitud de línea:** Máximo 79 caracteres para código y 72 para docstrings.


## 2. Documentación y Docstrings (PEP 257)

Para que el paquete sea útil en investigación y docencia, cada módulo, clase y función debe estar documentado siguiendo **[PEP 257](https://peps.python.org/pep-0257/)**.

* Usa comillas triples `"""` para los docstrings.
* Explica claramente los parámetros de entrada y los valores de retorno.

**Ejemplo de formato:**
```python
def optimize(objective_function, bounds):
    """
    Ejecuta el proceso de optimización para una función dada.

    Args:
        objective_function (callable): La función matemática a minimizar.
        bounds (list): Lista de tuplas con los límites (min, max) por dimensión.

    Returns:
        dict: Un diccionario con la mejor posición y el mejor valor de fitness.
    """
    pass
```


## 3. Registro de Cambios (CHANGELOG.md)
Cualquier modificación al proyecto debe registrarse en el archivo CHANGELOG.md antes de realizar el commit o merge. Los cambios deben agruparse bajo los siguientes encabezados:

* Added: Para nuevas funciones o algoritmos.
* Changed: Para cambios en lógica existente.
* Fixed: Para corrección de errores (bugs).
* Removed: Para funcionalidades eliminadas.


## 4. Flujo de Trabajo
* Ramas (Branches): Crea una rama nueva para cada contribución:

    * ```feat/nombre-del-algoritmo```
    * ```fix/descripcion-del-error```

* Pruebas: Asegúrate de que tu código no rompa los módulos existentes. Si agregas un algoritmo nuevo, incluye su prueba unitaria en la carpeta test/.

* Pull Requests: Al enviar tus cambios, describe brevemente la metaheurística o mejora implementada y adjunta resultados de prueba si es posible.

---

Agradecemos tu esfuerzo para hacer de EEEA_PY una herramienta robusta para la optimización en Python.