# Directiva de desarrollo academico

Este repositorio contiene desarrollos y experimentos de investigacion para refinamiento y comparacion de modelos de elevacion.

## Trazabilidad obligatoria

- Documentar cada desarrollo con una descripcion clara del objetivo, supuestos, entradas, salidas y parametros relevantes.
- Registrar cada experimento en Markdown y JSON, incluyendo tiempos de ejecucion, errores, advertencias, versiones de dependencias cuando sea relevante y rutas de artefactos generados.
- Mantener los scripts reproducibles mediante argumentos de linea de comandos y valores por defecto explicitos.
- No procesar subcarpetas salvo que el experimento lo declare expresamente.
- Evitar cambios destructivos sobre los datos originales; todas las salidas deben escribirse en carpetas de resultados del workspace.

## Datos geoespaciales

- Indicar siempre el CRS de entrada asumido o detectado y el CRS de salida.
- Cuando el CRS no este embebido en los datos, documentar la inferencia usada.
- Documentar la resolucion de rasterizacion y la regla de agregacion por pixel.
- Registrar metricas de comparacion, mascara de validez, numero de muestras y formula empleada.

## Visualizacion

- Los visores y artefactos interactivos deben poder regenerarse desde datos intermedios documentados.
- Las simplificaciones o downsampling para visualizacion deben registrarse separadamente de los calculos metricos.
