# Máquinas de Soporte Vectorial (SVM)

- Obtiene resultados adecuados en varios dominios
- SVM clasifica tanto datos linealmente separables como no linealmente separables

## Separación lineal
- Encuentra el hiperplano estable que maximiza la separación (margen)
- Maximiza el margen pero le da prioridad a la clasificación

## Separación no lineal
La idea consiste en aplicar transformaciones a los datos para llevarlos a un espacio de mayor dimensión donde sean linealmente separables. Es decir, realizar un mapeo

$$
\mathbf{x} = (x_1,x_2,x_3,...,x_n)\rightarrow \phi(\mathbf{x})=( \phi_1(\mathbf{x}), \phi_2(\mathbf{x}),\phi_3(\mathbf{x}),...,\phi_m(\mathbf{x}) )
$$

El problema radica en encontrar el máximo margen en el hiper-plano es costoso en el tiempo. Por lo que se escoge $\phi_i(\mathbf{x})$ para que sea la función propia (eigenfunction) de
una función kernel $K(x,y)$.

## Funciones kernel

Una función kernel deben cumplir tres condiciones:
- Continuas
- Simétrica
- Positiva

Ejemplos:
- Kernel lineal $K(\mathbf{x},\mathbf{y})=\mathbf{x}^T\mathbf{y}$
- Kernel polinómico $K(\mathbf{x},\mathbf{y})=(1-\mathbf{x}^T\mathbf{y})^P$
- Kernel Gaussiano $e^{-\gamma||x-y||^2}$

## Minimización del Riesgo Estructural

- SVM tiene la habiliad de separar casi cualquier conjunto de datos
- Sin embargo, al escoger el margen máximo está seleccionando la solución más robusta
- Debido a esto posee buena generalización a pesar de su gran capacidad de separación
- Lo anterior ha sido acuñado bajo el término **minimización del riesgo estructural** (structural risk minimization)

## Ventajas y desventajas

### Ventajas
- Efectivo en espacios con muchas dimensiones;
- Efectivo incluso cuando las dimensiones son mayores a la muestra;
- Usa un subconjunto de los puntos de entrenamiento en la función de
decisión, por lo que es eficiente en memoria;
- Versátil, ya que permite usar múltiples funciones kernel para cambiar la
decisión

### Desventajas
- Si el número de variables es mucho mayor a la muestra, evitar over-fitting es
crucial. Para esto escoja las funciones kernel con cuidado y use un término
de regularización;
- Las SVMs no proveen un estimado de la probabilidad, pero puede ser
calculado.
