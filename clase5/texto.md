# Validación Cruzada

## Objetivos

- Enteder el balance sesgo y varianza (bias-variance trade-off)
- Esquemas para conocer la fuente del error en un modelo
- Tipos de validacion cruzada y su su practico
- Casos especiales de validación

Tomado de https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote12.html

## Definiciones

Dado un dataset $D = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_n,y_n)\}$ extraído de un distribución $P(X,Y)$. Es importante anotar que dado una entrada $\mathbf{x}$ puede que no exista una única etiqueta $y$. Dos entradas idénticas pueden tener diferentes salidas. Por esto, dado un vector de características $\mathbf{x}$ se tiene una distribución de posibles salidas. Con esto se puede definir la **etiqueta esperada**

### Etiqueta esperada (dado $\mathbf{x} \in \mathbb{R}^d$)

$$
\bar{y}(\mathbf{x}) = E_{y \vert \mathbf{x}} \left[Y\right] = \int\limits_y y \, \Pr(y \vert \mathbf{x}) \partial y.
$$

Después se invoca un algoritmo de ML $\mathcal{A}$ en el dataset para aprender la hipotesis (es decir, el clasificador). Este proceso se denota $h_D = \mathcal{A}(D)$

### Módelo Clasificador Esperado (dado $\mathcal{A}$)

$$
\bar{h} = E_{D \sim P^n} \left[ h_D \right] = \int\limits_D h_D \Pr(D) \partial D
$$

### Error de Generalización (dado $\mathcal{A}$)

$$
E_{\substack{(\mathbf{x},y) \sim P\\ D \sim P^n}} \left[\left(h_{D}(\mathbf{x}) - y\right)^{2}\right] = \int_{D} \int_{\mathbf{x}} \int_{y} \left( h_{D}(\mathbf{x}) - y\right)^{2} \mathrm{P}(\mathbf{x},y) \mathrm{P}(D) \partial \mathbf{x} \partial y \partial D

$$

## Descompoción del Error de Prueba Esperado

Descomponemos, utilizando $\bar{h}_D(\mathbf{x})$

$$
\begin{align*}
	E_{\mathbf{x},y,D}\left[\left[h_{D}(\mathbf{x}) - y\right]^{2}\right] &= E_{\mathbf{x},y,D}\left[\left[\left(h_{D}(\mathbf{x}) - \bar{h}(\mathbf{x})\right) + \left(\bar{h}(\mathbf{x}) - y\right)\right]^{2}\right] \\
    &= E_{\mathbf{x}, D}\left[(\bar{h}_{D}(\mathbf{x}) - \bar{h}(\mathbf{x}))^{2}\right] + 2 \mathrm{\;} E_{\mathbf{x}, y, D} \left[\left(h_{D}(\mathbf{x}) - \bar{h}(\mathbf{x})\right)\left(\bar{h}(\mathbf{x}) - y\right)\right] + E_{\mathbf{x}, y} \left[\left(\bar{h}(\mathbf{x}) - y\right)^{2}\right] 
\end{align*}
$$

El término de la mitad se vuelve cero como se muestra a continuación

$$
\begin{align*}
	E_{\mathbf{x}, y, D} \left[\left(h_{D}(\mathbf{x}) - \bar{h}(\mathbf{x})\right) \left(\bar{h}(\mathbf{x}) - y\right)\right] &= E_{\mathbf{x}, y} \left[E_{D} \left[ h_{D}(\mathbf{x}) - \bar{h}(\mathbf{x})\right] \left(\bar{h}(\mathbf{x}) - y\right) \right] \\
    &= E_{\mathbf{x}, y} \left[ \left( E_{D} \left[ h_{D}(\mathbf{x}) \right] - \bar{h}(\mathbf{x}) \right) \left(\bar{h}(\mathbf{x}) - y \right)\right] \\
    &= E_{\mathbf{x}, y} \left[ \left(\bar{h}(\mathbf{x}) - \bar{h}(\mathbf{x}) \right) \left(\bar{h}(\mathbf{x}) - y \right)\right] \\
    &= E_{\mathbf{x}, y} \left[ 0 \right] \\
    &= 0
\end{align*}
$$

Volviendo a la expresión anterior, quedamos con la varianza y otro termino:

$$
	E_{\mathbf{x}, y, D} \left[ \left( h_{D}(\mathbf{x}) - y \right)^{2} \right] = \underbrace{E_{\mathbf{x}, D} \left[ \left(h_{D}(\mathbf{x}) - \bar{h}(\mathbf{x}) \right)^{2} \right]}_\mathrm{Variance} + E_{\mathbf{x}, y}\left[ \left( \bar{h}(\mathbf{x}) - y \right)^{2} \right]
$$

El segundo término se puede descomponer utilizando $\bar{y}(\mathbf{x})$

$$
\begin{align*}
	E_{\mathbf{x}, y} \left[ \left(\bar{h}(\mathbf{x}) - y \right)^{2}\right] &= E_{\mathbf{x}, y} \left[ \left(\bar{h}(\mathbf{x}) -\bar y(\mathbf{x}) )+(\bar y(\mathbf{x}) - y \right)^{2}\right]  \\
  &=\underbrace{E_{\mathbf{x}, y} \left[\left(\bar{y}(\mathbf{x}) - y\right)^{2}\right]}_\mathrm{Noise} + \underbrace{E_{\mathbf{x}} \left[\left(\bar{h}(\mathbf{x}) - \bar{y}(\mathbf{x})\right)^{2}\right]}_\mathrm{Bias^2} + 2 \mathrm{\;} E_{\mathbf{x}, y} \left[ \left(\bar{h}(\mathbf{x}) - \bar{y}(\mathbf{x})\right)\left(\bar{y}(\mathbf{x}) - y\right)\right]
\end{align*}
$$

El tercer término se vuelve cero:

$$
\begin{align*}
	E_{\mathbf{x}, y} \left[\left(\bar{h}(\mathbf{x}) - \bar{y}(\mathbf{x})\right)\left(\bar{y}(\mathbf{x}) - y\right)\right] &= E_{\mathbf{x}}\left[E_{y \mid \mathbf{x}} \left[\bar{y}(\mathbf{x}) - y \right] \left(\bar{h}(\mathbf{x}) - \bar{y}(\mathbf{x}) \right) \right] \\
    &= E_{\mathbf{x}} \left[ E_{y \mid \mathbf{x}} \left[ \bar{y}(\mathbf{x}) - y\right] \left(\bar{h}(\mathbf{x}) - \bar{y}(\mathbf{x})\right)\right] \\
    &= E_{\mathbf{x}} \left[ \left( \bar{y}(\mathbf{x}) - E_{y \mid \mathbf{x}} \left [ y \right]\right) \left(\bar{h}(\mathbf{x}) - \bar{y}(\mathbf{x})\right)\right] \\
    &= E_{\mathbf{x}} \left[ \left( \bar{y}(\mathbf{x}) - \bar{y}(\mathbf{x}) \right) \left(\bar{h}(\mathbf{x}) - \bar{y}(\mathbf{x})\right)\right] \\
    &= E_{\mathbf{x}} \left[ 0 \right] \\
    &= 0
\end{align*}
$$

Esto nos da la descomposión del error generalizado

$$
	\underbrace{E_{\mathbf{x}, y, D} \left[\left(h_{D}(\mathbf{x}) - y\right)^{2}\right]}_\mathrm{Expected\;Test\;Error} = \underbrace{E_{\mathbf{x}, D}\left[\left(h_{D}(\mathbf{x}) - \bar{h}(\mathbf{x})\right)^{2}\right]}_\mathrm{Variance} + \underbrace{E_{\mathbf{x}, y}\left[\left(\bar{y}(\mathbf{x}) - y\right)^{2}\right]}_\mathrm{Noise} + \underbrace{E_{\mathbf{x}}\left[\left(\bar{h}(\mathbf{x}) - \bar{y}(\mathbf{x})\right)^{2}\right]}_\mathrm{Bias^2}

$$

- **Varianza**: Captura que tanto el clasificador cambia si se entrena con un dataset de entrenamiento diferente. Es una medidad de cuanto "overfitting" presenta el modelo

- **Sesgo**: Es el error inherente al modelo, aun si se entrenara con un dataset infinito. Es el sesgo del modelo a una determinada solución 

- **Ruido**: Error intrínseco de los datos. Es una medida de la ambigüedad debido a la distribución de los datos y la representación de características. Como es un aspecto de los datos, no se puede disminuir.

<p align="center">
  <img src="./biasvariance.png" />
</p>