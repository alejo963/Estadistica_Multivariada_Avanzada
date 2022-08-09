# Clase 2

## Regresión

Quiero encontrar una relación $y = f(x)$, o en términos probabilísticos

$$
P(x|y) = \frac{P(y|x)P(x)}{E(P(y|x))}

\quad E(P(y|x)) = \hat{y}
$$
donde $y$ es una variable continua

### Regresion Lineal

- Ajustar una línea a las observaciones
- Usar esta línea para rpedecir valores no observados
- Usualmente se encuentra con el método de **mínimos cuadrados**
- Si la relación no es lineal, se puede **procesar las entradas** para reflejar esta relación

Supuesto del modelo: una relación lineal
$$
y=f(x)=\omega_0 + \sum^d_{j=1}\omega_j\times x_j
$$
 Al predecir
$$
\hat{y}=\hat{f}(x)=\omega_0 + \sum^d_{j=1}\omega_j\times x_j
$$
De forma matricial
$$
\hat{y}=\hat{f}(x)={X}^TW
$$

### MLE (Estimador de Máxima Verosimilitud)

$$
y_i\in R, \quad y_i=wx_i+\epsilon \sim N(0,\sigma^2)
$$
$$
y\sim N(wx,\sigma^2)
$$

Verosimilitud es
$$
\max_w \prod_{i=1}^2P(y_i|\overrightarrow{x_i},w)=\max_w\sum_{i=1}^n\ln P(y_i|\overrightarrow{x_i},w)
$$

$$
\max_w\sum_{i=1}^n\ln \left(\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(\overrightarrow{x_i}w-y_i)^2}{2\sigma^2} } \right)
$$

$$
\max_w\sum-\left(\overrightarrow{x_i}w-y_i^2 \right)
$$

$$
\min_w\sum\left( x_iw-y_i \right)^2
$$