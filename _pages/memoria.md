---
layout: category
author_profile: true
---

## [Memoria](https://github.com/felix947/Memoria)
En este repositorio publique los 2 modelos de portafolio que ocupe para mi memoria de titulo, para realizar las predicciones ocupamos SVR, Random Forest Regressor y una red neuronal LSTM:
- [Programacion dinamica estocastica discreta](https://github.com/felix947/Memoria/blob/main/programacio%CC%81n_dinamica.ipynb): Este modelo toma en cuenta el riesgo del individuo, la informacion hasta momento t y ademas el target objetivo que uno persigue al momento de invertir en distintos fondos.
    El modelo se presenta en tiempo discreto y supondremos que la cartera se reasigna cada año entre los 2 activos dependiendo de la historia pasada de los rendimientos del mercado y del tamaño actual del fondo, que se compara con un objetivo a priori. Supondremos que no hay incrementos salariales reales y que por simplicidad el salario es 1 cada año.
    
    Luego el fondo al tiempo $t+1$ es dada por la ecuación
    \begin{equation}
        f_{t+1}=(f_{t}+c)(\sum_{i=1}^{n-1}y_t^{i}(e^{X_t^{i}}-e^{X_t^{n}})+e^{X_t^{n}})
    \end{equation}
    
    Donde $f_t$ es el fondo al tiempo $t$, $c$ es la tasa de contribución , $y_t^{i}$ es la proporción del fondo invertido en el i-esimo activo durante $[t,t+1]$, $X_t^{i}$ es la fuerza de interés para el i-esimo activo en el año $[t,t+1]$ que se asume constante en el año $[t,t+1]$.
    
    Las secuencias $\{X_t^{i}\}$ son IID (asumidas), es decir independientes e identicamente distribuidas con distribución Normal, mientras que la estructura de correlación
para las fuerzas anuales de interés $X_t^{i}$ y $X_t^{j}$ viene dada por la matriz de varianza-covarianza, que se supone constante para cualquier $t$.
    $X_t^{i}\stackrel{}{\sim} N(\mu_i,\sigma_i^2)$
    
    Donde asumiremos sin pérdida de generalidad que
    $\mu_1>\mu_2>...>\mu_n$
    $\sigma_1^2>\sigma_2^2>...>\sigma_n^2$
    
    \textbf{Formulación del problema}: Definimos la función de costo al tiempo $t$
    \begin{equation}
        C_t=(F_t-f_t)^2+\alpha(F_t-f_t)
    \end{equation}
    $t=0,1,...,N-1$
    \begin{equation}
        C_N=\theta[(F_N-f_N)^2+\alpha(F_N-f_N)]
    \end{equation}
    $t=N$
    
    Al variar el parámetro $\alpha$, en realidad estamos considerando diferentes funciones de desutilidad con diferentes factores de aversión al riesgo, de modo que estamos considerando individuos con diferentes perfiles de riesgo. De hecho, puede demostrarse \cite{Pratt} que la aversión al riesgo de un individuo puede medirse bien por $A(x) = -u''(x)/u'(x)$ si $u(x)$ es la función de utilidad del individuo, o por $ A(z) = l''(z)/l'(z)$ si $l(z)$ es la función de desutilidad (o pérdida) del individuo, donde la relación entre las funciones de utilidad y desutilidad es $u(x) = -al(z) + b$ $(a > 0)$ y la relación entre pérdida y ganancia es $x + z = $constante.
    
    La función de desutilidad considerada aquí es $l(z) = z^2 + \alpha z$ (siendo la pérdida $z=F_t - f_t$ ), y por tanto la aversión al riesgo resultante es:
    
    \begin{equation}
        A(z)=\frac{l''(x)}{l'(x)}=\frac{2}{\alpha+2z}
    \end{equation}
    
    Con $\alpha\geq 0$ y $\theta\geq 1$ donde $F_t$ es el objetivo anual del fondo en el momento.
    Observamos que para nuestro modelo si $\alpha \rightarrow 0$ es decir para individuos muy aversos al riesgo, el objetivo tiende a $F_t$. Por otro lado si $\alpha\rightarrow \infty$ para individuos neutrales al riesgo, el target tiende a infinito, lo que significa que el individuo quiere ganar la mayor cantidad de dinero posible con tasta e rendimiento más altas de lo habitual. Por lo tanto $F_t$ e $\infty$ son límites inferior y superior de nuestros objetivos reales cuando consideremos la dependencia entre ellos y la aversión al riesgo.
    
    El costo en el momento $N$ tiene un peso de $\theta$ que puede ser superior a 1. Cuando $\theta$ es superior a 1 se da mas importancia al objetivo final que a los anuales, la justificación de esta elección es que no se contemplan decrementos distintos a la jubilación considerados en nuestro modelo, por lo que la busqueda del objetivo final (el momento de jubilación) se puede considerar más importante que el objetivo anual (antes de la jubilación). El costo futuro tatal en el tiempo $t$ se obtiene descontando los costos futuros hasta $N$, utilizando un factor de descuento interporal subjetivo $\beta$:
    \begin{equation}
        G_t=\sum_{s=t}^{N}\beta^{s-t}C_s
    \end{equation}
    
    Tambien definimos la sigma álgebra de toda la información disponible al tiempo $t$

    \begin{equation}
        J_t=\sigma(f_0,f_1,...,f_t,y_0,....,y_{t-1})
    \end{equation}
    
    Para $t=0,1,...,N$, con $ 
    J_0=\sigma (f_0)$
    siendo $f_0$ el tamaño del fondo cuando el afiliado ingresa al esquema que puede ser 0 o mayor que 0 si existe un valor de transferencia. La función de valor al tiempo $t$ es la siguiente
    \begin{equation}
        J(J_t)=\min_{\pi_t}\mathbb{E}(G_t|J_t)
    \end{equation}
    
    Donde $\pi_t$ es el conjunto de futuras inversiones, para el caso de 2 activos tendremos
    $\pi_t=\{y_1,...,y_n\}$
    
    Principio de optimalidad de Bellman: Aplicando el principio de optimalidad de Bellman obtenemos
    \begin{equation}
        J(J_t)=\min_{\pi_t}\mathbb{E}[\sum_{s=t}^N \beta^{s-t}C_s|J_t]=\min_{y_t}[C_t+\beta\mathbb{E}(J(J_{t+1}|J_t)]
    \end{equation}
    
    Como las secuencias $\{X_t^{i}\}$ son asumidas independientes para cada $t$, $\{f_t\}$ es Cadena de Markov y por lo tanto
    \begin{equation}
    \mathbb{P}(f_{t+1}|J_t)=\mathbb{P}(f_{t+1}|f_t)
    \end{equation}
    \begin{equation}
    \mathbb{P}(f_{t+1},f_{t+2},...,f_N|J_t)=\mathbb{P}(f_{t+1},f_{t+2},...,f_N|f_t)
    \end{equation}
    Luego
    \begin{equation}
    \mathbb{P}(G_t|J_t)=\mathbb{P}(G_t|f_t)
    \end{equation}
    y además
    \begin{equation}
    J(J_t)=\min_{\pi_t}\mathbb{E}(G_t|J_t)=\min_{y_t}\mathbb{E}(G_t|f_t)=J(f_t,t)
    \end{equation}
    Luego el problema de programación dinámica se transforma en
    \begin{equation}
        J(f_t,t)=\min_{y_t}[(F_t-f_t)^2+\alpha(F_t-f_t)+\beta\mathbb{E}(J(f_{t+1},t+1)|f_t)]
    \end{equation}
    
    Con condición final
    \begin{equation}
        J(f_N,N)=C_N=\theta[(F_N-f_N)^2+\alpha(F_N-f_N)]
    \end{equation}
    con $F_{t=1,...,N}$
- [Asset liability managment](https://github.com/felix947/Memoria/blob/main/ALM.ipynb): Este modelo tambien toma en cuenta el riesgo del inviduo como la informacion a tiempo t, a diferencia del anterior este modelo es una serie de problemas de optimizacion en donde se resuelve un problema dual al del Markowitz.

	El problema esta descrito en las siguientes ecuaciones, donde ocupamos gurobipy para resolverlo:
    \begin{equation}
	\max \mathbb{E}(R_p)-\gamma\operatorname{Var}(R_p)
    \end{equation}
	Sujeto a 
    \begin{equation}
	\sum_{i=1}^2x_i=1
    \end{equation}
    \begin{equation}
	\mathbb{E}(R_p)\geq 0.05
    \end{equation}
	Tenemos que 
    \begin{equation}
	\mathbb{E}(R_p)=\mathbb{E}(x_te^{\lambda_t}+(1-x_t)e^{\mu_t})=x_te^{\lambda+\sigma_1^2/2}+(1-x_t)e^{\mu+\sigma_2^2/2}
    \end{equation}
	Por otro lado 
    \begin{equation}
	\operatorname{Var}(R_p)=\operatorname{Var}(x_te^{\lambda_t}+(1-x_t)e^{\mu_t})=x_t^2\operatorname{Var}(e^{\lambda_t})+(1-x_t)^2\operatorname{Var}(e^{\mu_t})+2x_t(1-x_t) \operatorname{Cov}(e^{\lambda_t},e^{\mu_t})
    \end{equation}
	Donde
    \begin{equation}
	\operatorname{Var}(e^{\lambda_t})=[e^{\sigma_1^2}-1]e^{2\lambda+\sigma_1^2}
	\end{equation}
    \begin{equation}
    \operatorname{Var}(e^{\mu_t})=[e^{\sigma_2^2}-1]e^{2\mu+\sigma_2^2}
	\end{equation}
    \begin{equation}
    \operatorname{Cov}(e^{\lambda_t},e^{\mu_t})=e^{(\mu+\lambda)+(\sigma_1^2+\sigma_2^2)/2}[e^{\sigma_1\sigma_2\rho}-1]
    \end{equation}
	Por lo tanto la funcion a maximizar es la siguiente 
    \begin{equation}
	x_te^{\lambda+\sigma_1^2/2}+(1-x_t)e^{\mu+\sigma_2^2/2}-\gamma[x_t^2[e^{\sigma_1^2}-1]e^{2\lambda+\sigma_1^2}+(1-x_t)^2[e^{\sigma_2^2}-1]e^{2\mu+\sigma_2^2}+2x_t(1-x_t)e^{(\mu+\lambda)+(\sigma_1^2+\sigma_2^2)/2}[e^{\sigma_1\sigma_2\rho}-1]]
    \end{equation}
	La restriccion es 
    \begin{equation}
	x_te^{\lambda+\sigma_1^2/2}+(1-x_t)e^{\mu+\sigma_2^2/2}\geq e^{0.05}
    \end{equation}
	La variable $\gamma$ nos entrega la aversion al riesgo del individuo, mientras mas cercano a 0 es el valor de gamma tendremos mayor neutralidad al riesgo, pues estamos ignorando la desviacion estandar y solo concentrandonos en el valor esperado

	Modelo ALM considerando un pasivo como la contribucion (10%) del fondo fijo.
    \begin{equation}
	\max \mathbb{E}(R_p-R_L)-\gamma\operatorname{Var}(R_P-R_L)
    \end{equation}
	Sujeto a 
    \begin{equation}
	\sum_{i=1}^2x_i=1
    \end{equation}
    \begin{equation}
	\mathbb{E}(R_p-R_L)\geq e^{0.05}
    \end{equation}