
# Detach stable dynamics from unit-root dynamics

$$
\newcommand{\t}[1]{\hat{#1}}
\newcommand{\SS}{\Sigma}
\newcommand{\TT}{\Gamma}
$$

## Original state space

$$
\begin{gathered}
x_t =  T \, x_{t-1} + k + R \, v_t  \\[10pt]
y_t = Z \, x_t + d + H \, w_t
\end{gathered}
$$

## Schur decomposition

$$
\begin{gathered}
(I, T) \longrightarrow (\SS, \TT, Q', U) \\[10pt]
I = Q' \, \SS  \, U' \\[10pt]
T = Q' \, \TT  \, U' \\[10pt]
Q \, Q' = I \\[10pt]
U \, U' = I \\[10pt]
\end{gathered}
$$

## Create transformed state space

#### Transition equation

Substitute for $I$ and $T$
$$
Q' \, \SS \, U' \, x_t =  Q' \, \TT  \, U' \, x_{t-1} + k + R \, v_t  \\[10pt]
$$

Premultiply by $Q$, premultiply by $\SS^{-1}$
$$
U'\,x_t = \SS ^{-1}\,\TT\,U'\,x_{t-1} + \SS ^{-1}\,Q\,k + \SS ^{-1}\,Q\,R\,v_t
$$

Introduce transformed vector and matrices
$$
\t x_t = \t T\,\t x_{t-1} + \t k + \t R\,e_t
$$

where
$$
\begin{gathered}
\t x \equiv U' \, x_t \quad \text{or} \quad x_t \equiv U \, \t x_t \\[10pt]
\t T \equiv \SS^{-1}\,\TT\\[10pt]
\t k \equiv \SS^{-1}\,Q\, k \\[10pt]
\t R \equiv \SS^{-1}\,Q\, R
\end{gathered}
$$



#### Measurement equation

Subsitute $x_t \equiv U\, \t x_t$
$$
y_t = \t Z \, \t x_t + d + H \, w_t
$$
where $\t Z \equiv Z \, U$

