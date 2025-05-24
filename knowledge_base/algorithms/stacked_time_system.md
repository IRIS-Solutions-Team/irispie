# Linear stacked-time system

$$
\newcommand{\xX}{\widetilde{x}}
\newcommand{\xY}{\widetilde{y}}
\newcommand{\xU}{\widetilde{u}}
\newcommand{\xV}{\widetilde{v}}
\newcommand{\xT}{\widetilde{T}}
\newcommand{\xP}{\widetilde{P}}
\newcommand{\xR}{\widetilde{R}}
\newcommand{\xK}{\widetilde{K}}
\newcommand{\xA}{\widetilde{A}}
\newcommand{\xB}{\widetilde{B}}
\newcommand{\xC}{\widetilde{C}}
\newcommand{\xA}{\widetilde{A}}
\newcommand{\xH}{\widetilde{H}}
\newcommand{\xW}{\widetilde{w}}
$$
## Final form of the stacked-time system

$$
\begin{gathered}
\xX = \xT \ x_0 + \xP \ \xU + \xR \ \xV + \xK \\[10pt]
\xY = \xA \ x_0 + \xB \ \xU + \xC \ \xV + \xH \ \xW + \xD 
\end{gathered}
$$
where
* $\xX$ is a stacked vector of selected transition variables,
* $\xY$ is a stacked vector of selected measurement variables,
* $x_0$ is the initial condition of the original recursive system,
* $\xU$ is a stacked vector of selected unanticipated shocks,
* $\xV$ is a stacked vector of selected anticipated shocks,
* $\xW$ is a stacked vector of selected measurement shocks,
* $\xT$, $\xP$, $\xR$, $\xK$, $\xA$, $\xB$, $\xC$, $\xH$, $\xD$ are the stacked-time impact matrices derived from the original recursive system.
