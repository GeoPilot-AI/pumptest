# Theory

Every solution here describes radial flow to a well in a horizontal aquifer
of infinite extent, homogeneous within each layer, initially at rest. All are
linear in the pumping rate, which is what lets a pumping schedule be handled
by superposition. Notation: drawdown \(s\), rate \(Q\), transmissivity \(T\),
storativity \(S\), radial distance \(r\), time \(t\), aquifer thickness \(b\).

## Theis (1935): confined aquifer

A line sink in a confined aquifer:

\[
s(r,t) = \frac{Q}{4\pi T}\,W(u), \qquad u = \frac{r^2 S}{4 T t}, \qquad W(u) = E_1(u) = \int_u^\infty \frac{e^{-y}}{y}\,dy .
\]

\(W\) is evaluated directly as the exponential integral.

## Cooper and Jacob (1946): the straight line

For small \(u\), \(W(u) \approx -0.5772 - \ln u\), so drawdown is linear in
\(\log_{10} t\):

\[
s = \frac{2.30\,Q}{4\pi T}\log_{10}\frac{2.25\,T t}{r^2 S}.
\]

\(T\) follows from the slope per log cycle and \(S\) from the time \(t_0\)
where the line crosses zero drawdown. `cooper_jacob` walks the start of the
line later until every point used has \(u \le 0.05\).

## Hantush and Jacob (1955): leaky aquifer

Leakage through an aquitard without storage, with leakage factor
\(B = \sqrt{T c}\) and aquitard resistance \(c = b'/K'\):

\[
s = \frac{Q}{4\pi T}\,W\!\left(u, \frac{r}{B}\right), \qquad
W(u,\rho) = \int_u^\infty \frac{1}{y}\exp\!\left(-y - \frac{\rho^2}{4y}\right) dy .
\]

## Papadopulos and Cooper (1967): large-diameter well, with skin

A well of screen radius \(r_w\) whose water level falls in a casing of radius
\(r_c\), so part of the early discharge comes from the casing. With a skin
factor \(S_k\) (Agarwal et al., 1970), the Laplace transform of drawdown per
unit rate is

\[
\bar s(r,p) = \frac{K_0(q r)}{p\left[2\pi T\, r_w q K_1(q r_w) + \pi r_c^2\, p \left(K_0(q r_w) + S_k\, r_w q K_1(q r_w)\right)\right]},
\qquad q = \sqrt{\frac{S p}{T} + \frac{1}{B^2}},
\]

and in the pumped well \(K_0(q r)\) is replaced by \(K_0(q r_w) + S_k\, r_w q K_1(q r_w)\).
With \(r_c \to 0\), \(S_k = 0\) and \(B \to \infty\) this is the Laplace
transform of the Theis solution. One kernel therefore covers confined and
leaky aquifers, wellbore storage and skin.

## Neuman (1974): unconfined aquifer, delayed yield

A fully penetrating well in an anisotropic unconfined aquifer
(\(K_d = K_z/K_r\)) whose water table releases specific yield \(S_y\)
instantly. Separating variables in the vertical gives eigenvalues
\(\varepsilon_n\), the roots of

\[
\varepsilon \tan\varepsilon = \frac{p\, S_y\, b}{K_z},
\]

and the Laplace transform of drawdown per unit rate

\[
\bar s(r,z,p) = \frac{1}{2\pi T p}\sum_{n\ge 0} a_n\, K_0\!\left(\sqrt{\frac{r^2 S p}{T} + \beta\varepsilon_n^2}\right)\phi_n(z),
\qquad a_n = \frac{2\sin\varepsilon_n}{\varepsilon_n + \sin\varepsilon_n\cos\varepsilon_n},
\qquad \beta = \frac{K_d r^2}{b^2},
\]

where \(\phi_n\) is \(\cos(\varepsilon_n z / b)\) averaged over the
observation screen (\(\sin\varepsilon_n/\varepsilon_n\) for a fully
penetrating well). Two limits pin the derivation, and both are tested: as
\(S_y \to 0\) the series collapses to Theis with \(S\), and at late time to
Theis with \(S + S_y\). The roots are found by bracketed Newton iteration,
vectorised over all terms and all Laplace variables at once; the series is
summed to at most 400 terms.

## From Laplace to time: Stehfest (1970)

\[
s(t) \approx \frac{\ln 2}{t}\sum_{i=1}^{N} V_i\, \bar s\!\left(\frac{i \ln 2}{t}\right),
\]

with \(N = 16\). This holds Theis to \(4\times10^{-5}\) relative wherever
\(u \le 2\), which covers every drawdown large enough to measure.

## Superposition

For a schedule of rates \(Q_k\) starting at \(t_k\) and a drawdown per unit
rate \(s_1(r,t)\),

\[
s(r,t) = \sum_k (Q_k - Q_{k-1})\, s_1(r, t - t_k)\,\mathbb{1}[t > t_k],
\]

so recovery is a step to zero and a step test is a list of rising steps.

## Fitting

Parameters are estimated by bounded nonlinear least squares (scipy's
trust-region reflective method) on the residual between measured and computed
drawdown, over all observations at once. Positive parameters are estimated as
\(\log_{10}\) values, so one optimiser handles a transmissivity of 10⁻³ and a
storativity of 10⁻⁶ in the same problem. Several starting points (from a
Cooper-Jacob line and a spread over the leakage, specific yield, anisotropy
and skin parameters) are tried and the lowest residual wins.

The finite-difference Jacobian uses a relative step of 10⁻⁴: Stehfest
round-off sits near 10⁻⁷ relative, and scipy's default step falls beneath it,
which stops the optimiser on its first iteration.

**Uncertainty.** With Jacobian \(J\) in the fitted coordinates, residual sum of
squares \(\mathrm{SSR}\), \(n\) points and \(k\) parameters,
\(\mathrm{Cov} = (J^\top J)^{-1}\,\mathrm{SSR}/(n-k)\). Standard errors are
reported in each parameter's own units (for a log-scaled parameter
\(\sigma_p = p \ln 10\, \sigma_{\log_{10} p}\)), with the correlation matrix.

**Model choice.** \(\mathrm{AIC} = n \ln(\mathrm{SSR}/n) + 2k\) and
\(\mathrm{AIC_c} = \mathrm{AIC} + 2k(k+1)/(n-k-1)\).

## Step-drawdown well losses

Jacob's (1947) relation \(s_w = BQ + CQ^2\) splits the drawdown in the
pumped well into an aquifer loss and a nonlinear well loss. `step_drawdown`
fits \(B\) and \(C\) by linear least squares on drawdown and reports the well
efficiency \(BQ/(BQ + CQ^2)\) at each step.

## Diagnostics

The Bourdet et al. (1989) derivative \(ds/d\ln t\), each point a weighted mean
of the slopes to neighbours at least a chosen log-time distance away.

## Not modelled yet

Partial penetration, aquitard storage (Hantush 1960; Neuman and Witherspoon),
Moench's gradual drainage, double-porosity and fractured media, barrier and
recharge boundaries, and slug tests. Nothing in the package does them today.

## References

- Agarwal, R.G., Al-Hussainy, R. and Ramey, H.J. (1970). An investigation of wellbore storage and skin effect in unsteady liquid flow. *SPE Journal* 10(3), 279-290.
- Bourdet, D., Ayoub, J.A. and Pirard, Y.M. (1989). Use of pressure derivative in well-test interpretation. *SPE Formation Evaluation* 4(2), 293-302.
- Cooper, H.H. and Jacob, C.E. (1946). A generalized graphical method for evaluating formation constants and summarizing well-field history. *Trans. AGU* 27(4), 526-534.
- Hantush, M.S. and Jacob, C.E. (1955). Non-steady radial flow in an infinite leaky aquifer. *Trans. AGU* 36(1), 95-100.
- Jacob, C.E. (1947). Drawdown test to determine effective radius of artesian well. *Trans. ASCE* 112, 1047-1064.
- Neuman, S.P. (1974). Effect of partial penetration on flow in unconfined aquifers considering delayed gravity response. *Water Resources Research* 10(2), 303-312.
- Papadopulos, I.S. and Cooper, H.H. (1967). Drawdown in a well of large diameter. *Water Resources Research* 3(1), 241-244.
- Stehfest, H. (1970). Algorithm 368: Numerical inversion of Laplace transforms. *Communications of the ACM* 13(1), 47-49.
- Theis, C.V. (1935). The relation between the lowering of the piezometric surface and the rate and duration of discharge of a well using ground-water storage. *Trans. AGU* 16(2), 519-524.
