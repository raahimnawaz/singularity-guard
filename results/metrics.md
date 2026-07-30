### well-conditioned (no singularity)

| controller | peak &#124;q̇&#124;∞ [rad/s] | saturated [%] | pos RMS [m] | pos max [m] | ori RMS [rad] | accel RMS [rad/s²] | w min |
|---|---|---|---|---|---|---|---|
| naive $J^{-1}$ | 0.67 | 0.0 | 0.00000 | 0.0000 | 0.00000 | 0.5 | 9.39e-01 |
| fixed-$\lambda$ DLS ($\lambda$=0.1) | 0.67 | 0.0 | 0.00116 | 0.0018 | 0.00076 | 0.5 | 9.39e-01 |
| adaptive-$\lambda$ DLS ($w_0$=0.3, $\lambda_{max}$=0.2) | 0.67 | 0.0 | 0.00000 | 0.0000 | 0.00000 | 0.5 | 9.39e-01 |

### skim: approach and retreat

| controller | peak &#124;q̇&#124;∞ [rad/s] | saturated [%] | pos RMS [m] | pos max [m] | ori RMS [rad] | accel RMS [rad/s²] | w min |
|---|---|---|---|---|---|---|---|
| naive $J^{-1}$ | 1.88 | 0.0 | 0.00000 | 0.0000 | 0.00000 | 10.4 | 2.00e-02 |
| fixed-$\lambda$ DLS ($\lambda$=0.1) | 1.26 | 0.0 | 0.00400 | 0.0088 | 0.00374 | 2.1 | 2.54e-01 |
| adaptive-$\lambda$ DLS ($w_0$=0.3, $\lambda_{max}$=0.2) | 1.93 | 0.0 | 0.00087 | 0.0033 | 0.00087 | 3.7 | 1.62e-01 |

### through: exact crossing

| controller | peak &#124;q̇&#124;∞ [rad/s] | saturated [%] | pos RMS [m] | pos max [m] | ori RMS [rad] | accel RMS [rad/s²] | w min |
|---|---|---|---|---|---|---|---|
| naive $J^{-1}$ | 2433.53 | 52.2 | 0.74029 | 1.3347 | 0.54083 | 6.8 | 7.50e-04 |
| fixed-$\lambda$ DLS ($\lambda$=0.1) | 1.86 | 0.0 | 0.10251 | 0.2164 | 0.07302 | 2.5 | 3.90e-08 |
| adaptive-$\lambda$ DLS ($w_0$=0.3, $\lambda_{max}$=0.2) | 2.78 | 0.0 | 0.10363 | 0.2170 | 0.07136 | 4.2 | 9.12e-05 |
