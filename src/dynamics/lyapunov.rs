use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Validate embedding params and adjust if needed.
/// Returns (adjusted_dim, adjusted_tau). Returns (0, 0) if invalid.
fn validate_embedding(n: usize, dim: usize, tau: usize, min_points: usize) -> (usize, usize) {
    let n_points = n.saturating_sub((dim - 1) * tau);
    if n_points >= min_points {
        return (dim, tau);
    }

    for new_dim in (2..dim).rev() {
        let np = n.saturating_sub((new_dim - 1) * tau);
        if np >= min_points {
            return (new_dim, tau);
        }
    }

    for new_tau in (1..tau).rev() {
        let np = n.saturating_sub((dim - 1) * new_tau);
        if np >= min_points {
            return (dim, new_tau);
        }
    }

    for new_dim in (2..dim).rev() {
        for new_tau in (1..tau).rev() {
            let np = n.saturating_sub((new_dim - 1) * new_tau);
            if np >= min_points {
                return (new_dim, new_tau);
            }
        }
    }

    (0, 0)
}

/// Rosenstein's method for largest Lyapunov exponent.
///
/// Returns (lambda_max, divergence_curve, iterations).
#[pyfunction]
#[pyo3(signature = (signal, dimension=None, delay=None, min_tsep=None, max_iter=None))]
pub fn lyapunov_rosenstein<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dimension: Option<usize>,
    delay: Option<usize>,
    min_tsep: Option<usize>,
    max_iter: Option<usize>,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let raw = signal.as_slice()?;
    let y: Vec<f64> = raw.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = y.len();

    if n < 50 {
        return Ok((
            f64::NAN,
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
        ));
    }

    let dim = dimension.unwrap_or(3);
    let tau = delay.unwrap_or(1);

    let min_embed_points = 50usize.max(n / 4);
    let (dim, tau) = validate_embedding(n, dim, tau, min_embed_points);
    if dim == 0 {
        return Ok((
            f64::NAN,
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
        ));
    }

    let tsep = min_tsep.unwrap_or(tau * dim);
    let mut max_it = max_iter.unwrap_or((n / 10).min(500));

    let n_points = n - (dim - 1) * tau;
    if n_points == 0 {
        return Ok((
            f64::NAN,
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
        ));
    }

    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..dim).map(|d| y[i + d * tau]).collect())
        .collect();

    if n_points < tsep + max_it + 10 {
        max_it = if n_points > tsep + 10 {
            n_points - tsep - 10
        } else {
            0
        };
        max_it = max_it.max(10);
        if max_it < 10 || n_points < tsep + max_it + 10 {
            return Ok((
                f64::NAN,
                PyArray1::from_vec(py, vec![]),
                PyArray1::from_vec(py, vec![]),
            ));
        }
    }

    // Find nearest neighbors (brute force)
    let mut nn_indices: Vec<i64> = vec![-1; n_points];

    for i in 0..n_points {
        let mut best_dist = f64::INFINITY;
        let mut best_j: i64 = -1;

        for j in 0..n_points {
            if (i as i64 - j as i64).unsigned_abs() < tsep as u64 {
                continue;
            }
            let dist: f64 = embedded[i]
                .iter()
                .zip(embedded[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist > 0.0 && dist < best_dist {
                best_dist = dist;
                best_j = j as i64;
            }
        }

        nn_indices[i] = best_j;
    }

    // Track divergence
    let mut divergence = vec![0.0f64; max_it];
    let mut counts = vec![0u64; max_it];
    let track_end = n_points.saturating_sub(max_it);

    for i in 0..track_end {
        let j = nn_indices[i];
        if j < 0 || (j as usize) >= track_end {
            continue;
        }
        let j = j as usize;

        for k in 0..max_it {
            let ii = i + k;
            let jj = j + k;
            if ii >= n_points || jj >= n_points {
                break;
            }
            let dist: f64 = embedded[ii]
                .iter()
                .zip(embedded[jj].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist > 0.0 {
                divergence[k] += dist.ln();
                counts[k] += 1;
            }
        }
    }

    for k in 0..max_it {
        if counts[k] > 0 {
            divergence[k] /= counts[k] as f64;
        } else {
            divergence[k] = f64::NAN;
        }
    }

    let iterations: Vec<f64> = (0..max_it).map(|k| k as f64).collect();

    // Fit slope to initial linear region
    let fit_end = 10usize.max(max_it / 5).min(max_it);

    let mut x_fit: Vec<f64> = Vec::new();
    let mut y_fit: Vec<f64> = Vec::new();
    for k in 0..fit_end {
        if divergence[k].is_finite() {
            x_fit.push(k as f64);
            y_fit.push(divergence[k]);
        }
    }

    if x_fit.len() < 3 {
        return Ok((
            f64::NAN,
            PyArray1::from_vec(py, divergence),
            PyArray1::from_vec(py, iterations),
        ));
    }

    let n_pts = x_fit.len() as f64;
    let sum_x: f64 = x_fit.iter().sum();
    let sum_y: f64 = y_fit.iter().sum();
    let sum_xy: f64 = x_fit.iter().zip(y_fit.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = x_fit.iter().map(|x| x * x).sum();
    let denom = n_pts * sum_xx - sum_x * sum_x;

    let lambda_max = if denom.abs() > 1e-15 {
        let slope = (n_pts * sum_xy - sum_x * sum_y) / denom;
        slope / tau as f64
    } else {
        f64::NAN
    };

    Ok((
        lambda_max,
        PyArray1::from_vec(py, divergence),
        PyArray1::from_vec(py, iterations),
    ))
}

/// Kantz's method for largest Lyapunov exponent.
///
/// Returns (max_lyapunov, divergence_curve).
#[pyfunction]
#[pyo3(signature = (signal, dimension=None, delay=None, min_tsep=None, epsilon=None, max_iter=None))]
pub fn lyapunov_kantz<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dimension: Option<usize>,
    delay: Option<usize>,
    min_tsep: Option<usize>,
    epsilon: Option<f64>,
    max_iter: Option<usize>,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    let y = signal.as_slice()?;
    let n = y.len();

    let dim = dimension.unwrap_or(3);
    let tau = delay.unwrap_or(1);
    let tsep = min_tsep.unwrap_or(dim * tau);

    let n_points = n.saturating_sub((dim - 1) * tau);
    if n_points < tsep + 2 {
        return Ok((f64::NAN, PyArray1::from_vec(py, vec![])));
    }

    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..dim).map(|d| y[i + d * tau]).collect())
        .collect();

    let eps = epsilon.unwrap_or_else(|| {
        let mean_val = y.iter().sum::<f64>() / n as f64;
        let std_dev = (y.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / n as f64).sqrt();
        std_dev * 0.1
    });

    let max_it = max_iter.unwrap_or(n_points / 4).min(n_points - 1);

    let mut divergence = vec![0.0f64; max_it];
    let mut counts = vec![0usize; max_it];

    for i in 0..n_points {
        for j in 0..n_points {
            if (i as isize - j as isize).unsigned_abs() < tsep {
                continue;
            }
            let dist: f64 = embedded[i]
                .iter()
                .zip(embedded[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < eps && dist > 0.0 {
                for k in 0..max_it {
                    let ii = i + k;
                    let jj = j + k;
                    if ii >= n_points || jj >= n_points {
                        break;
                    }
                    let d: f64 = embedded[ii]
                        .iter()
                        .zip(embedded[jj].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if d > 0.0 {
                        divergence[k] += d.ln();
                        counts[k] += 1;
                    }
                }
            }
        }
    }

    let mut div_curve = Vec::with_capacity(max_it);
    for k in 0..max_it {
        if counts[k] > 0 {
            div_curve.push(divergence[k] / counts[k] as f64);
        }
    }

    let fit_len = (div_curve.len() / 10).max(2).min(div_curve.len());
    let max_lyap = if fit_len >= 2 {
        let n_pts = fit_len as f64;
        let sum_x: f64 = (0..fit_len).map(|i| i as f64).sum();
        let sum_y: f64 = div_curve[..fit_len].iter().sum();
        let sum_xy: f64 = div_curve[..fit_len]
            .iter()
            .enumerate()
            .map(|(i, y)| i as f64 * y)
            .sum();
        let sum_xx: f64 = (0..fit_len).map(|i| (i as f64).powi(2)).sum();
        let denom = n_pts * sum_xx - sum_x * sum_x;
        if denom.abs() > 1e-15 {
            (n_pts * sum_xy - sum_x * sum_y) / denom
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    Ok((max_lyap, PyArray1::from_vec(py, div_curve)))
}
