use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// FTLE via local linearization of neighbor trajectories.
///
/// Returns (ftle_values, confidence) arrays.
#[pyfunction]
#[pyo3(signature = (trajectory, time_horizon=10, n_neighbors=10, epsilon=None))]
pub fn ftle_local_linearization<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray2<f64>,
    time_horizon: usize,
    n_neighbors: usize,
    epsilon: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let traj = trajectory.as_array();
    let (n_points, dim) = (traj.nrows(), traj.ncols());
    let n_valid = n_points.saturating_sub(time_horizon);

    if n_valid < 10 || dim == 0 {
        return Ok((
            PyArray1::from_vec(py, vec![f64::NAN; n_points]),
            PyArray1::from_vec(py, vec![0.0; n_points]),
        ));
    }

    // Auto epsilon: 20th percentile of sampled pairwise distances
    let eps = epsilon.unwrap_or_else(|| {
        let sample_size = 100.min(n_points);
        let mut dists: Vec<f64> = Vec::new();
        for i in 0..sample_size {
            for j in 0..sample_size {
                if i != j {
                    let d: f64 = (0..dim)
                        .map(|d| (traj[[i, d]] - traj[[j, d]]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    dists.push(d);
                }
            }
        }
        if dists.is_empty() {
            return 1.0;
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((0.20) * (dists.len() - 1) as f64) as usize;
        dists[idx]
    });

    let mut ftle = vec![f64::NAN; n_points];
    let mut confidence = vec![0.0f64; n_points];

    for i in 0..n_valid {
        // Find neighbors within epsilon ball
        let mut valid_neighbors: Vec<usize> = Vec::new();
        for j in 0..n_points {
            if j != i && j + time_horizon < n_points {
                let d: f64 = (0..dim)
                    .map(|dd| (traj[[i, dd]] - traj[[j, dd]]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if d < eps {
                    valid_neighbors.push(j);
                }
            }
        }

        // Fall back to k-nearest if not enough in epsilon ball
        if valid_neighbors.len() < n_neighbors {
            let mut dists: Vec<(usize, f64)> = (0..n_points)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f64 = (0..dim)
                        .map(|dd| (traj[[i, dd]] - traj[[j, dd]]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            valid_neighbors = dists
                .iter()
                .take(n_neighbors)
                .filter(|&&(j, _)| j + time_horizon < n_points)
                .map(|&(j, _)| j)
                .collect();
        }

        if valid_neighbors.len() < dim + 1 {
            continue;
        }

        let nn = valid_neighbors.len();
        let mut delta_x0 = vec![vec![0.0f64; dim]; nn];
        let mut delta_xt = vec![vec![0.0f64; dim]; nn];

        for (k, &j) in valid_neighbors.iter().enumerate() {
            for d in 0..dim {
                delta_x0[k][d] = traj[[j, d]] - traj[[i, d]];
                delta_xt[k][d] = traj[[j + time_horizon, d]] - traj[[i + time_horizon, d]];
            }
        }

        let phi_t = match lstsq(&delta_x0, &delta_xt, dim) {
            Some(p) => p,
            None => continue,
        };

        let mut phi = vec![vec![0.0f64; dim]; dim];
        for r in 0..dim {
            for c in 0..dim {
                phi[r][c] = phi_t[c][r];
            }
        }

        let sigma = sigma_max(&phi, dim);

        if sigma > 0.0 {
            ftle[i] = sigma.ln() / time_horizon as f64;
        }

        // Confidence based on R²
        let mut ss_res = 0.0f64;
        let mut ss_tot = 0.0f64;
        for k in 0..nn {
            for d in 0..dim {
                let mut pred = 0.0;
                for d2 in 0..dim {
                    pred += delta_x0[k][d2] * phi_t[d2][d];
                }
                ss_res += (delta_xt[k][d] - pred).powi(2);
                ss_tot += delta_xt[k][d].powi(2);
            }
        }

        if ss_res > 0.0 && ss_tot > 0.0 {
            let r2 = 1.0 - ss_res / ss_tot;
            confidence[i] = r2.clamp(0.0, 1.0);
        } else {
            confidence[i] = 0.5;
        }
    }

    Ok((
        PyArray1::from_vec(py, ftle),
        PyArray1::from_vec(py, confidence),
    ))
}

/// FTLE via direct perturbation of delay-embedded signal.
///
/// Returns (ftle_values, jacobian_norms).
#[allow(unused_variables)]
#[pyfunction]
#[pyo3(signature = (signal, dimension=3, delay=1, time_horizon=10, perturbation=1e-6, n_perturbations=10))]
pub fn ftle_direct_perturbation<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dimension: usize,
    delay: usize,
    time_horizon: usize,
    perturbation: f64,
    n_perturbations: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let y = signal.as_slice()?;
    let n = y.len();

    let n_points = n.saturating_sub((dimension - 1) * delay);
    if n_points < time_horizon + 2 {
        return Ok((
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
        ));
    }

    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..dimension).map(|d| y[i + d * delay]).collect())
        .collect();

    let n_valid = n_points - time_horizon;
    let mut ftle_vals = vec![0.0f64; n_valid];
    let mut jac_norms = vec![0.0f64; n_valid];

    for i in 0..n_valid {
        let mut best_dist = f64::INFINITY;
        let mut best_j = 0;
        for j in 0..n_valid {
            if (i as isize - j as isize).unsigned_abs() < dimension * delay {
                continue;
            }
            let dist: f64 = embedded[i]
                .iter()
                .zip(embedded[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < best_dist && dist > 1e-15 {
                best_dist = dist;
                best_j = j;
            }
        }

        if best_dist < f64::INFINITY {
            let d0 = best_dist;
            let dt: f64 = embedded[i + time_horizon]
                .iter()
                .zip(embedded[best_j + time_horizon].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if d0 > 1e-15 && dt > 0.0 {
                ftle_vals[i] = (dt / d0).ln() / time_horizon as f64;
                jac_norms[i] = dt / d0;
            }
        }
    }

    Ok((
        PyArray1::from_vec(py, ftle_vals),
        PyArray1::from_vec(py, jac_norms),
    ))
}

// --- Helper functions ---

/// Solve least squares: A @ X = B via normal equations.
fn lstsq(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let m = a.len();
    let p = b[0].len();

    let mut ata = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }

    let ata_inv = mat_invert(&ata, n)?;

    let mut atb = vec![vec![0.0f64; p]; n];
    for i in 0..n {
        for j in 0..p {
            for k in 0..m {
                atb[i][j] += a[k][i] * b[k][j];
            }
        }
    }

    let mut x = vec![vec![0.0f64; p]; n];
    for i in 0..n {
        for j in 0..p {
            for k in 0..n {
                x[i][j] += ata_inv[i][k] * atb[k][j];
            }
        }
    }

    Some(x)
}

/// Largest singular value via power iteration on M^T M.
fn sigma_max(mat: &[Vec<f64>], n: usize) -> f64 {
    let mut c = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += mat[k][i] * mat[k][j];
            }
        }
    }

    let mut v = vec![1.0 / (n as f64).sqrt(); n];

    for _ in 0..200 {
        let mut w = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += c[i][j] * v[j];
            }
        }

        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return 0.0;
        }
        v = w.iter().map(|x| x / norm).collect();
    }

    let mut cv = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            cv[i] += c[i][j] * v[j];
        }
    }
    let lambda: f64 = cv.iter().zip(v.iter()).map(|(a, b)| a * b).sum();

    if lambda > 0.0 { lambda.sqrt() } else { 0.0 }
}

/// Invert a square matrix using Gauss-Jordan elimination.
fn mat_invert(mat: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let mut aug = vec![vec![0.0f64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = mat[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    for col in 0..n {
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in col + 1..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return None;
        }

        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for j in 0..2 * n {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..2 * n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    let mut inv = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }

    Some(inv)
}
