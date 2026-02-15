use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Hurst exponent via R/S analysis.
///
/// Port of manifold/primitives/individual/fractal.py::hurst_exponent(method='rs')
///
/// Config constants (from manifold.primitives.config):
///   rs_min_k = 10
///   rs_max_k_ratio = 0.5
///   rs_max_k_cap = 100
#[pyfunction]
#[pyo3(signature = (signal, method="rs"))]
pub fn hurst_exponent(signal: PyReadonlyArray1<f64>, method: &str) -> PyResult<f64> {
    let raw = signal.as_slice()?;

    // Flatten and strip NaN
    let y: Vec<f64> = raw.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = y.len();

    const RS_MIN_K: usize = 10;

    if n < RS_MIN_K {
        return Ok(f64::NAN);
    }

    if method == "dfa" {
        return dfa(&y);
    }

    // rs_max_k_ratio = 0.5, rs_max_k_cap = 100
    let max_k = ((n as f64 * 0.5) as usize).min(100);

    let mut k_values: Vec<f64> = Vec::new();
    let mut rs_values: Vec<f64> = Vec::new();

    for k in RS_MIN_K..max_k {
        let n_subseries = n / k;
        let mut rs_sum = 0.0f64;

        for i in 0..n_subseries {
            let subseries = &y[i * k..(i + 1) * k];

            let mean: f64 = subseries.iter().sum::<f64>() / k as f64;

            // Cumulative deviation from mean
            let mut cum_dev = Vec::with_capacity(k);
            let mut running = 0.0;
            for &val in subseries {
                running += val - mean;
                cum_dev.push(running);
            }

            // Range
            let r = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);

            // Standard deviation with ddof=1
            let variance: f64 =
                subseries.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (k - 1) as f64;
            let s = variance.sqrt();

            if s > 0.0 {
                rs_sum += r / s;
            }
        }

        if n_subseries > 0 {
            let rs_avg = rs_sum / n_subseries as f64;
            if rs_avg > 0.0 {
                k_values.push((k as f64).ln());
                rs_values.push(rs_avg.ln());
            }
        }
    }

    if k_values.len() < 3 {
        return Ok(f64::NAN);
    }

    // Linear fit: log(R/S) = H * log(k) + c
    let n_pts = k_values.len() as f64;
    let sum_x: f64 = k_values.iter().sum();
    let sum_y: f64 = rs_values.iter().sum();
    let sum_xy: f64 = k_values.iter().zip(rs_values.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = k_values.iter().map(|x| x * x).sum();

    let denom = n_pts * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return Ok(f64::NAN);
    }

    let h = (n_pts * sum_xy - sum_x * sum_y) / denom;

    // Clip to [0, 1]
    Ok(h.clamp(0.0, 1.0))
}

/// Detrended Fluctuation Analysis (order=1).
///
/// Port of primitives/complexity.py::_dfa
fn dfa(y: &[f64]) -> PyResult<f64> {
    let n = y.len();
    if n < 20 {
        return Ok(f64::NAN);
    }

    // Cumulative sum of (signal - mean)
    let mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut cum = Vec::with_capacity(n);
    let mut running = 0.0;
    for &val in y {
        running += val - mean;
        cum.push(running);
    }

    // Logspace scales: 20 points from 10 to min(n*0.25, 100)
    let min_scale: f64 = 10.0;
    let max_scale: f64 = (n as f64 * 0.25).min(100.0);
    let log_min = min_scale.log10();
    let log_max = max_scale.log10();

    let mut scales: Vec<usize> = Vec::new();
    for i in 0..20 {
        let t = log_min + (log_max - log_min) * (i as f64) / 19.0;
        let s = 10.0f64.powf(t) as usize;
        if scales.is_empty() || *scales.last().unwrap() != s {
            scales.push(s);
        }
    }

    let mut log_scales: Vec<f64> = Vec::new();
    let mut log_flucts: Vec<f64> = Vec::new();

    for &scale in &scales {
        let n_segments = n / scale;
        if n_segments < 2 {
            continue;
        }

        let mut f_sq_sum = 0.0;
        let mut f_sq_count = 0usize;

        for seg in 0..n_segments {
            let start = seg * scale;
            let segment = &cum[start..start + scale];

            // Linear detrend (polyfit degree 1): y = a*x + b
            // Using normal equations for least squares
            let sn = scale as f64;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sxy = 0.0f64;
            let mut sxx = 0.0f64;
            for (i, &v) in segment.iter().enumerate() {
                let x = i as f64;
                sx += x;
                sy += v;
                sxy += x * v;
                sxx += x * x;
            }
            let denom = sn * sxx - sx * sx;
            if denom.abs() < 1e-30 {
                continue;
            }
            let a = (sn * sxy - sx * sy) / denom;
            let b = (sy - a * sx) / sn;

            // Mean squared residual
            let mut ms = 0.0;
            for (i, &v) in segment.iter().enumerate() {
                let trend = a * (i as f64) + b;
                ms += (v - trend).powi(2);
            }
            ms /= sn;

            f_sq_sum += ms;
            f_sq_count += 1;
        }

        if f_sq_count > 0 {
            let fluct = (f_sq_sum / f_sq_count as f64).sqrt();
            log_scales.push((scale as f64).ln());
            log_flucts.push(fluct.ln());
        }
    }

    if log_scales.len() < 3 {
        return Ok(f64::NAN);
    }

    // Linear fit: log(F) = alpha * log(scale) + c
    let np = log_scales.len() as f64;
    let sx: f64 = log_scales.iter().sum();
    let sy: f64 = log_flucts.iter().sum();
    let sxy: f64 = log_scales.iter().zip(log_flucts.iter()).map(|(x, y)| x * y).sum();
    let sxx: f64 = log_scales.iter().map(|x| x * x).sum();

    let denom = np * sxx - sx * sx;
    if denom.abs() < 1e-15 {
        return Ok(f64::NAN);
    }

    let alpha = (np * sxy - sx * sy) / denom;
    Ok(alpha)
}
