//! This module contains the Rust transcription of C functions used
//! in [aubio](https://github.com/aubio/aubio), in order to avoid depending
//! on C bindings.
use crate::{BlissError, BlissResult};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/*
 * Timbral part
*/

/// Compute spectral centroid from FFT magnitudes (in bins)
///
/// This is a direct transcription of aubio's cvec_centroid function from
/// src/spectral/statistics.c
fn spectral_centroid(spec_norm: &[f32]) -> f32 {
    let sum: f32 = spec_norm.iter().sum();

    if sum == 0.0 {
        return 0.0;
    }

    let mut sc = 0.0;
    for (j, &norm_value) in spec_norm.iter().enumerate() {
        sc += j as f32 * norm_value;
    }

    sc / sum
}

/// Compute spectral rolloff from FFT magnitudes (in bins)
///
/// This is a direct transcription of aubio's aubio_specdesc_rolloff function from
/// src/spectral/statistics.c
/// Returns the bin number below which 95% of the spectrum energy is found.
fn spectral_rolloff(spec_norm: &[f32]) -> f32 {
    let mut cumsum = 0.0;
    let mut rollsum = 0.0;

    // Compute sum of squared magnitudes
    for &norm_value in spec_norm.iter() {
        cumsum += norm_value * norm_value;
    }

    if cumsum == 0.0 {
        return 0.0;
    }

    // Find bin where 95% of energy is reached
    cumsum *= 0.95;
    let mut j = 0;
    while rollsum < cumsum && j < spec_norm.len() {
        rollsum += spec_norm[j] * spec_norm[j];
        j += 1;
    }

    j as f32
}

/// Convert frequency bin to frequency (Hz)
///
/// This is a direct transcription of aubio's aubio_bintofreq function from
/// src/mathutils.c
///
/// - `bin` Frequency bin to convert
/// - `sample_rate` Sampling rate of the original signal
/// - `fft_size` Size of the FFT window
pub(crate) fn bin_to_freq(bin: f32, sample_rate: f32, fft_size: f32) -> f32 {
    let freq = sample_rate / fft_size;
    freq * bin.max(0.0)
}

/// Spectral descriptor shape - matches aubio's SpecShape enum
#[derive(Debug, Clone, Copy)]
pub(crate) enum SpecShape {
    Centroid,
    Rolloff,
}

/// Spectral descriptor object - matches aubio's SpecDesc API
pub(crate) struct SpecDesc {
    shape: SpecShape,
    _buf_size: usize,
}

impl SpecDesc {
    /// Create new spectral descriptor - matches aubio's API
    pub(crate) fn new(shape: SpecShape, buf_size: usize) -> BlissResult<Self> {
        Ok(SpecDesc {
            shape,
            _buf_size: buf_size,
        })
    }

    /// Compute spectral descriptor value - matches aubio's do_result() API
    pub(crate) fn do_result(&self, fftgrain: &[f32]) -> BlissResult<f32> {
        let num_bins = fftgrain.len() / 2;
        let norm = &fftgrain[..num_bins];

        let result = match self.shape {
            SpecShape::Centroid => spectral_centroid(norm),
            SpecShape::Rolloff => spectral_rolloff(norm),
        };

        Ok(result)
    }
}

/// Pure Rust Phase Vocoder implementation
///
/// This is a transcription of aubio's phase vocoder (pvoc) to avoid C dependencies.
/// NOTE: This reproduces the buggy behavior where only 256 bins are output
/// It performs:
/// 1. Buffer sliding with overlap
/// 2. Hann window application (hanningz variant: 0.5 * (1 - cos(2πi/N)))
/// 3. FFT shift (swap first and second halves)
/// 4. FFT computation
/// 5. Conversion to aubio's packed format
pub(crate) struct PVoc {
    win_s: usize,
    hop_s: usize,
    data: Vec<f32>,
    dataold: Vec<f32>,
    window: Vec<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    buffer: Vec<Complex<f32>>, // Reusable FFT buffer to avoid allocations
}

impl PVoc {
    pub(crate) fn new(win_s: usize, hop_s: usize) -> BlissResult<Self> {
        if hop_s < 1 {
            return Err(BlissError::AnalysisError(format!(
                "pvoc: hop_size ({}) must be >= 1",
                hop_s
            )));
        }
        if win_s < 2 {
            return Err(BlissError::AnalysisError(format!(
                "pvoc: win_size ({}) must be >= 2",
                win_s
            )));
        }
        if win_s < hop_s {
            return Err(BlissError::AnalysisError(format!(
                "pvoc: hop_size ({}) is larger than win_size ({})",
                hop_s, win_s
            )));
        }

        // Create Hann window (hanningz variant)
        let mut window = vec![0.0; win_s];
        for (i, item) in window.iter_mut().enumerate().take(win_s) {
            *item = 0.5 * (1.0 - (2.0 * PI * i as f32 / win_s as f32).cos());
        }

        // Create FFT planner
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(win_s);

        let dataold_size = if win_s > hop_s { win_s - hop_s } else { 1 };

        Ok(PVoc {
            win_s,
            hop_s,
            data: vec![0.0; win_s],
            dataold: vec![0.0; dataold_size],
            window,
            fft,
            buffer: vec![Complex::new(0.0, 0.0); win_s],
        })
    }

    /// Compute spectral frame
    ///
    /// This is a transcription of aubio_pvoc_do from phasevoc.c
    /// (matching the original buffer overflow bug with 512-element buffer)
    ///
    /// Like aubio's bindings, this accepts input of any size >= hop_s and only uses the first hop_s elements.
    ///
    /// - `input` New input signal (>= hop_s samples, only first hop_s used)
    /// - `fftgrain` Output buffer (win_s samples: win_s/2 norm + win_s/2 phase = 256+256)
    pub(crate) fn do_(&mut self, input: &[f32], fftgrain: &mut [f32]) -> BlissResult<()> {
        if input.len() < self.hop_s {
            return Err(BlissError::AnalysisError(format!(
                "pvoc input size mismatch: expected at least {}, got {}",
                self.hop_s,
                input.len()
            )));
        }
        if fftgrain.len() != self.win_s {
            return Err(BlissError::AnalysisError(format!(
                "pvoc output size mismatch: expected {}, got {}",
                self.win_s,
                fftgrain.len()
            )));
        }

        // Slide buffers (aubio_pvoc_swapbuffers)
        let end = self.win_s.saturating_sub(self.hop_s);

        // Copy old data to beginning of data buffer
        for i in 0..end {
            self.data[i] = self.dataold[i];
        }

        // Copy new input to end of data buffer
        self.data[end..(self.hop_s + end)].copy_from_slice(&input[..self.hop_s]);

        // Update dataold for next iteration
        for i in 0..end {
            self.dataold[i] = self.data[i + self.hop_s];
        }

        // Apply window (fvec_weight)
        for i in 0..self.win_s {
            self.data[i] *= self.window[i];
        }

        // FFT shift (fvec_shift) - swap first and second halves
        let half = self.win_s / 2;
        let start = if 2 * half < self.win_s {
            half + 1 // odd length: middle element moves to end
        } else {
            half
        };

        for j in 0..half {
            self.data.swap(j, j + start);
        }

        // Compute FFT - reuse buffer to avoid allocation
        for (i, &x) in self.data.iter().enumerate() {
            self.buffer[i] = Complex::new(x, 0.0);
        }
        self.fft.process(&mut self.buffer);

        // Convert to CVec format - BUGGY VERSION: only 256 bins
        // This matches the original buffer overflow behavior where bin 256 was lost
        // Output format: [norm_0, ..., norm_255, phas_0, ..., phas_255]
        let num_bins = self.win_s / 2; // 256 for win_s=512 (NOT 257!)

        // Bin 0 (DC): always real, use abs
        fftgrain[0] = self.buffer[0].re.abs();
        fftgrain[num_bins] = if self.buffer[0].re < 0.0 { PI } else { 0.0 };

        // Bins 1 to 254: compute magnitude and phase normally
        for i in 1..num_bins - 1 {
            let re = self.buffer[i].re;
            let im = self.buffer[i].im;
            fftgrain[i] = (re * re + im * im).sqrt(); // magnitude
            fftgrain[num_bins + i] = im.atan2(re); // phase
        }

        // Bin 255: aubio puts the NYQUIST bin here (bin 256) due to the buffer overflow bug!
        // When CVec has length 256, aubio computes norm[255] = |Re[256]| (Nyquist)
        let nyquist_idx = self.win_s / 2; // 256 for win_s=512
        fftgrain[num_bins - 1] = self.buffer[nyquist_idx].re.abs(); // |Re[256]| -> fftgrain[255]
        fftgrain[num_bins + num_bins - 1] = if self.buffer[nyquist_idx].re < 0.0 {
            PI
        } else {
            0.0
        };

        Ok(())
    }
}

/*
 * Tempo part
*/

/// Phase Vocoder for Tempo Detection
///
/// Similar to PVoc in timbral.rs, but outputs CORRECT 257 bins (not buggy 256)
struct PVocTempo {
    win_s: usize,
    hop_s: usize,
    data: Vec<f32>,
    dataold: Vec<f32>,
    window: Vec<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    buffer: Vec<Complex<f32>>, // Reusable FFT buffer to avoid allocations
}

impl PVocTempo {
    fn new(win_s: usize, hop_s: usize) -> BlissResult<Self> {
        if hop_s < 1 {
            return Err(BlissError::AnalysisError(format!(
                "pvoc: hop_size ({}) must be >= 1",
                hop_s
            )));
        }
        if win_s < 2 {
            return Err(BlissError::AnalysisError(format!(
                "pvoc: win_size ({}) must be >= 2",
                win_s
            )));
        }
        if win_s < hop_s {
            return Err(BlissError::AnalysisError(format!(
                "pvoc: hop_size ({}) is larger than win_size ({})",
                hop_s, win_s
            )));
        }

        // Create Hann window (hanningz variant)
        let mut window = vec![0.0; win_s];
        for (i, item) in window.iter_mut().enumerate().take(win_s) {
            *item = 0.5 * (1.0 - (2.0 * PI * i as f32 / win_s as f32).cos());
        }

        // Create FFT planner
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(win_s);

        let dataold_size = if win_s > hop_s { win_s - hop_s } else { 1 };

        Ok(PVocTempo {
            win_s,
            hop_s,
            data: vec![0.0; win_s],
            dataold: vec![0.0; dataold_size],
            window,
            fft,
            buffer: vec![Complex::new(0.0, 0.0); win_s],
        })
    }

    /// Compute spectral frame.
    ///
    /// Unlike the buggy timbral version, this outputs win_s + 2 elements:
    /// - 257 norm values (bins 0-256, including DC and Nyquist)
    /// - 257 phase values
    ///
    /// Total: 514 elements
    ///
    /// - `input` New input signal (>= hop_s samples, only first hop_s used)
    /// - `fftgrain` Output buffer (win_s + 2 samples = 514 for win_s=512)
    fn do_(&mut self, input: &[f32], fftgrain: &mut [f32]) -> BlissResult<()> {
        if input.len() < self.hop_s {
            return Err(BlissError::AnalysisError(format!(
                "pvoc input size mismatch: expected at least {}, got {}",
                self.hop_s,
                input.len()
            )));
        }
        if fftgrain.len() != self.win_s + 2 {
            return Err(BlissError::AnalysisError(format!(
                "pvoc output size mismatch: expected {}, got {}",
                self.win_s + 2,
                fftgrain.len()
            )));
        }

        // Slide buffers (aubio_pvoc_swapbuffers)
        let end = self.win_s.saturating_sub(self.hop_s);

        // Copy old data to beginning of data buffer
        for i in 0..end {
            self.data[i] = self.dataold[i];
        }

        // Copy new input to end of data buffer
        self.data[end..(self.hop_s + end)].copy_from_slice(&input[..self.hop_s]);

        // Update dataold for next iteration
        for i in 0..end {
            self.dataold[i] = self.data[i + self.hop_s];
        }

        // Apply window (fvec_weight)
        for i in 0..self.win_s {
            self.data[i] *= self.window[i];
        }

        // FFT shift (fvec_shift) - swap first and second halves
        let half = self.win_s / 2;
        let start = if 2 * half < self.win_s {
            half + 1 // odd length: middle element moves to end
        } else {
            half
        };

        for j in 0..half {
            self.data.swap(j, j + start);
        }

        // For odd-length arrays, shift the second half left by one position
        if start != half {
            for j in 0..half {
                self.data.swap(j + start - 1, j + start);
            }
        }

        // Compute FFT - reuse buffer to avoid allocation
        for (i, &x) in self.data.iter().enumerate() {
            self.buffer[i] = Complex::new(x, 0.0);
        }
        self.fft.process(&mut self.buffer);

        // Convert to CVec format - CORRECT VERSION: 257 bins
        // Output format: [norm_0, ..., norm_256, phas_0, ..., phas_256]
        let num_bins = self.win_s / 2 + 1; // 257 for win_s=512

        // Bin 0 (DC): always real, use abs
        fftgrain[0] = self.buffer[0].re.abs();
        fftgrain[num_bins] = if self.buffer[0].re < 0.0 { PI } else { 0.0 };

        // Bins 1 to 255: compute magnitude and phase normally
        for i in 1..num_bins - 1 {
            let re = self.buffer[i].re;
            let im = self.buffer[i].im;
            fftgrain[i] = (re * re + im * im).sqrt(); // magnitude
            fftgrain[num_bins + i] = im.atan2(re); // phase
        }

        // Bin 256 (Nyquist): always real, use abs
        let nyquist_idx = self.win_s / 2; // 256 for win_s=512
        fftgrain[num_bins - 1] = self.buffer[nyquist_idx].re.abs();
        fftgrain[num_bins + num_bins - 1] = if self.buffer[nyquist_idx].re < 0.0 {
            PI
        } else {
            0.0
        };

        Ok(())
    }
}

/// SpecFlux onset detection
///
/// Transcription of aubio_specdesc_specflux from aubio/src/spectral/specdesc.c:235-243
/// Computes spectral flux: sum of positive magnitude differences between frames
struct SpecFlux {
    /// Previous frame magnitudes
    oldmag: Vec<f32>,
}

impl SpecFlux {
    /// Create new SpecFlux onset detector
    ///
    /// - `size` FFT window size (e.g., 512)
    fn new(size: usize) -> Self {
        // aubio uses size/2+1 for the CVec length (real FFT bins)
        let rsize = size / 2 + 1;
        SpecFlux {
            oldmag: vec![0.0; rsize],
        }
    }

    /// Compute spectral flux onset detection function
    ///
    /// Direct transcription of aubio_specdesc_specflux
    ///
    /// - `fftgrain_norm` Magnitude spectrum (norm part of CVec)
    /// - Returns: Onset detection function value (sum of positive differences)
    fn do_(&mut self, fftgrain_norm: &[f32]) -> f32 {
        let mut onset = 0.0;

        // Sum positive differences: if (norm[j] > oldmag[j]) onset += norm[j] - oldmag[j]
        for (j, &norm_val) in fftgrain_norm.iter().enumerate() {
            if norm_val > self.oldmag[j] {
                onset += norm_val - self.oldmag[j];
            }
            self.oldmag[j] = norm_val;
        }

        onset
    }
}

/// Helper functions for PeakPicker
/// Compute mean of a slice
fn vec_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f32 = data.iter().sum();
    sum / data.len() as f32
}

/// Compute median of a slice using quickselect
/// Transcribed from aubio/src/mathutils.c:435-484
fn vec_median(data: &mut [f32]) -> f32 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    let mut low = 0;
    let mut high = n - 1;
    let median = (low + high) / 2;

    loop {
        if high <= low {
            // One element only
            return data[median];
        }

        if high == low + 1 {
            // Two elements only
            if data[low] > data[high] {
                data.swap(low, high);
            }
            return data[median];
        }

        // Find median of low, middle and high items; swap into position low
        let middle = (low + high) / 2;
        if data[middle] > data[high] {
            data.swap(middle, high);
        }
        if data[low] > data[high] {
            data.swap(low, high);
        }
        if data[middle] > data[low] {
            data.swap(middle, low);
        }

        // Swap low item (now in position middle) into position (low+1)
        data.swap(middle, low + 1);

        // Nibble from each end towards middle, swapping items when stuck
        let mut ll = low + 1;
        let mut hh = high;

        loop {
            ll += 1;
            while data[low] > data[ll] {
                ll += 1;
            }

            hh -= 1;
            while data[hh] > data[low] {
                hh -= 1;
            }

            if hh < ll {
                break;
            }

            data.swap(ll, hh);
        }

        // Swap middle item (in position low) back into correct position
        data.swap(low, hh);

        // Re-set active partition
        if hh <= median {
            low = ll;
        }
        if hh >= median {
            high = hh - 1;
        }
    }
}

/// Push value to end of vector, shifting all elements left
/// Transcribed from aubio/src/mathutils.c:308-314
fn vec_push(data: &mut [f32], new_elem: f32) {
    for i in 0..data.len() - 1 {
        data[i] = data[i + 1];
    }
    data[data.len() - 1] = new_elem;
}

/// Check if position is a peak (greater than neighbors)
/// Transcribed from aubio/src/mathutils.c:511-517
fn vec_peakpick(onset: &[f32], pos: usize) -> bool {
    if pos == 0 || pos >= onset.len() - 1 {
        return false;
    }
    onset[pos] > onset[pos - 1] && onset[pos] > onset[pos + 1] && onset[pos] > 0.0
}

/// Quadratic interpolation for peak position
/// Transcribed from aubio/src/mathutils.c:486-498
fn vec_quadratic_peak_pos(x: &[f32], pos: usize) -> f32 {
    if pos == 0 || pos >= x.len() - 1 {
        return pos as f32;
    }

    let x0 = if pos < 1 { pos } else { pos - 1 };
    let x2 = if pos + 1 < x.len() { pos + 1 } else { pos };

    if x0 == pos {
        return if x[pos] <= x[x2] {
            pos as f32
        } else {
            x2 as f32
        };
    }
    if x2 == pos {
        return if x[pos] <= x[x0] {
            pos as f32
        } else {
            x0 as f32
        };
    }

    let s0 = x[x0];
    let s1 = x[pos];
    let s2 = x[x2];

    pos as f32 + 0.5 * (s0 - s2) / (s0 - 2.0 * s1 + s2)
}

/// Biquad IIR filter
/// Transcription of aubio biquad filter from aubio/src/temporal/biquad.c
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    // State variables for filter
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Biquad {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Apply filter to one sample
    fn process_sample(&mut self, x0: f32) -> f32 {
        let y0 = self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        // Update state
        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0;

        y0
    }

    /// Reset filter state
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    /// Apply filtfilt (forward-backward zero-phase filter)
    /// Transcribed from aubio/src/temporal/filter.c:77-93
    fn do_filtfilt(&mut self, data: &mut [f32], tmp: &mut [f32]) {
        let length = data.len();

        // Forward pass
        for item in data.iter_mut().take(length) {
            *item = self.process_sample(*item);
        }
        self.reset();

        // Mirror
        for i in 0..length {
            tmp[length - i - 1] = data[i];
        }

        // Backward pass on mirrored
        for item in tmp.iter_mut().take(length) {
            *item = self.process_sample(*item);
        }
        self.reset();

        // Invert back
        for i in 0..length {
            data[i] = tmp[length - i - 1];
        }
    }
}

/// Peak Picker for onset detection
///
/// Transcription of aubio_peakpicker from aubio/src/onset/peakpicker.c
/// Uses adaptive thresholding to detect peaks in onset detection function
struct PeakPicker {
    threshold: f32,
    win_post: usize,

    biquad: Biquad,
    onset_keep: Vec<f32>,
    onset_proc: Vec<f32>,
    onset_peek: Vec<f32>,
    thresholded: f32,
    scratch: Vec<f32>,
}

impl PeakPicker {
    /// Create new peak picker with default parameters
    /// Transcribed from aubio/src/onset/peakpicker.c:161-187
    fn new() -> Self {
        let threshold = 0.1;
        let win_post = 5;
        let win_pre = 1;

        let buf_size = win_post + win_pre + 1;

        // Biquad coefficients from aubio (2nd order Butterworth lowpass, cutoff 0.34)
        let biquad = Biquad::new(0.159_987_9, 0.31997577, 0.159_987_9, 0.23484048, 0.0);

        PeakPicker {
            threshold,
            win_post,
            biquad,
            onset_keep: vec![0.0; buf_size],
            onset_proc: vec![0.0; buf_size],
            onset_peek: vec![0.0; 3],
            thresholded: 0.0,
            scratch: vec![0.0; buf_size],
        }
    }

    /// Process one onset value and detect peaks
    /// Transcribed from aubio/src/onset/peakpicker.c:86-123
    ///
    /// Returns: peak position if detected (0.0 if no peak)
    fn do_(&mut self, onset: f32) -> f32 {
        // Push new novelty to the end
        vec_push(&mut self.onset_keep, onset);

        // Store a copy
        self.onset_proc.copy_from_slice(&self.onset_keep);

        // Filter this copy
        self.biquad
            .do_filtfilt(&mut self.onset_proc, &mut self.scratch);

        // Calculate mean for onset_proc
        let mean = vec_mean(&self.onset_proc);

        // Copy to scratch and compute its median
        self.scratch.copy_from_slice(&self.onset_proc);
        let median = vec_median(&mut self.scratch);

        // Shift peek array
        for j in 0..2 {
            self.onset_peek[j] = self.onset_peek[j + 1];
        }

        // Calculate new thresholded value
        self.thresholded = self.onset_proc[self.win_post] - median - mean * self.threshold;

        self.onset_peek[2] = self.thresholded;

        // Check for peak at position 1 (middle of 3-element window)
        if vec_peakpick(&self.onset_peek, 1) {
            // Interpolate exact peak position
            return vec_quadratic_peak_pos(&self.onset_peek, 1);
        }

        0.0
    }

    /// Get current thresholded value
    fn get_thresholded(&self) -> f32 {
        self.thresholded
    }

    /// Set threshold parameter
    fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }
}

/// Additional helper functions for BeatTracking
/// Find index of maximum element in slice
/// Matches C aubio's fvec_max_elem behavior:
/// - Iterates through all elements
/// - Updates pos whenever current value >= tmp (not just >)
/// - This ensures pos points to the last occurrence of the maximum value
fn vec_max_elem(data: &[f32]) -> usize {
    let mut pos = 0;
    let mut tmp = 0.0;
    for (j, &val) in data.iter().enumerate() {
        // C logic: pos = (tmp > val) ? pos : j;
        // This means: update pos to j unless tmp > val
        if tmp <= val {
            pos = j;
            tmp = val;
        }
    }
    pos
}

/// Reverse a slice in place
fn vec_rev(data: &mut [f32]) {
    let len = data.len();
    for j in 0..(len / 2) {
        data.swap(j, len - 1 - j);
    }
}

/// Element-wise multiplication (weight)
fn vec_weight(data: &mut [f32], weight: &[f32]) {
    let length = data.len().min(weight.len());
    for j in 0..length {
        data[j] *= weight[j];
    }
}

/// Compute autocorrelation
/// Transcribed from aubio/src/mathutils.c:655-669
fn vec_autocorr(input: &[f32], output: &mut [f32]) {
    let length = input.len();
    for i in 0..length {
        let mut tmp = 0.0;
        for j in i..length {
            tmp += input[j - i] * input[j];
        }
        output[i] = tmp / (length - i) as f32;
    }
}

/// Beat Tracking for BPM estimation
///
/// Transcription of aubio_beattracking from aubio/src/tempo/beattracking.c
/// Uses autocorrelation and comb filterbank to estimate tempo
struct BeatTracking {
    hop_size: usize,
    samplerate: u32,

    rwv: Vec<f32>,    // Rayleigh weighting
    gwv: Vec<f32>,    // Gaussian weighting
    dfwv: Vec<f32>,   // Detection function weighting
    dfrev: Vec<f32>,  // Reversed detection function
    acf: Vec<f32>,    // Autocorrelation function
    acfout: Vec<f32>, // Filtered autocorrelation
    phwv: Vec<f32>,   // Phase weighting
    phout: Vec<f32>,  // Phase output

    timesig: u32,
    step: usize,
    rayparam: u32, // Changed to u32 to match C's uint_t for proper truncation
    lastbeat: f32,
    counter: i32,
    flagstep: u32,
    g_var: f32,
    gp: f32,
    bp: f32, // Beat period (in detection function frames)
    rp: f32,
    rp1: f32,
    rp2: f32,
}

impl BeatTracking {
    /// Get time signature from autocorrelation
    /// Transcribed from aubio/src/tempo/beattracking.c:317-335
    fn get_timesig(acf: &[f32], gp: usize) -> u32 {
        if gp < 2 {
            return 4;
        }

        let mut three_energy = 0.0;
        let mut four_energy = 0.0;
        let acflen = acf.len();

        if acflen > 6 * gp + 2 {
            // C code doesn't have bounds checking here because the outer condition ensures validity
            for k in -2..2 {
                three_energy += acf[(3 * gp as i32 + k) as usize];
                four_energy += acf[(4 * gp as i32 + k) as usize];
            }
        } else {
            // Expanded to be more accurate in time sig estimation
            // C code doesn't have bounds checking - we need to add it to avoid panics
            for k in -2..2 {
                let idx3 = (3 * gp as i32 + k) as usize;
                let idx6 = (6 * gp as i32 + k) as usize;
                let idx4 = (4 * gp as i32 + k) as usize;
                let idx2 = (2 * gp as i32 + k) as usize;

                if idx3 < acflen && idx6 < acflen {
                    three_energy += acf[idx3] + acf[idx6];
                } else if idx3 < acflen {
                    three_energy += acf[idx3];
                }

                if idx4 < acflen && idx2 < acflen {
                    four_energy += acf[idx4] + acf[idx2];
                } else if idx4 < acflen {
                    four_energy += acf[idx4];
                }
            }
        }

        if three_energy > four_energy {
            3
        } else {
            4
        }
    }

    /// Create new beat tracking object
    /// Transcribed from aubio/src/tempo/beattracking.c:58-108
    fn new(winlen: usize, hop_size: usize, samplerate: u32) -> Self {
        // Default rayleigh parameter - sets preferred tempo to 120bpm
        // C uses float for initialization but stores as uint_t for rp fallback
        let rayparam_float = 60.0 * samplerate as f32 / 120.0 / hop_size as f32;
        let rayparam = rayparam_float as u32; // Truncate for struct storage, matching C
                                              // Use float rayparam for initialization, matching C lines 66, 100, 105-106
        let dfwvnorm = ((2.0_f32.ln() / rayparam_float) * (winlen + 2) as f32).exp();

        // Length over which beat period is found
        let laglen = winlen / 4;
        // Step increment
        let step = winlen / 4;

        // Initialize Rayleigh weighting - use float rayparam, matching C lines 105-106
        let mut rwv = vec![0.0; laglen];
        for (i, item) in rwv.iter_mut().enumerate().take(laglen) {
            let i_f = (i + 1) as f32;
            *item = (i_f / rayparam_float.powi(2))
                * (-(i_f.powi(2)) / (2.0 * rayparam_float.powi(2))).exp();
        }

        // Initialize detection function weighting - use float rayparam, matching C line 100
        let mut dfwv = vec![0.0; winlen];
        for (i, item) in dfwv.iter_mut().enumerate().take(winlen) {
            *item = ((2.0_f32.ln() / rayparam_float) * (i + 1) as f32).exp() / dfwvnorm;
        }

        BeatTracking {
            hop_size,
            samplerate,
            rwv,
            gwv: vec![0.0; laglen],
            dfwv,
            dfrev: vec![0.0; winlen],
            acf: vec![0.0; winlen],
            acfout: vec![0.0; laglen],
            phwv: vec![1.0; 2 * laglen], // Initialize to ones for flat phase weighting (initial state)
            phout: vec![0.0; winlen],
            timesig: 0,
            step,
            rayparam,
            lastbeat: 0.0,
            counter: 0,
            flagstep: 0,
            g_var: 3.901,
            gp: 0.0,
            bp: 0.0,
            rp: 1.0,
            rp1: 0.0,
            rp2: 0.0,
        }
    }

    /// Process detection function frame and estimate beats
    /// Transcribed from aubio/src/tempo/beattracking.c:125-262
    fn do_(&mut self, dfframe: &[f32], output: &mut [f32]) {
        let step = self.step;
        let laglen = self.rwv.len();
        let winlen = self.dfwv.len();

        // Number of harmonics in shift invariant comb filterbank
        let numelem = if self.timesig == 0 {
            4
        } else {
            self.timesig as usize
        };

        // Copy dfframe, apply weighting, and reverse
        self.dfrev.copy_from_slice(dfframe);
        vec_weight(&mut self.dfrev, &self.dfwv);
        vec_rev(&mut self.dfrev);

        // Compute autocorrelation
        vec_autocorr(dfframe, &mut self.acf);

        // Clear acfout
        for val in self.acfout.iter_mut() {
            *val = 0.0;
        }

        // Compute shift invariant comb filterbank
        for i in 1..laglen - 1 {
            for a in 1..=numelem {
                for b in 1..2 * a {
                    if i * a + b - 1 < self.acf.len() {
                        self.acfout[i] += self.acf[i * a + b - 1] / (2.0 * a as f32 - 1.0);
                    }
                }
            }
        }

        // Apply Rayleigh weight
        vec_weight(&mut self.acfout, &self.rwv);

        // Find non-zero Rayleigh period
        let maxindex = vec_max_elem(&self.acfout);
        if maxindex > 0 && maxindex < self.acfout.len() - 1 {
            self.rp = vec_quadratic_peak_pos(&self.acfout, maxindex);
        } else {
            self.rp = self.rayparam as f32; // Cast u32 to f32 for rp
        }

        self.checkstate();

        let bp = self.bp;

        if bp == 0.0 {
            // Zero output
            for val in output.iter_mut() {
                *val = 0.0;
            }
            return;
        }

        // Compute beat phase
        let kmax = (winlen as f32 / bp).floor() as usize;

        // Clear phout
        for val in self.phout.iter_mut() {
            *val = 0.0;
        }

        // In C, "for (i = 0; i < bp; i++)" where bp is float compares each i against bp as float
        // e.g. if bp=54.712, loop runs while i < 54.712, so i=0..54 (55 iterations), not 0..53
        let mut i = 0;
        while (i as f32) < bp && i < self.phout.len() {
            for k in 0..kmax {
                // C uses ROUND(x) = FLOOR(x+0.5), not Rust's .round() which uses banker's rounding
                let idx = i + ((bp * k as f32) + 0.5).floor() as usize;
                if idx < self.dfrev.len() {
                    self.phout[i] += self.dfrev[idx];
                }
            }
            i += 1;
        }

        vec_weight(&mut self.phout, &self.phwv);
        // Find phase
        let maxindex = vec_max_elem(&self.phout);
        let mut phase = if maxindex >= winlen - 1 {
            step as f32 - self.lastbeat
        } else {
            vec_quadratic_peak_pos(&self.phout, maxindex)
        };

        // Take back one frame delay
        phase += 1.0;

        // Reset output
        for val in output.iter_mut() {
            *val = 0.0;
        }

        let mut i = 1;
        let mut beat = bp - phase;

        // Skip beat if too close
        let skip_cond = (step as f32 - self.lastbeat - phase) < -0.40 * bp;
        if skip_cond {
            beat += bp;
        }

        // Start counting beats
        while beat + bp < 0.0 {
            beat += bp;
        }

        if beat >= 0.0 && i < output.len() {
            output[i] = beat;
            i += 1;
        }

        while beat + bp <= step as f32 && i < output.len() {
            beat += bp;
            output[i] = beat;
            i += 1;
        }

        self.lastbeat = beat;
        // Store number of beats as first element
        output[0] = i as f32;
    }

    /// Check beat tracking state and update bp
    /// Transcribed from aubio/src/tempo/beattracking.c:338-461
    fn checkstate(&mut self) {
        let laglen = self.rwv.len();
        let acflen = self.acf.len();
        let step = self.step;

        let mut counter = self.counter;
        let mut flagstep = self.flagstep;
        let mut gp = self.gp;
        let rp = self.rp;
        let mut rp1 = self.rp1;
        let mut rp2 = self.rp2;
        let mut flagconst = false;
        let mut bp;

        // If gp is set, compute shift invariant comb filterbank
        if gp > 0.0 {
            // Zero out acfout
            for val in self.acfout.iter_mut() {
                *val = 0.0;
            }

            // Compute shift invariant comb filterbank
            for i in 1..(laglen - 1) {
                for a in 1..=self.timesig {
                    for b in 1..(2 * a) {
                        let idx = i * a as usize + b as usize - 1;
                        if idx < acflen {
                            self.acfout[i] += self.acf[idx];
                        }
                    }
                }
            }

            // Apply Gaussian weighting
            vec_weight(&mut self.acfout, &self.gwv);

            // Find new gp
            let maxindex = vec_max_elem(&self.acfout);
            gp = vec_quadratic_peak_pos(&self.acfout, maxindex);
        } else {
            // Still only using general model
            gp = 0.0;
        }

        // Look for step change - difference between gp and rp > 2*g_var
        // Always true in first case, since gp = 0
        if counter == 0 {
            if (gp - rp).abs() > 2.0 * self.g_var {
                flagstep = 1; // Have observed step change
                counter = 3; // Setup 3 frame counter
            } else {
                flagstep = 0;
            }
        }

        // 3rd frame after flagstep initially set
        if counter == 1 && flagstep == 1 {
            // Check for consistency between previous beat period values
            if (2.0 * rp - rp1 - rp2).abs() < self.g_var {
                // Can activate context dependent model
                flagconst = true;
                counter = 0; // Reset counter and flagstep
            } else {
                // Not consistent, don't flag consistency
                flagconst = false;
                counter = 2; // Let it look next time
            }
        } else if counter > 0 {
            counter -= 1;
        }

        // Update history
        rp2 = rp1;
        rp1 = rp;

        // Determine bp based on state
        if flagconst {
            // First run of new hypothesis
            gp = rp;
            self.timesig = Self::get_timesig(&self.acf, gp as usize);

            // Compute Gaussian weighting
            for j in 0..laglen {
                let diff = (j + 1) as f32 - gp;
                self.gwv[j] = (-0.5 * diff * diff / (self.g_var * self.g_var)).exp();
            }

            bp = gp;

            // Flat phase weighting
            for val in self.phwv.iter_mut() {
                *val = 1.0;
            }
        } else if self.timesig > 0 {
            // Context dependent model
            bp = gp;

            // Gaussian phase weighting
            if step as f32 > self.lastbeat {
                for j in 0..(2 * laglen) {
                    let diff = 1.0 + j as f32 - step as f32 + self.lastbeat;
                    self.phwv[j] = (-0.5 * diff * diff / (bp / 8.0)).exp();
                }
            } else {
                // Flat phase weighting
                for val in self.phwv.iter_mut() {
                    *val = 1.0;
                }
            }
        } else {
            // Initial state
            bp = rp;
            // Flat phase weighting
            for val in self.phwv.iter_mut() {
                *val = 1.0;
            }
        }

        // If tempo is > 206 bpm, double it
        // C applies this unconditionally after all state logic
        while bp > 0.0 && bp < 25.0 {
            bp *= 2.0;
        }

        // Update state
        self.counter = counter;
        self.flagstep = flagstep;
        self.gp = gp;
        self.bp = bp;
        self.rp1 = rp1;
        self.rp2 = rp2;
    }

    /// Get current BPM estimate
    /// Transcribed from aubio/src/tempo/beattracking.c:424-431
    fn get_bpm(&self) -> f32 {
        if self.bp != 0.0 {
            let period_samples = self.hop_size as f32 * self.bp;
            let period_s = period_samples / self.samplerate as f32;
            60.0 / period_s
        } else {
            0.0
        }
    }
}

/*
 * Helper functions for silence detection
*/

/// Compute next power of two >= a
/// Transcribed from aubio/src/mathutils.c:590-595
fn next_power_of_two(a: usize) -> usize {
    let mut i = 1;
    while i < a {
        i <<= 1;
    }
    i
}

/// Compute mean energy (sum of squares / length)
/// Transcribed from aubio/src/mathutils.c:328-340
fn level_lin(data: &[f32]) -> f32 {
    let mut energy = 0.0;
    for &x in data {
        energy += x * x;
    }
    energy / data.len() as f32
}

/// Compute dB SPL (10 * log10 of mean energy)
/// Transcribed from aubio/src/mathutils.c:609-612
fn db_spl(data: &[f32]) -> f32 {
    10.0 * level_lin(data).log10()
}

/// Silence detection (returns true if dB SPL < threshold)
/// Transcribed from aubio/src/mathutils.c:615-618
fn is_silence(data: &[f32], threshold: f32) -> bool {
    db_spl(data) < threshold
}

/*
 * Tempo Coordinator - Integrates all components
*/

/// Tempo detection coordinator
/// Transcribed from aubio/src/tempo/tempo.c:32-54, 57-103, 166-230
pub(crate) struct Tempo {
    // Components
    pv: PVocTempo,
    od: SpecFlux,
    pp: PeakPicker,
    bt: BeatTracking,

    // Buffers
    fftgrain: Vec<f32>, // PVoc output (257 norm + 257 phase = 514)
    of: f32,            // Onset detection value (single float)
    dfframe: Vec<f32>,  // Detection function buffer (winlen samples)
    out: Vec<f32>,      // Beat positions output (step elements)
    onset: f32,         // Peak picker output (single float)

    // State
    silence: f32,        // Silence threshold in dB
    blockpos: isize,     // Current position in dfframe (-1 to step-1)
    winlen: usize,       // dfframe buffer size
    step: usize,         // dfframe hop size (winlen/4)
    hop_size: usize,     // FFT hop size
    total_frames: usize, // Total frames processed
    last_beat: usize,    // Time of last beat in samples
    cycle_count: usize,  // Number of beattracking cycles completed
}

impl Tempo {
    /// Create new tempo detector
    /// Transcribed from aubio/src/tempo/tempo.c:166-230
    pub fn new(buf_size: usize, hop_size: usize, samplerate: u32) -> BlissResult<Self> {
        // Validate parameters (from tempo.c:173-185)
        if hop_size < 1 {
            return Err(BlissError::AnalysisError(
                "error while loading aubio tempo object: creation error".to_string(),
            ));
        }
        if buf_size < 2 {
            return Err(BlissError::AnalysisError(
                "error while loading aubio tempo object: creation error".to_string(),
            ));
        }
        if buf_size < hop_size {
            return Err(BlissError::AnalysisError(
                "error while loading aubio tempo object: creation error".to_string(),
            ));
        }
        if samplerate < 1 {
            return Err(BlissError::AnalysisError(
                "error while loading aubio tempo object: creation error".to_string(),
            ));
        }

        // Calculate winlen: next power of 2 >= (5.8 * samplerate / hop_size)
        // This gives about 6 seconds of detection function data
        let mut winlen = next_power_of_two(((5.8 * samplerate as f32) / hop_size as f32) as usize);
        if winlen < 4 {
            winlen = 4;
        }
        let step = winlen / 4;

        // Create components
        let pv = PVocTempo::new(buf_size, hop_size)?;
        let od = SpecFlux::new(buf_size);
        let mut pp = PeakPicker::new();
        pp.set_threshold(0.3); // Set threshold to match C aubio tempo default
        let bt = BeatTracking::new(winlen, hop_size, samplerate);

        // Allocate buffers
        let fftgrain = vec![0.0; (buf_size / 2 + 1) * 2]; // 257*2 = 514 for buf_size=512
        let dfframe = vec![0.0; winlen];
        let out = vec![0.0; step];

        Ok(Tempo {
            pv,
            od,
            pp,
            bt,
            fftgrain,
            of: 0.0,
            dfframe,
            out,
            onset: 0.0,
            silence: -90.0,
            blockpos: 0,
            winlen,
            step,
            hop_size,
            total_frames: 0,
            last_beat: 0,
            cycle_count: 0,
        })
    }

    /// Process one hop of audio
    /// Transcribed from aubio/src/tempo/tempo.c:57-103
    pub fn do_(&mut self, input: &[f32]) -> BlissResult<f32> {
        let winlen = self.winlen;
        let step = self.step;

        // 1. Phase vocoder
        self.pv.do_(input, &mut self.fftgrain)?;

        // 2. Onset detection (specflux)
        let fftgrain_norm = &self.fftgrain[0..(self.fftgrain.len() / 2)];
        self.of = self.od.do_(fftgrain_norm);

        // 3. Every step frames, run beat tracking and rotate buffer
        if self.blockpos == (step as isize) - 1 {
            // Run beat tracking
            self.bt.do_(&self.dfframe, &mut self.out);

            self.cycle_count += 1;

            // Rotate dfframe left by step samples
            for i in 0..(winlen - step) {
                self.dfframe[i] = self.dfframe[i + step];
            }
            // Zero out the last step samples
            for i in (winlen - step)..winlen {
                self.dfframe[i] = 0.0;
            }

            self.blockpos = -1;
        }

        self.blockpos += 1;

        // 4. Peak picker
        self.onset = self.pp.do_(self.of);

        // 5. Store thresholded value in dfframe
        let thresholded = self.pp.get_thresholded();

        self.dfframe[winlen - step + self.blockpos as usize] = thresholded;

        // 6. Check for beat at current position
        let mut tempo_out = 0.0;
        let num_beats = self.out[0] as usize;

        for i in 1..num_beats {
            let beat_pos = self.out[i];

            if self.blockpos == beat_pos.floor() as isize {
                // Found a beat at current position
                tempo_out = beat_pos - beat_pos.floor();

                // Test for silence - if silent, unset beat
                if is_silence(input, self.silence) {
                    tempo_out = 0.0;
                } else {
                    // Update last_beat timestamp
                    self.last_beat =
                        self.total_frames + (tempo_out * self.hop_size as f32).round() as usize;
                }
            }
        }

        self.total_frames += self.hop_size;

        Ok(tempo_out)
    }

    /// Get current BPM estimate
    /// Transcribed from aubio/src/tempo/tempo.c:232-234
    pub fn get_bpm(&self) -> f32 {
        self.bt.get_bpm()
    }
}
