use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder, VarMap};
use std::time::Instant;

const HIDDEN: usize = 4096;
const BATCH: usize = 2048;
const WARMUP: usize = 5;
const ITERATIONS: usize = 100;

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(HIDDEN, HIDDEN, vb.pp("fc1"))?,
            fc2: linear(HIDDEN, HIDDEN, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.relu()?;
        self.fc2.forward(&x)
    }
}

fn ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn main() -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to open CUDA device 0: {e}");
            eprintln!("Need: NVIDIA driver + CUDA Toolkit + candle built with --features cuda");
            eprintln!("For RTX 50-series (sm_120), try: set CUDA_COMPUTE_CAP=120");
            return Err(e);
        }
    };
    println!("Using device: {device:?}\n");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Mlp::new(vb)?;
    let x = Tensor::randn(0f32, 1.0, (BATCH, HIDDEN), &device)?;

    // Warmup: load CUDA kernels
    for _ in 0..WARMUP {
        let _ = model.forward(&x)?;
    }
    device.synchronize()?;

    // BENCHMARK 1: WRONG (Asynchronous Trap)
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = model.forward(&x)?;
    }
    let time_wrong = start.elapsed();

    // BENCHMARK 2: CORRECT (CPU Sync + Timer)
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = model.forward(&x)?;
    }
    device.synchronize()?;
    let time_sync = start.elapsed();

    // BENCHMARK 3: BEST PRACTICE (Candle)

    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = model.forward(&x)?;
    }
    device.synchronize()?;
    let time_best = start.elapsed();

    println!("--- BENCHMARK RESULTS COMPARISON ---");
    println!(
        "- [WRONG] Without Sync           : {:8.3} ms (Fake Result!)",
        ms(time_wrong)
    );
    println!();
    println!(
        "- [CORRECT] CPU Sync + Timer     : {:8.3} ms",
        ms(time_sync)
    );
    println!(
        "- [CORRECT] Candle synchronize() : {:8.3} ms",
        ms(time_best)
    );
    println!();
    println!("Hint: if [WRONG] << [CORRECT], the async trap is visible on this GPU.");

    Ok(())
}
