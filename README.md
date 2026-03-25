<h1 align="center"><em>TrajLoom</em>: Dense Future Trajectory Generation from Video</h1>

<p align="center">
  <strong>Zewei Zhang<sup>1</sup></strong>
  &nbsp;&nbsp;
  <strong>Jia Jun Cheng Xian<sup>2,3</sup></strong>
  &nbsp;&nbsp;
  <strong>Kaiwen Liu<sup>2,3</sup></strong>
  &nbsp;&nbsp;
  <strong>Ming Liang<sup>4</sup></strong>
  &nbsp;&nbsp;
  <strong>Hang Chu<sup>4</sup></strong>
  &nbsp;&nbsp;
  <a href="https://www.ece.mcmaster.ca/~junchen/"><strong>Jun Chen<sup>1</sup></strong></a>
  &nbsp;&nbsp;
  <a href="https://lrjconan.github.io/"><strong>Renjie Liao<sup>2,3,5</sup></strong></a>
</p>

<div align="center">
  <sup>1</sup>McMaster University &nbsp;&nbsp;
  <sup>2</sup>University of British Columbia &nbsp;&nbsp;
  <sup>3</sup>Vector Institute &nbsp;&nbsp;
  <sup>4</sup>Viggle AI &nbsp;&nbsp;
  <sup>5</sup>Canada CIFAR AI Chair
</div>

<p></p>

<div align="center">
  <a href="https://trajloom.github.io/"><img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2603.22606"><img src="https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv Preprint"></a>
  <a href="https://huggingface.co/zeweizhang/TrajLoom"><img src="https://img.shields.io/badge/HuggingFace-Model-ffbd45.svg?logo=huggingface&logoColor=ffbd45" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/datasets/zeweizhang/TrajLoomDatasets"><img src="https://img.shields.io/badge/HuggingFace-Dataset-ffbd45.svg?logo=huggingface&logoColor=ffbd45" alt="Hugging Face Dataset"></a>
</div>

<p align="center">
  <img src="./assets/Trajloom.gif" width="96%" alt="TrajLoom teaser GIF" />
</p>


---

## 🔧 Environment

A CUDA-capable GPU is recommended for training, inference, and benchmarking.

```bash
bash install_env.sh
```

> If you prefer manual installation, use `install_env.sh` as the source of truth for dependencies and environment setup.

---

## 📦 Data and Config

The data and model checkpoints can be downloaded from the Hugging Face links above. After downloading them locally, update the paths in the three config files below.

### Config files

- `configs/trajloom_vae_config.json`: VAE config
- `configs/trajloom_generator_config.json`: future-trajectory generator config
- `configs/vis_predictor_config.json`: visibility predictor config

If you build your own cached trajectory latents, also recompute the latent statistics and update `configs/vae_latent_stats.json` accordingly.

---

## 🚀 Inference

### VAE Reconstruction Sanity Check

Use the VAE reconstruction script first to verify that your trajectory data, visibility masks, and latent statistics are configured correctly. Replace the placeholder paths below with your local paths.

```bash
python "/path/to/TrajLoom/run_trajloom_vae_recon.py" \
  --config "/path/to/TrajLoom/configs/trajloom_vae_config.json" \
  --video_dir "/path/to/videos/" \
  --video_glob "*.mp4" \
  --gt_dir "/path/to/ground_truth/tracks/" \
  --out_dir "/path/to/output/" \
  --pred_len 81 \
  --save_video
```

Expected output: reconstructed trajectories and/or visualizations saved to the output directory specified in the config.

### Future Trajectory Generation

Once the VAE setup is working, run the generator to predict future trajectories from observed history. Replace the placeholder paths below with your local paths.

```bash
python "/path/to/TrajLoom/run_trajloom_generator.py" \
  --gen_config "/path/to/TrajLoom/configs/trajloom_generator_config.json" \
  --gen_ckpt "/path/to/checkpoints/trajloom_generator.pt" \
  --vis_config "/path/to/TrajLoom/configs/vis_predictor_config.json" \
  --vis_ckpt "/path/to/checkpoints/visibility_predictor.pt" \
  --video_dir "/path/to/videos/" \
  --video_glob "*.mp4" \
  --gt_dir "/path/to/ground_truth/tracks/" \
  --out_dir "/path/to/output/" \
  --pred_len 81
```

Expected output: predicted future trajectories, visibility predictions, and rendered visualizations in the configured output directory.

---

## 📈 Benchmark and Visualization

After running the inference commands above, you can use the saved outputs in `--out_dir` to render qualitative results and compute benchmark metrics.

### Render Trajectories

```bash
python -m benchmark.render_trajectory \
  --root_dir "/path/to/benchmark_outputs/" \
  --video_id <video_id> \
  --out_dir "/path/to/render_outputs/<video_id>"
```

### For VAE reconstruction outputs, compute VEPE with:

```bash
python -m benchmark.compute_vae_recon_metrics \
  --runs_root "/path/to/vae_recon_runs/" \
  --out_json "/path/to/vae_recon_metrics.json"
```

### For generated trajectory outputs, compute trajectory quality metrics with:

```bash
python -m benchmark.compute_trajectory_quality_metrics \
  --runs_root "/path/to/trajectory_runs/" \
  --use_visibility \
  --out_json "/path/to/trajectory_quality_metrics.json"
```

And compute FVMD with:

```bash
python -m benchmark.compute_fvmd_from_trajectory \
  --runs_root "/path/to/fvmd_runs/" \
  --clip_len 24 --clip_stride 1 \
  --use_gt_visibility \
  --out_json "/path/to/fvmd_tracks.json"
```

---

## 🏋️ Training

**TODO:** training code will be released later.

Planned public training components:

- TrajLoom-VAE training
- TrajLoom-Flow training


---

## 🙏 Acknowledgements

We gratefully acknowledge <a href="https://github.com/gboduljak/what-happens-next">What Happens Next? Anticipating Future Motion by Generating Point Trajectories (WHN)</a> as the most important baseline for this work. Parts of our model implementation are adapted from their open-source code.

We thank <a href="https://github.com/aharley/alltracker">AllTracker</a> for providing the dense tracking system used to extract trajectories in our pipeline.

We also acknowledge <a href="https://github.com/DSL-Lab/FVMD-frechet-video-motion-distance">FVMD: Fréchet Video Motion Distance</a> for the motion-consistency metric formulation used in our evaluation.

---

## 📚 Citation

```bibtex
@misc{zhang2026trajloomdensefuturetrajectory,
      title={TrajLoom: Dense Future Trajectory Generation from Video}, 
      author={Zewei Zhang and Jia Jun Cheng Xian and Kaiwen Liu and Ming Liang and Hang Chu and Jun Chen and Renjie Liao},
      year={2026},
      eprint={2603.22606},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.22606}, 
}
```
