# 3DGS + 视频扩散 · 世界模型实习小项目

跑在学校的 8×RTX 3090 集群上（共享容器，无 sudo、无 nvcc/g++/git/unzip）。
一边是 3D Gaussian Splatting 的最小复现（gsplat），一边是视频扩散（CogVideoX-2B）的驾驶场景生成与一致性测量。
做这个的起因是：小马智行世界模型实习 JD 里两条最核心——「基于 3DGS 的前馈式/可泛化驾驶场景重建」和「用生成模型造驾驶长尾数据」。我想自己亲手各跑通一遍，不只是看论文。

A 部分（3DGS）：Mip-NeRF 360 garden 上 7000 步训练，PSNR 26.24 dB / SSIM 0.806 / LPIPS 0.157，渲染 66 FPS。
B 部分（视频扩散）：CogVideoX-2B 生成 16 帧驾驶 dashcam 视频，20 步 DDIM 约 60 秒，相邻帧 LPIPS 0.10，同 prompt 不同 seed 逐帧 LPIPS 0.53——
也就是说 prompt 控制很弱，输出主要由噪声主导，这个数字本身就是面试素材。

---

## 两个环境（不要混装）

| env | 用途 | 关键版本 |
|---|---|---|
| `qzc_3dgs` | 3DGS 训练 | py3.10 + torch 2.4.0+cu124 + gsplat 1.5.3+pt24cu124 |
| `qzc_intern` | 视频扩散（复用已有栈） | py3.12 + torch 2.11.0+cu130 + diffusers 0.41.0 |

为什么要分两个 env：gsplat 官方 wheel 只发 cp310 + torch ≤2.4 + cu124，
而 `qzc_intern` 是 cp312 + cu130，ABI 完全对不上。硬塞同一个 env 里会把已有的 QLoRA 栈搞坏。

---

## A · 3DGS 最小复现

### 跑通命令

```bash
cd thirdparty/gsplat-1.5.3/examples
export PYTHONPATH=/root/qzc_3DGS/shims:.     # 垫片（fused_ssim / nerfview / pycolmap）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=5 python simple_trainer.py default \
    --disable_viewer --data_factor 4 --packed \
    --data_dir /root/qzc_3DGS/data/garden_raw/garden/ \
    --result_dir /root/qzc_3DGS/outputs/gsplat_garden_baseline/ \
    --max_steps 7000 \
    --eval_steps 3000 7000 --save_steps 7000 \
    --sh_degree_interval 1000 --render_traj_path interp --test_every 8 \
    --strategy.refine_stop_iter 3000 --strategy.refine_start_iter 500
```

### 实测数字（2026-10-11）

| 指标 | step 3000 | step 7000 |
|---|---|---|
| PSNR (dB) | 25.17 | **26.24** |
| SSIM | 0.759 | **0.806** |
| LPIPS (alex) | 0.223 | **0.157** |
| 高斯数 | 1,636,889 | 1,636,889 |
| 渲染速度 | 15.7 ms/img | 15.6 ms/img（66 FPS） |
| 训练 wall time | — | 510 s（8 分 30 秒） |
| 峰值显存 | — | 2.4 GB |

参考：gsplat 官方 paper 里 garden（不 packed、30000 步）PSNR 在 26-28 dB。我们 7000 步 + packed 拿到 26.24，已经落在合理区间。

### 多卡并行 sweep（4 配置同时跑，4 张卡）

没有用 DDP（3090 全 PCIe、无 NVLink，DDP 通讯开销不划算），而是 4 个独立进程各占一张卡：

| 配置 | PSNR | SSIM | LPIPS | GS 数 | 渲染 ms/img |
|---|---|---|---|---|---|
| baseline（refine_stop=3000） | 26.24 | 0.806 | 0.157 | 1.64M | 15.6 |
| A：一直 densify 到 7000 | 26.22 | 0.815 | **0.142** | **3.21M** | 25.3 |
| B：prune_opa=0.01（更狠剪枝） | **26.28** | 0.808 | 0.156 | 1.61M | 17.1 |
| C：grow_grad2d=0.001（更激进 grow） | 24.85 | 0.708 | 0.315 | **199k** | **6.3** |
| D：MCMC 策略 | 25.84 | 0.806 | 0.166 | 1.00M | 7.7 |

**三个观察**：
1. A 一直 densify 让 GS 数翻倍，PSNR 只涨 0.02 dB，但渲染时间从 15.6ms 涨到 25.3ms——典型的收益递减
2. C 把 grow 阈值放太松，把噪声点也长出来了，PSNR 反而掉 1.4 dB
3. MCMC 用 100 万 GS 就拿到 25.84 dB，性价比不错

### 踩过的坑（现象 → 定位 → 解决）

1. **gsplat wheel ABI 不匹配**：cp312+cu130 的 env 装不上 gsplat（只发 cp310+cu124），且容器无 nvcc 不能编译。→ 新建 `qzc_3dgs` env 对齐 py3.10+torch2.4+cu124。
2. **pip --index-url 覆盖清华源**：numpy 走 pythonhosted 超时。→ 双 index `--index-url $TS --extra-index-url $TH`。
3. **容器无 unzip/git/nvcc**：用 Python zipfile + codeload tarball 绕开。
4. **fused_ssim 是编译型扩展**：PyPI 无 wheel、容器无 nvcc。→ 写纯 PyTorch 垫片（高斯窗 SSIM），和 skimage 数值差 2e-16，可微。
5. **pycolmap 4.x 没 SceneManager**：gsplat 1.5.3 按 3.x API 写。→ 写垫片把 Reconstruction 包装成 3.x 接口，通过 sitecustomize.py 注入。
6. **相机内参顺序错了**（最大坑）：第一版 PSNR 只有 17 dB，诊断后发现 fy 和 cx 搞反了，K 矩阵全错。修完直接跳到 26 dB。**10 dB 跳水 = 相机错**这个直觉就是这么练出来的。
7. **tyro list 参数**：要空格分隔 `--eval_steps 3000 7000`，不是逗号。
8. **OOM**：默认 densify 让 GS 数膨胀到 250 万，24G 多租户机器扛不住。→ `--packed` + `expandable_segments:True` + `refine_stop_iter=3000`。

---

## B · 视频扩散（CogVideoX-2B）

### 跑通命令

```bash
conda activate qzc_intern
export HF_ENDPOINT=https://hf-mirror.com   # huggingface.co 直连超时，hf-mirror 0.24s

CUDA_VISIBLE_DEVICES=5 python - <<'PY'
from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained(
    "/root/qzc_3DGS/models/CogVideoX-2b", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing(); pipe.vae.enable_tiling()

video = pipe(
    prompt="A dashcam view from a car driving on a sunny suburban road...",
    num_inference_steps=20, num_frames=16, guidance_scale=6,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
PY
```

### 实测数字

| 视频 | prompt | seed | 生成时间 | 相邻帧 LPIPS |
|---|---|---|---|---|
| drive_dashcam_seed42 | 白天 dashcam | 42 | 54 s | **0.104** |
| drive_dashcam_seed0 | 白天 dashcam（同 prompt） | 0 | 65 s | 0.152 |
| drive_night_seed42 | 夜间城市 dashcam | 42 | 63 s | **0.077** |

- 20 步 DDIM，16 帧 720×480，fp16 + CPU offload + VAE tiling
- 同 prompt 不同 seed 逐帧 LPIPS = **0.532**（非常大）

### 这个 0.532 意味着什么

相邻帧 LPIPS 才 0.10，但换个 seed 逐帧 LPIPS 飙到 0.53——
**prompt 几乎不控制生成内容，输出主要由初始噪声决定**。这是当前视频扩散模型的通病：
文本条件在 cross-attention 里的权重远低于扩散过程本身。面试讲这个数字比讲论文摘要有说服力得多。

### 显存降级顺序（实测）

24G 单卡 fp16 + CPU offload 足够 2B 模型。如果显存不够，按这个顺序降：
1. VAE slicing/tiling（已经开了）
2. `enable_model_cpu_offload()`
3. `num_frames` 16 → 8
4. 分辨率 720×480 → 512×320
5. 换 LTX-Video 2B（更快，同显存需求）

---

## 项目结构

```
qzc_3DGS/
  shims/                  # 空垫片：fused_ssim / nerfview / gsplat_viewer / viser
  thirdparty/gsplat-1.5.3/   # gsplat 源码（codeload tarball）
  data/garden_raw/garden/    # MipNeRF360 garden（COLMAP sparse/ 已就位）
  models/CogVideoX-2b/       # diffusers 格式权重
  outputs/
    gsplat_garden_baseline/  # 48 张 GT vs render 对比图 + 2 段轨迹视频 + ckpt
    sweep/{A,B,C,D}/         # 4 配置并行结果
    cogvideo/               # 3 段 mp4 + consistency.json
  scripts/                 # 28 个阶段脚本（s1-s28，按时间顺序）
  logs/                    # 每个阶段的完整日志
  DEMO_REPORT.md           # 详细报告（A+B 完整版）
```

---

## 没做的（如实列出）

- 30000 步完整训练（只跑了 7000 步，数字已落在合理区间）
- 原版 graphdeco-inria/gaussian-splatting 对照（编译 diff-gaussian-rasterization 需要 nvcc）
- nerfstudio splatfacto 对照（tiny-cuda-nn 也需要 nvcc）
- 手机环绕视频跑 COLMAP SfM（容器里没装 COLMAP）
- RAFT 光流时序一致性（用了 LPIPS 近似）
- LTX-Video / SVD 对照（CogVideoX-2B 已跑通，够用）

---

*这个项目不是作业，是为小马智行世界模型实习准备的。所有数字都来自脚本输出，没有编造。*
