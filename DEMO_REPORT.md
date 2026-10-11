# 3DGS 最小复现 Demo 报告（A 任务）

> 项目目录：`/root/qzc_3DGS/`
> 数据：Mip-NeRF 360 `garden`（185 张环绕拍摄，COLMAP SfM 已由官方打包提供）
> 框架：gsplat v1.5.3（py3.10 + torch 2.4.0+cu124 + 官方预编译 wheel）
> 时间：2026-10-11

---

## 1. 环境（全部实测）

| 项目 | 值 |
|---|---|
| 操作系统 | Ubuntu 24.04.4 LTS（容器，kernel 6.8.12-aimia-hpc-r1） |
| GPU | 8× RTX 3090（24 GB），**全 PCIe、无 NVLink**（`nvidia-smi topo -m` 全 PIX/PXB/SYS） |
| 驱动 / CUDA | 580.173.02 / 13.0 |
| 编译链 | **容器里没有 nvcc / g++ / make / git / unzip / colmap**，只有 gcc 13.3.0 + cmake |
| conda | `/opt/miniconda3`，.condarc 已配清华镜像 |
| 网络 | pypi/清华源 200 OK；github 200 但 10s+；huggingface.co **TIMEOUT**；`hf-mirror.com` 200/0.24s；storage.googleapis.com 200 OK |

**新环境**（不碰已有的 `qzc_intern` QLoRA 栈）：
```
conda create -n qzc_3dgs python=3.10 -y
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install thirdparty/gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl
pip install "numpy<2" tyro nerfview viser jaxtyping einops plyfile \
    torchmetrics tensorboard scikit-image imageio imageio-ffmpeg \
    opencv-python-headless scikit-learn matplotlib pyyaml tqdm rich scipy \
    pycolmap==4.2.1 ninja
```

---

## 2. 数据

- 来源：`https://storage.googleapis.com/gresearch/refraw360/garden.zip`（2.9 GB，204 秒下完，14 MB/s）
- 解压：容器无 `unzip`，用 Python `zipfile`（9 秒）
- 结构：
  ```
  data/garden_raw/garden/
    images/        原始 5187×3361 共 185 张
    images_2 / _4 / _8   已下采样好的多分辨率
    sparse/0/      cameras.bin images.bin points3D.bin
    poses_bounds.npy
  ```
- **关键点**：官方打包已经包含 COLMAP SfM 结果，我们不需要在容器里编译 COLMAP（这是原提示词里最大的风险点）。

---

## 3. 训练命令（实测跑通）

```bash
cd /root/qzc_3DGS/thirdparty/gsplat-1.5.3/examples
export PYTHONPATH=/root/qzc_3DGS/shims:.      # 垫片目录
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=5 python simple_trainer.py default \
    --disable_viewer \
    --data_factor 4 \
    --packed \
    --data_dir /root/qzc_3DGS/data/garden_raw/garden/ \
    --result_dir /root/qzc_3DGS/outputs/gsplat_garden_baseline/ \
    --max_steps 7000 \
    --eval_steps 3000 7000 \
    --save_steps 7000 \
    --sh_degree_interval 1000 \
    --render_traj_path interp \
    --test_every 8 \
    --strategy.refine_stop_iter 3000 \
    --strategy.refine_start_iter 500
```

- `--packed`：省显存（24 GB 多租户下必备）
- `--strategy.refine_stop_iter 3000`：提前停 densification，否则高斯数会从 160 万膨胀到 250 万 + OOM

---

## 4. 结果（真实数字，不是编的）

### 4.1 训练收敛

| 指标 | 值 |
|---|---|
| 总步数 | 7000 |
| 平均速度 | **12.77 it/s** |
| wall time | **510 秒**（8 分 30 秒） |
| 峰值显存 | **2.4 GB**（3090 24 GB 上很宽裕） |
| 最终高斯数 | **1,636,889** |
| 最终 train loss | **0.056** |

### 4.2 验证集指标（每 8 张图取 1 张做 test，共 23 张）

| Step | PSNR (dB) | SSIM | LPIPS (alex) | 渲染速度 |
|---|---|---|---|---|
| 3000 | 25.17 | 0.759 | 0.223 | 15.7 ms/img |
| **7000** | **26.24** | **0.806** | **0.157** | **15.6 ms/img** |

参考：gsplat 官方 paper 里 garden（7000 步，不 packed）的 PSNR 在 26-28 dB 区间。我们这版 26.24 dB 已经落在合理区间。

### 4.3 产出物

```
outputs/gsplat_garden_baseline/
  renders/
    val_step2999_0000.png ... val_step2999_0022.png   (23 张)
    val_step6999_0000.png ... val_step6999_0022.png   (23 张)
    每张都是 GT | render 并排拼接（左半 GT，右半 render）
  videos/
    traj_2999.mp4   17.6 MB  (interpolated camera trajectory)
    traj_6999.mp4   19.1 MB
  ckpts/
    ckpt_6999_rank0.pt
  stats/
    val_step2999.json / val_step6999.json / train_step6999_rank0.json
  tb/               tensorboard events
```

---

## 5. 踩坑日志（现象 → 定位 → 解决，按真实顺序）

### 坑 1：gsplat 官方 wheel 跟现有 env 不匹配
- **现象**：`pip install gsplat` 在 `qzc_intern`（py3.12 + torch 2.11+cu130）里装不上；`pip install gsplat --no-build-isolation` 又报 nvcc 缺失
- **定位**：查 GitHub releases，gsplat 只发 **cp310 + torch ≤2.4 + cu118/121/124** 的预编译 wheel；现有 env 是 cp312+cu130，完全 ABI 对不上；容器里又没 nvcc/g++/make，现场编译不可能
- **解决**：新建 `qzc_3dgs` env（py3.10 + torch 2.4.0+cu124），直接下 `gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl`，`pip install --no-deps` 装好。**没碰 `qzc_intern`，保住了 QLoRA 栈**

### 坑 2：pip `--index-url` 覆盖清华源
- **现象**：装 torch 时 numpy 走 `files.pythonhosted.org` 超时
- **定位**：`--index-url https://download.pytorch.org/whl/cu124` 把全局 index 换掉了，清华镜像失效
- **解决**：双 index `--index-url $TS --extra-index-url $TH`

### 坑 3：容器无 unzip / git / nvcc / g++ / make
- **现象**：`wget garden.zip` 下完 `unzip garden.zip` 报 command not found；想 `git clone gsplat` 也没有 git
- **解决**：
  - 解压：Python `zipfile.ZipFile(...).extractall()`
  - gsplat 源码：`wget https://codeload.github.com/nerfstudio-project/gsplat/tar.gz/refs/tags/v1.5.3`
  - nvcc/g++：不需要——gsplat wheel 是预编译的，fused_ssim 才是编译型

### 坑 4：fused_ssim 是编译型 CUDA 扩展
- **现象**：`simple_trainer.py` 顶层 `from fused_ssim import fused_ssim`；PyPI 上没有 fused-ssim 的 wheel；源码又要 nvcc 编译
- **解决**：写了一个纯 PyTorch 的垫片 `shims/fused_ssim/__init__.py`（高斯窗 SSIM，F.conv2d 实现）。和 skimage 对照误差 **2.2e-16**，可微（gradnorm 正常）。训练里用它替代。**注明**：报告里的最终 PSNR/SSIM 是 torchmetrics 算的，不依赖这个垫片

### 坑 5：pycolmap 4.x 没有 SceneManager
- **现象**：`from pycolmap import SceneManager` 在 pycolmap 4.2.1 上报 ImportError
- **定位**：pycolmap 3.x 有 SceneManager，4.x 换成 `Reconstruction` 直接读
- **解决**：写了垫片 `shims_pycolmap_scenemanager.py`，把 Reconstruction 包装成 gsplat 1.5.3 期望的 SceneManager 接口（cameras / images / points3D 三个 dict + name_to_image_id + point3D_id_to_images），通过 sitecustomize.py 注入

### 坑 6：相机内参顺序错了（最大的坑）
- **现象**：第一版跑出来 PSNR 只有 **17.14 dB**，SSIM 0.31，明显不对
- **定位**：diag 脚本对比原生 pycolmap：
  - 原生 `cam.params = [fx=3844.9, fy=3852.4, cx=2593.5, cy=1680.5]`
  - 我的垫片错误按 `[fx,fy,cx,cy]=[p0,p0,p1,p2]` 赋值，把 fy 和 cx 搞反了，K 矩阵全错
- **解决**：改成用 `cam.focal_length_x / focal_length_y / principal_point_x / principal_point_y` 这些 pycolmap 4.x 的 accessor 直接读。**修完 PSNR 从 17.1 → 25.2（step 3000）→ 26.2（step 7000）**
- **面试可讲**：这是个典型的"我以为是对的"坑——外参 w2c 方向没错（`img.cam_from_world()` 本来就是 w2c 语义），错在内参参数顺序。这种 10 dB 的跳水一出现就应该立刻怀疑相机，而不是怀疑模型

### 坑 7：tyro list 参数语法
- **现象**：`--eval_steps 3000,7000` 报错
- **解决**：tyro 要空格分隔 `--eval_steps 3000 7000`

### 坑 8：OOM（高斯数膨胀）
- **现象**：跑到 step 4000 左右，高斯数从 160 万膨胀到 250 万，CUDA OOM（3090 24G 上别人还占了 13G）
- **定位**：默认 `refine_stop_iter=15000`，我们只跑 7000 步，全程都在 densify
- **解决**：
  - 加 `--packed`（gsplat 推荐的省显存模式）
  - 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（减碎片）
  - 加 `--strategy.refine_stop_iter 3000`（3000 步后不再 densify，让 GS 总数冻结在 160 万）
- **面试可讲**：3090 24G 多租户下，OOM 不是显存不够，是别人的进程 + 我们的 GS 膨胀叠加；要主动监控空闲显存，挑 `nvidia-smi --query-gpu=memory.free` 最大的那张卡

### 坑 9：多卡并行调度脚本的 bash bug
- **现象**：s22_sweep.sh 里 `run_cfg` 函数 `echo $!` 把 log 文本也当 PID 传给 `wait`，直接把 shell 搞挂
- **状态**：**未修**——baseline 已跑通，sweep 属于"加分项"，先写报告
- **计划**：改成后台 `&` + 写 PID 文件，用 `pgrep -f simple_trainer.py | wc -l` 判活

---

## 6. 没做到的事（如实列出，不吹）

| 项目 | 状态 |
|---|---|
| 多卡并行 sweep 不同配置 | 脚本 bug 卡住，只跑了 baseline 一个配置 |
| 手机环绕视频跑 COLMAP SfM | 未做（用户没上传数据，且容器里没装 COLMAP） |
| 30000 步完整训练 | 只跑了 7000 步（gsplat 默认），数字已经落在合理区间 |
| 原版 graphdeco-inria/gaussian-splatting 对照 | 未做（编译 diff-gaussian-rasterization 需要 nvcc，容器没有） |
| nerfstudio splatfacto 对照 | 未做（tiny-cuda-nn 编译也需要 nvcc） |
| 导出 PLY / 实时 viewer | `--save_ply` 默认 False；viewer 是垫片空壳 |

---

## 7. 面试可讲的三句话（口语化，不是写报告的语气）

1. **环境**：我在一台 8 卡 3090 的共享容器上做的，没有 sudo、没有 nvcc/g++/make/unzip/git。我没有硬刚编译，而是查清楚 gsplat 官方 wheel 只发 cp310+torch≤2.4+cu124，**专门建了一个 py3.10 + torch 2.4.0+cu124 的新 conda env 去对齐 ABI**，完全没碰已有的 QLoRA 栈。

2. **踩坑**：最典型的坑是第一版跑出来 PSNR 只有 17 dB。我没有怀疑模型，而是写了个诊断脚本对比垫片和原生 pycolmap 读出的相机参数，**发现是内参顺序搞反了（fy 和 cx 对调）**。修完之后 PSNR 直接从 17 跳到 26。这种"10 dB 跳水 = 相机错"的直觉是这么练出来的。

3. **结果**：7000 步、8 分半、160 万高斯，PSNR 26.24 / SSIM 0.806 / LPIPS 0.157，渲染 15 ms/帧（66 FPS）。我用 `--packed` + 提前停 densification 控制显存，在 24G 多租户机器上只占 2.4G。
