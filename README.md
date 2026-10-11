# gsplat-cogvideo-demo

3DGS 最小复现 + CogVideoX-2B 驾驶场景生成的实习 demo，包含：

- 多卡配置 sweep（`configs/sweep/multi_gpu_sweep.yaml`）
- 时序一致性测量脚本（`scripts/temporal_consistency.py`）
- 常见踩坑记录（`docs/pitfalls.md`）

## 1. 环境

> 最小依赖仅用于时序一致性评估；训练/推理由你当前 3DGS + CogVideoX 环境负责。

```bash
python -m pip install -U pip
python -m pip install numpy pillow
```

## 2. 多卡 sweep 模板

`sweep` 文件定义了不同 GPU 卡数、分辨率、batch 和采样步数的推荐组合，可直接用于批量跑实验（按需改成你使用的调度器/启动器格式）。

```bash
cat /home/runner/work/gsplat-cogvideo-demo/gsplat-cogvideo-demo/configs/sweep/multi_gpu_sweep.yaml
```

建议优先比较：

1. 单卡基线（1xGPU）
2. 双卡吞吐折中（2xGPU）
3. 四卡高分辨率（4xGPU）

## 3. 时序一致性测量

输入一段生成视频帧目录（按文件名排序，如 `0001.png, 0002.png, ...`）：

```bash
python /home/runner/work/gsplat-cogvideo-demo/gsplat-cogvideo-demo/scripts/temporal_consistency.py \
  --frames_dir /abs/path/to/frames \
  --output /abs/path/to/temporal_metrics.json
```

输出指标：

- `mean_l1`: 邻帧归一化像素 L1（越小越稳定）
- `mean_grad_l1`: 邻帧 Sobel 梯度 L1（越小越稳定，边缘闪烁更敏感）
- `temporal_score`: `1 / (1 + mean_l1 + 0.5 * mean_grad_l1)`（越大越好）

## 4. 踩坑记录

详见：`docs/pitfalls.md`

## 5. 最小复现实习 demo 验收建议

- 固定一个驾驶场景 prompt，分别跑 1/2/4 卡配置
- 对每组输出视频抽帧后跑 `temporal_consistency.py`
- 对比吞吐（it/s、总时长）与 `temporal_score`
- 在 `docs/pitfalls.md` 中持续补充异常案例和修复方案
