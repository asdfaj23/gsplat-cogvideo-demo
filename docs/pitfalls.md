# 3DGS + CogVideoX-2B 驾驶场景 demo 踩坑记录

## 1) 多卡吞吐没有提升

**现象**
- 卡数增加后，step time 几乎不变甚至更慢。

**常见原因**
- `per_gpu_batch_size` 太小导致通信开销占比高。
- dataloader worker 不足，GPU 等数据。

**建议**
- 优先把总 batch 做到 `gpus * per_gpu_batch_size >= 4`（显存允许前提下）。
- 为数据加载与解码单独做 profiling（CPU 利用率、I/O 带宽）。

---

## 2) 画面时序抖动（闪烁）明显

**现象**
- 单帧质量可接受，但播放时边缘和高频纹理跳动。

**常见原因**
- diffusion step 太低。
- 相机轨迹扰动太大，邻帧视角变化超出模型稳定区间。

**建议**
- 提高 `diffusion_steps`，并固定评测 prompt 和 seed。
- 使用 `scripts/temporal_consistency.py` 对不同配置统一打分。

---

## 3) 显存溢出（OOM）

**现象**
- 高分辨率 + 多帧生成时 OOM。

**建议**
- 降低 `resolution` 或 `num_frames`。
- 开启混合精度（fp16/bf16）并检查无意的 fp32 回退。
- 检查是否有未释放的中间缓存（尤其是可视化/日志代码路径）。
