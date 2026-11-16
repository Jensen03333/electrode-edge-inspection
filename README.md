# 极片涂层细裂纹的端侧检测

这个项目面向锂电极片的在线外观检查。输入一张涂层图像，程序在 x86 CPU 上找出表面裂纹、保存像素掩膜，再用真实逐帧耗时回放相机触发、排队、超时和丢帧。训练数据是台式狭缝涂布实验中的碳黑/Nafion 涂层。工厂极片上的检出率需要另行测量。[实验材料与成像方法](https://doi.org/10.1038/s41597-025-06419-1)

### 本机实测

测试集来自一个完整留出的采集设置，共 275 张图，其中 236 张带表面裂纹。i7-12700H，CPU 单线程，单帧推理，固定阈值。逐帧计时包括文件解码、缩放、推理、恢复原图尺寸和生成判定，摄像头传输与保存图片另计。每个模型预热后对每张图计时一次，两个模型交替先后运行。

| 指标 | FP32 | 静态 INT8 |
| --- | ---: | ---: |
| ONNX 文件字节数 | 123,517 | 56,991 |
| 测试集像素 Dice | 0.7511 | 0.7529 |
| 裂纹像素召回 | 69.88% | 70.03% |
| 无此类裂纹的 39 张图中触发报警的张数 | 33 | 32 |
| 端到端 P95（毫秒） | 25.67 | 38.17 |
| 30 帧/秒工位仿真，275 次触发中按时完成 | 275 | 167 |

当任意像素被判为裂纹，当前基线即报整图有裂纹。误报计数反映的是这个工作点。INT8 使文件缩小约 54%，CPU 执行记录确认 11 个卷积运行了量化算子。在这台机器及这轮设置下，有时限的工位优先选 FP32。30 帧/秒仿真设 1 个处理位、2 个等待位，从触发算起 66.67 毫秒截止。INT8 的另外 108 帧中，107 帧超时、1 帧因队列已满丢弃。[逐帧指标](reports/frames/) · [CPU 实测](reports/benchmark.json) · [工位仿真](reports/replay.json)

## 第一次运行

在项目目录执行以下命令。准备脚本会从 CoatingVision 作者仓库的固定版本取回 8 对示例并校验文件，再生成独立的原图、二值标注和叠加图。

```powershell
python -m pip install numpy pillow
python prepare_samples.py
```

打开 `data/coatingvision-demo/overlays/sample1.png`。左半边是原图，右半边的红色表示参考标注。试着在左半边指出同一条细裂纹，并找到它最难辨认的一段。用 `data/coatingvision-demo/manifest.csv` 核对尺寸与像素数。这里的 8 张图用于检查读图与标注解析。[动手做第一课](LAB.md)

仓库自带两份训练好的 ONNX。可以立即对自己的 JPG 或 PNG 做 CPU 推理，训练环境可单独安装。

```powershell
python -m pip install -r requirements-edge.txt
python inspect_frame.py --model models/int8.onnx --input data/coatingvision-demo/frames --out runs/first-inspection
```

`runs/first-inspection/` 中有二值掩膜、预测叠加图和含逐帧耗时的 CSV。此处的红色是模型预测。可以给 `--input` 传单张图片或图片目录。

## 项目各阶段

1. 核对原图、标注通道与来源，用来源分组划分训练、验证、量化校准和测试数据。
2. 训练输出单通道裂纹分数图的分割模型，保存验证集上选出的权重，导出 FP32 ONNX。
3. 用校准集得到 INT8 模型。在同一 CPU 和测试集上比较两份模型的裂纹像素召回、无裂纹图误报、Dice 与端到端耗时。
4. 让仿真工位按固定间隔触发图像，把逐帧实测耗时送入有限容量的队列，记录每帧按时完成、超时或丢弃的原因。
5. C++ 参考程序从预解码的灰度帧开始，在 Windows x64 上独立构造输入张量并运行 ONNX Runtime。对一张测试图分别核对 FP32 和 INT8，每份模型的 76,800 个输出分数与同版本 Python 参考端完全一致。JPEG 读入和缩放由 Python 完成。[C++ 编译与核对步骤](native/README.md) · [实测记录](reports/native.json)

## 复现实验

完整运行需要从 Figshare 下载 `CoatingVision.zip`。解包脚本校验作者发布的 MD5，并按采集设置分组。训练、验证、校准、测试图数为 1,345、303、304、275。相邻图块所属的同一采集设置保留在同一集合中。文件下载后，从项目根目录执行下面的命令，把第一个路径换成实际位置。[完整数据](https://doi.org/10.6084/m9.figshare.29260121.v1) · [划分报告](reports/dataset.json)

```powershell
python prepare_dataset.py --archive "C:/path/to/CoatingVision.zip"
python train_model.py --data-root data/full --train data/full/train.csv --val data/full/val.csv --output models --epochs 12 --batch-size 8
python quantize_model.py --fp32 models/fp32.onnx --calibration data/full/calibration.csv --evaluation data/full/test.csv --data-root data/full --output models/int8.onnx
python benchmark_model.py --fp32 models/fp32.onnx --int8 models/int8.onnx --evaluation data/full/test.csv --data-root data/full --output runs/benchmark --threads 1
python station_replay.py --images data/full/test/images --timings reports/frames/int8.csv --arrival-ms 33.333333333333336 --deadline-ms 66.66666666666667 --queue-capacity 2 --out runs/replay-first
python review_errors.py
```

训练与导出实测环境为 Python 3.10、PyTorch 2.5.1+cu121、ONNX 1.17.0。量化与报告采用 Python 3.12、ONNX 1.23.2、ONNX Runtime 1.31.1、NumPy 2.2.6 和 Pillow。训练额外依赖 PyTorch 与 ONNX，量化额外依赖 ONNX。可分别安装到两个虚拟环境。复现本机已发表的时序回放可使用 `reports/frames/` 中的 CSV；测另一台设备时，用它自己的 `runs/benchmark/` 替换。

## 查看漏检

[裂纹像素召回最低的帧](reports/errors/lowest-int8-recall.png) · [量化后召回下降最多的帧](reports/errors/largest-recall-loss.png) · [无此类裂纹但误报像素最多的帧](reports/errors/largest-negative-alarm.png)。每幅图从左至右为原图、FP32、INT8。绿色是命中，红色是漏检，蓝色是误报。选择规则写在 `review_errors.py` 中；这些图用于解释已冻结模型的结果。

## 换成真实锂电极片照片

冻结两份模型后，用 Schoo 等人发表的两张中试线极片图进行跨材料检查。暗色主裂纹上有连续响应，也有背景散点；亮色泥裂上的响应较断续。两张出版图片没有像素级真值，因此这里展示预测叠图，尚无真实锂电极片检出率。[外部样例、出处与复现](reports/external/README.md)

对照图由本项目从 Vignesh Sampath、Andrew S. Lee、Samuel David Miller、Noah H. Paulson、Yuepeng Zhang 与 Logan Ward 发布的 CoatingVision v1 图像及标注制作，添加了预测叠加与文字说明。原数据使用 CC BY 4.0。[数据出处](https://doi.org/10.6084/m9.figshare.29260121.v1) · [许可](https://creativecommons.org/licenses/by/4.0/) · [作者样本仓库](https://github.com/vigsam-coder/CoatingVision)
