# C++ 单帧端侧推理

此程序在 Windows x64 上加载 ONNX Runtime CPU 和一个裂纹分割模型，读入 320×240 的 8 位灰度 PGM，将像素归一化至 0～1，并写出 76,800 个 float32 裂纹分数（logits）。FP32 与 INT8 模型使用同一个可执行文件。生产照片的 JPEG 解码、尺寸缩放和完整批量评估由仓库的 Python 参考程序负责。

## 构建

使用 Visual Studio 2022 的 **x64 Native Tools Command Prompt**，切换到仓库根目录。下载微软发布的 [ONNX Runtime 1.20.1 Windows x64 CPU SDK](https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1)，把示例中的路径换为本机解压目录。

```bat
set "ORT=C:\sdk\onnxruntime-win-x64-1.20.1"
mkdir runs\native
cl /nologo /O2 /EHsc /std:c++17 /I"%ORT%\include" native\crack_infer.cpp /Fo:runs\native\crack_infer.obj /Fe:runs\native\crack_infer.exe
```

## 核对两个模型

在已安装 NumPy、Pillow 和 onnxruntime==1.20.1 的 Python 环境下运行下列命令。`check_parity.py` 将同一张留出测试图解码为 PGM，然后独立执行 Python ORT 和已编译的 C++ 程序，逐像素核对原始分数与阈值为零时的二值判定。

```bat
python native\check_parity.py --exe runs\native\crack_infer.exe --dll "%ORT%\lib\onnxruntime.dll" --frame data\full\test\images\image_101.jpg --fp32 models\fp32.onnx --int8 models\int8.onnx --report runs\native\parity.json
```

提交的 [单帧实测记录](../reports/native.json) 给出输入照片、模型哈希、像素总数、分数最大误差与判定差异。整套测试图上的模型质量和运行速度请看仓库的独立基准评测。

直接运行单帧推理的接口如下。输出是按行排列的原始 float32 数据，每个像素一个值，正负号对应零阈值两侧。

```bat
runs\native\crack_infer.exe "%ORT%\lib\onnxruntime.dll" models\int8.onnx frame.pgm logits.f32
```
