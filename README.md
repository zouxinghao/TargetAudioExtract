#### 各部分功能
+ config.py 配置参数
+ util.py 包含各种辅助处理函数
+ fault_detect.py 处理故障检测
+ denoising.py 处理降噪
+ nn.py 神经网络模型
+ bss_eval.m及bss_eval bss_eval性能评估工具

#### 环境
python3.5, Ubuntu16.04, Matlab R2016b.

Matlab用于降噪部分调用BSS_EVAL工具包。

#### 依赖
+ numpy
+ sklearn
+ resampy
+ librosa
+ matlab engine
+ stft

#### 未知问题
import matlab.engine 必须在 tensorflow 之前
