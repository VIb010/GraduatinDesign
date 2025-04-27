# GraduationDesign毕业设计1———图像检索系统子模块
## Acne Classification with ResNet-101 and YOLO-based Cropping(CNN模块实现痘痘分类)

### 1.项目结构
`text`
```
.
├── data.py              # 自定义 Dataset：读取 YOLO .txt 标注并裁剪小图
├── Resnet_101.py        # 训练 / 验证 / 测试脚本
├── classify.py          # 推理 & 可视化脚本
├── generate_advice.py   # 调用deepseek的api，读取output/<image_stem>/annotated/<image_stem>.json文件里的pred_label,生成advice并重写json
├── final_model.pth      # 训练完成后导出的模型权重
├── checkpoints/         # 存放训练中保存的最佳模型
├── runs/                # TensorBoard 日志
├── user_data/           # 推理时的输入  
│   ├── image/           # └── 原始待分类图片 (.jpg/.png)  
│   └── json/            #     YOLO 检测结果 (.json)  
└── output/              # 推理结果输出  
    └── <image_stem>/    
        ├── original/    #     移动后的原图 & 原始 JSON  
        └── annotated/   #     带分类标注的图片 & 新 JSON  
```

### 2.环境依赖
```
  - Python ≥3.8
  
  - PyTorch ≥1.7
  
  - torchvision
  
  - Pillow
  
  - tensorboardX

  - openai
```

### 3.数据准备
####  1.训练/验证/测试集
将您的数据集按如下结构组织（以 ./Acne 为根目录）：

```
Acne/
  ├── train/
  │   ├── images/   # .jpg 图像
  │   └── labels/   # YOLO .txt：每行 “class x_center y_center w h”
  ├── val/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
```

`YOLODetectionDataset` 会读取 `images/*.jpg` 与同名的 `labels/*.txt`，按 YOLO 格式裁剪出每个检测框并返回 `(crop_tensor, class_id) ​data`。

####  2.推理输入
将待分类的原图放入 `user_data/image/`，对应的检测框 JSON文件 放入 `user_data/json/`，格式示例：
```
[
  { "bbox": [x1, y1, x2, y2] },
  { "bbox": [x1', y1', x2', y2'] }
]
```

### 4.快速开始
#### 1.模型训练
`bash`
```
python Resnet_101.py
```
- 根据`./Acne`目录进行数据读取以及训练。
  
- 会在 `checkpoints/` 下保存每轮最佳模型。

- 最终导出 `final_model.pth`。

- TensorBoard 日志写入 `runs/`。

#### 2.图片分类

`bash`

```
python classify.py
```


- 调用`final_model.pth`进行分类。

- 会在 `output/` 目录下生成每张图对应的子文件夹。

- `output/original/` 存放输入的原图和 JSON。

- `output/annotated/` 存放带分类标签的图和结果 JSON。

#### 3.生成护肤建议

如果您使用的是Linux/Mac,则在终端输入

```
export DEEPSEEK_API_KEY="您的deepseek官方api密钥"
```
若终端日志出现
```
RuntimeError: 请先通过 `export AIMLAPI_KEY="你的_API_Key"` 设置环境变量
```

则您可通过以下指令来确认您的deepseek官方api密钥是否成功被加载。


```
 echo $DEEPSEEK_API_KEY
```

在成功加载官方密钥后，即可通过以下指令生成对应建议。

```
python generate_advice.py --input_dir output
```


若成功生成建议，则应出现

```
Updated with advice: output/<image_stem>/annotated/<image_stem>.json
```


## 注意，使用生成护肤建议功能之前，请确保您的deepseek账号有充足余额来调用api。

### 5.目录文件说明
#### 1.data.py
- 自定义 `YOLODetectionDataset`，读取 YOLO 标注并裁剪检测框；返回 `(crop_tensor, class_id)`。

#### 2.Resnet_101.py

- 训练：`ResNet-101 + Dropout`，优化器使用 `AdamW`，学习率调度器为 `ReduceLROnPlateau`，`EarlyStopping`。

- 验证 & 测试：输出 `Loss/Accuracy`，并保存最佳模型。

#### 3.classify.py

- 加载 `final_model.pth`。

- 对用户提供的图像 & YOLO JSON 进行分类推理。

- 绘制边框 & 标签，输出到 `output/`。

#### 4.generate_advice.py

- 读取`output/<image_stem>/annotated/<image_stem>.json`文件,获取`pred_label`。

- `get_advice_for_label(label)`映射`pred_label`,发送`system/user prompt`，返回一段详细且易懂的护理方案，

- `process_files(input_dir)`匹配`input_dir/*/annotated/*.json`，缓存同一标签的建议，避免重复调用。

- 为每条记录添加 `"advice"` 并原地覆盖写回`output/<image_stem>/annotated/<image_stem>.json`文件。


























