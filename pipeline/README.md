# Gemini 图像生成流水线

两阶段 Pipeline：Gemini 3.5 Flash 优化提示词 → Gemini 3 Pro Image 生成图像。

## 架构

```
用户需求 + 参考图
       ↓
┌─────────────────────┐
│ Stage 1: Flash 3.5  │  分析参考图 → 输出结构化提示词
│ (提示词优化师)        │  temperature: 0.7
└─────────┬───────────┘
          ↓ 优化后的提示词
┌─────────────────────┐
│ Stage 2: Pro Image  │  参考图 + 提示词 → 生成图像
│ (图像生成)           │  temperature: 0.3, top_p: 0.5
└─────────────────────┘
```

## 环境要求

- Python 3.8+
- `gcloud` CLI 已安装并认证（用于获取 access token）
- 依赖：`pip install requests pyyaml`
- GCP 项目需开通 Vertex AI API（Gemini 模型访问权限）

## 使用方法

```bash
# 完整流水线（Flash优化 + Pro Image生成）
python3 generate_pipeline.py --config config.yaml

# 仅运行 Flash 优化，输出提示词（不生成图片）
python3 generate_pipeline.py --config config.yaml --optimize-only

# 跳过 Flash，使用 config 中的 manual_prompt 直接生成
python3 generate_pipeline.py --config config.yaml --skip-optimize
```

## 配置文件格式（config.yaml）

```yaml
# GCP 项目
project: your-project-id

# Flash 温度（控制提示词创意度）
flash_temperature: 0.7

# Pro Image 生成配置
generation_config:
  model: gemini-3-pro-image
  temperature: 0.3      # 越低越稳定
  top_p: 0.5
  aspect_ratio: "1:1"   # 支持 1:1, 16:9, 9:16, 4:3, 3:4

# 参考图片列表（相对于 config 文件所在目录）
images:
- file: raccoon_ref.jpeg
  role: 浣熊角色参考
- file: style_ref.jpeg
  role: 整体风格参考
- file: chick_ref.jpeg
  role: 小鸡角色参考
# ... 更多角色/风格/场景参考

# 生成张数
num_images: 3

# 输出目录和文件名前缀
output_dir: ./output
output_prefix: my_scene

# 用户生成需求（传给 Flash 的原始请求）
user_request: |
  以下是原始提示词，请在此基础上优化并生成最终图像生成提示词：
  === TASK ===
  Generate image
  === GLOBAL STYLE ===
  Style: Q版3D角色风格...
  === SUBJECTS ===
  ...
  === USER INPUT ===
  3D卡通风格，水平视角，小动物们在山西双林寺里玩耍@raccoon @chick ...

# （可选）跳过 Flash 时使用的手动提示词
# manual_prompt: "直接写给 Pro Image 的提示词..."
```

## 关键参数调优说明

| 参数 | 当前值 | 说明 |
|------|--------|------|
| flash_temperature | 0.7 | 控制 Flash 生成提示词的创意度。0.3=保守稳定，0.7=适度创意，1.0=很发散 |
| generation_config.temperature | 0.3 | Pro Image 的随机性。越低角色一致性越好，但构图可能重复 |
| generation_config.top_p | 0.5 | 与 temperature 配合控制生成多样性 |
| num_images | 3 | 每次生成的图片数量，建议3-5张供挑选 |

## Pipeline 设计要点

### Stage 1: Flash 提示词优化师

Flash 的 system prompt 包含以下约束：
- **画面风格**：严格遵循用户指定风格（3D卡通/2D扁平等），禁止偏离
- **场景**：简洁描述，风格与画面一致
- **角色**：逐一详细描述颜色、比例、五官、服装配饰、独特标识
- **动态感**：要求不同姿态、角色互动、错落布局、动态点缀
- **构图**：平视视角，角色30-40%画面，背景60-70%

### Stage 2: Pro Image 生成

发送给 Pro Image 的内容顺序：
1. **英文角色一致性指令**（char_ref_prefix）— 强调参考图是设计稿标准
2. **所有参考图片**（角色+风格+场景）
3. **Flash 优化后的中文提示词**

### 为什么 Flash 用中文输出

测试发现：当 Flash 输出英文提示词时，Pro Image 更依赖文字描述（英文理解好），反而降低了对参考图的依赖度，导致角色一致性下降。中文提示词让 Pro Image 更多"看图"，角色还原更好。

### char_ref_prefix 用英文的原因

这段指令是直接告诉 Pro Image "必须严格按参考图生成"，用英文确保模型理解这条元指令。

## 输出文件

运行后在 `output_dir` 下生成：
- `optimized_prompt.txt` — Flash 生成的优化提示词（方便调试）
- `{output_prefix}_1.png`, `{output_prefix}_2.png`, ... — 生成的图片

## 常见问题

**Q: 角色颜色/细节偏离参考图怎么办？**
降低 Pro Image temperature（如 0.2 或 0.1），或增加生成张数挑选最佳。

**Q: 背景太写实了怎么办？**
检查 Flash 生成的提示词（optimized_prompt.txt），如果场景描述用了写实词汇，可在 user_request 中强调"场景也必须是卡通风格"。

**Q: Flash 输出被截断？**
maxOutputTokens 已设为 8192，如仍不够可在代码中调大。

**Q: 想换场景怎么办？**
只需修改 config.yaml 中 user_request 里的 USER INPUT 部分（如改为"在长城上玩耍"），角色参考图不变。
