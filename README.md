# Nano Banana Pro Agent

基于 Google ADK 的图像生成 Agent，通过 Gemini 3.5 Flash + Gemini 3 Pro Image 两阶段 Pipeline 生成角色一致的 3D 卡通图片。

## 架构

```
用户输入 → [ADK Orchestrator: gemini-3.5-flash]
                    ↓
        optimize_prompt (gemini-3.5-flash)     提示词优化
                    ↓
        generate_images (gemini-3-pro-image)   图片生成
                    ↓
        evaluate_and_select (gemini-3.5-flash) 逐张评估 (>=8分通过)
                    ↓
            不够3张? → refine_prompt → generate_images → evaluate (最多2轮)
                    ↓
        edit_image (用户反馈修改单张图)
```

## 文件结构

```
image_gen_agent/
├── agent.py         # ADK Agent 主文件（tools + orchestration）
└── __init__.py
generate_pipeline.py # 独立脚本版 pipeline（不依赖 ADK）
deploy_agent.py      # 部署到 Gemini Enterprise Agent Platform
test_agent.py        # 本地测试
config_sample.yaml   # 配置示例
```

## 快速开始

```bash
# 安装依赖
pip install google-adk google-cloud-aiplatform[agent_engines,adk] requests pyyaml

# 本地测试（需要 gcloud auth）
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=global
export REF_IMAGES_DIR=/path/to/reference/images

# ADK Web UI
adk web --port 8080

# 或运行测试脚本
python test_agent.py
```

## 关键参数

| 参数 | 值 | 说明 |
|------|------|------|
| Flash temperature | 0.7 | 提示词优化创意度 |
| Pro Image temperature | 0.3 | 生图稳定性 |
| Pro Image top_p | 0.5 | 生图多样性 |
| 评估阈值 | >=8/10 | 严格角色一致性标准 |

## Agent Tools

| Tool | 功能 |
|------|------|
| `optimize_prompt` | Flash 3.5 生成结构化提示词 |
| `generate_images` | Pro Image 生成候选图 |
| `evaluate_and_select` | Flash 3.5 逐张评估角色一致性 |
| `refine_prompt` | 根据评估反馈修正提示词 |
| `edit_image` | 对单张图进行针对性修改 |
