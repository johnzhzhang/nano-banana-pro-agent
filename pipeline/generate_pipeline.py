"""
Gemini 图像生成流水线 (v12)
===========================
两阶段 Pipeline：
  Stage 1: Gemini 3.5 Flash 分析参考图 → 生成结构化中文提示词
  Stage 2: Gemini 3 Pro Image 根据参考图 + 提示词 → 生成图像

用法:
  python3 generate_pipeline.py --config config.yaml
  python3 generate_pipeline.py --config config.yaml --optimize-only
  python3 generate_pipeline.py --config config.yaml --skip-optimize
"""
import argparse, base64, json, os, subprocess, sys, requests, yaml


def get_token():
    """通过 gcloud CLI 获取当前认证的 access token"""
    r = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
    return r.stdout.strip()


def load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_mime(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp"}.get(ext, "image/png")


# ═══════════════════════════════════════════════════════════════════
# Stage 1: Flash 提示词优化师 System Prompt
# ═══════════════════════════════════════════════════════════════════
OPTIMIZER_SYSTEM = """你是一个专业的AI图像生成提示词优化师。

用户会提供：
1. 若干参考图片（角色参考、风格参考、场景参考等）
2. 一个简单的生成需求描述

你的任务是生成一个结构化的、高质量的图像生成提示词，供 Gemini Pro Image 模型使用。

输出规则：
- 使用中文
- 必须包含：【画面风格】【场景】【角色】【构图】【图片对应】五个部分
- 【画面风格】必须在开头明确声明用户指定的渲染风格，并禁止偏离该风格（如用户要求3D卡通则禁止写实，如用户要求2D扁平则禁止3D立体感）
- 【场景】描述要简洁，场景风格必须与画面风格一致
- 【角色】部分必须为每个角色详细描述：颜色、比例、五官、服装/配饰、独特标识
- 对颜色相近或容易混淆的角色，必须明确标注区别
- 在【图片对应】中说明 [IMG_N] 对应什么
- 直接输出提示词，不要输出解释性文字
- 提示词中用 [IMG_N] 引用对应图片

关键约束（必须在提示词中体现）：
- 画面风格严格遵循用户指定的风格方向（3D卡通/2D扁平/像素风等），不要擅自更改
- 所有角色风格必须统一，像出自同一款作品
- 角色外观必须严格参照参考图片，包括：精确的颜色（不可偏色）、身体花纹/斑纹的位置和形状、配饰和服装的每个细节（帽子颜色+装饰、衣服款式+颜色、蝴蝶结等）、体型和头身比例
- 在【角色】描述中，必须逐一列出每个角色从参考图中观察到的所有视觉特征，不可遗漏或简化
- 必须在提示词中强调"角色外观与参考图完全一致，不可自行创作或修改任何细节"

【动态感约束】（非常重要，让画面生动有趣，避免死板）：
- 角色姿态：每个角色必须有不同的动态姿势（跳跃、奔跑、挥手、转身、蹲下、飞起等），禁止所有角色排成一排站立不动
- 角色互动：角色之间要有互动关系（如追逐、手拉手、互相看对方、一起指向某处）
- 布局：角色在画面中错落分布，有前后层次关系（有的在前景大一些，有的在中景小一些），禁止整齐排列
- 场景生动元素：添加动态点缀（飘落的花瓣/树叶、飞舞的蝴蝶、飘动的旗帜、地上的小水坑倒影等）
- 整体氛围：像一张抓拍的欢乐瞬间，角色在玩耍中的自然状态，不是摆拍合影

【构图约束】（非常重要，必须在提示词的【构图】部分明确写出）：
- 视角：略低于平视（eye-level），从角色正面或微微仰视拍摄，禁止俯视/鸟瞰视角
- 比例：角色整体只占画面高度的30-40%，留出大量空间展示背景（建筑、天空、植被）
- 背景：必须完整展示场景环境，包含天空、建筑屋顶全貌、远景元素
- 构图参考：像手游宣传海报，角色在画面下半部分，背景占画面60-70%
- 禁止：角色撑满画面、近景特写、俯视拍摄"""


# ═══════════════════════════════════════════════════════════════════
# Stage 2: 发给 Pro Image 的角色一致性英文指令
# ═══════════════════════════════════════════════════════════════════
CHAR_REF_PREFIX = """[IMPORTANT] The following reference images are the EXACT design standards for each character. When generating:
1. Each character's colors, patterns, accessories, clothing must EXACTLY match the corresponding reference image — no creative deviation
2. Body proportions, facial features, unique markings must be reproduced 1:1
3. These reference images are the character "design sheets" — the output must look like different action poses of the SAME characters

"""


def stage1_optimize(config, token):
    """Stage 1: Flash 分析参考图并生成结构化中文提示词"""
    print("=" * 50)
    print("Stage 1: Gemini 3.5 Flash 提示词优化")
    print("=" * 50)

    project = config["project"]
    base_dir = config.get("base_dir", ".")

    # 构建 parts: system prompt + 参考图 + 用户需求
    parts = [{"text": OPTIMIZER_SYSTEM}]

    for img_info in config["images"]:
        path = os.path.join(base_dir, img_info["file"])
        parts.append({"inline_data": {"mime_type": get_mime(path), "data": load_image_b64(path)}})

    image_desc = "\n".join(
        f"- IMG_{i} 是{img['role']}" for i, img in enumerate(config["images"], 1)
    )
    parts.append({"text": f"以上{len(config['images'])}张图片：\n{image_desc}\n\n请生成提示词，要求：{config['user_request']}"})

    # 调用 Flash API
    url = (f"https://aiplatform.googleapis.com/v1/projects/{project}"
           f"/locations/global/publishers/google/models/gemini-3.5-flash:generateContent")

    resp = requests.post(url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "contents": [{"role": "USER", "parts": parts}],
            "generationConfig": {"temperature": config.get("flash_temperature", 0.7), "maxOutputTokens": 8192}
        })

    if resp.status_code != 200:
        print(f"❌ Flash 调用失败: {resp.status_code}\n{resp.text[:300]}")
        sys.exit(1)

    optimized = ""
    for c in resp.json().get("candidates", []):
        for p in c.get("content", {}).get("parts", []):
            if "text" in p:
                optimized += p["text"]

    print(f"✅ 优化提示词生成完成 ({len(optimized)} 字符)")
    print("-" * 50)
    print(optimized[:1500] + ("..." if len(optimized) > 1500 else ""))
    print("-" * 50)
    return optimized


def stage2_generate(config, optimized_prompt, token):
    """Stage 2: 参考图 + 优化提示词 → Pro Image 生成"""
    print("\n" + "=" * 50)
    print("Stage 2: Gemini 3 Pro Image 生成")
    print("=" * 50)

    project = config["project"]
    base_dir = config.get("base_dir", ".")
    num = config.get("num_images", 3)
    gen_config = config.get("generation_config", {})

    # 构建 parts 顺序: 英文一致性指令 → 参考图 → 中文提示词
    parts = [{"text": CHAR_REF_PREFIX}]
    for img_info in config["images"]:
        path = os.path.join(base_dir, img_info["file"])
        parts.append({"inline_data": {"mime_type": get_mime(path), "data": load_image_b64(path)}})
    parts.append({"text": optimized_prompt})

    request_body = {
        "contents": [{"role": "USER", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "temperature": gen_config.get("temperature", 0.3),
            "topP": gen_config.get("top_p", 0.5),
            "imageConfig": {"aspectRatio": gen_config.get("aspect_ratio", "1:1")}
        }
    }

    model = gen_config.get("model", "gemini-3-pro-image")
    url = (f"https://aiplatform.googleapis.com/v1/projects/{project}"
           f"/locations/global/publishers/google/models/{model}:generateContent")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    output_dir = config.get("output_dir", base_dir)
    os.makedirs(output_dir, exist_ok=True)
    prefix = config.get("output_prefix", "output")

    for i in range(1, num + 1):
        print(f"\n生成第 {i}/{num} 张...")
        resp = requests.post(url, headers=headers, json=request_body)
        if resp.status_code != 200:
            print(f"  ❌ HTTP {resp.status_code}: {resp.text[:200]}")
            continue
        for c in resp.json().get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                if "inlineData" in p:
                    img = base64.b64decode(p["inlineData"]["data"])
                    out_path = os.path.join(output_dir, f"{prefix}_{i}.png")
                    with open(out_path, "wb") as f:
                        f.write(img)
                    print(f"  ✅ {out_path} ({len(img)} bytes)")

    print("\n🎉 生成完成!")


def main():
    parser = argparse.ArgumentParser(description="Gemini 图像生成流水线: Flash优化 + Pro Image生成")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--skip-optimize", action="store_true", help="跳过Flash优化，使用config中的manual_prompt")
    parser.add_argument("--optimize-only", action="store_true", help="仅运行Flash优化，输出提示词不生成图片")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # base_dir 默认为 config 文件所在目录
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if "base_dir" not in config:
        config["base_dir"] = config_dir
    elif not os.path.isabs(config["base_dir"]):
        config["base_dir"] = os.path.join(config_dir, config["base_dir"])

    token = get_token()

    if args.skip_optimize:
        prompt = config["manual_prompt"]
        stage2_generate(config, prompt, token)
    else:
        optimized = stage1_optimize(config, token)
        # 保存优化后的 prompt 方便调试
        save_path = os.path.join(config.get("output_dir", config["base_dir"]), "optimized_prompt.txt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(optimized)
        print(f"\n💾 提示词已保存: {save_path}")

        if not args.optimize_only:
            stage2_generate(config, optimized, token)


if __name__ == "__main__":
    main()
