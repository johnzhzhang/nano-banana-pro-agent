"""
Happy Element 图像生成 ADK Agent (v3)
- 评估循环：生成 → 逐张评估 → 不够则补生成，直到凑齐3张合格图
- Memory Bank 集成
- 详细日志
"""
import base64, json, logging, os, subprocess, time, requests
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
import google.genai.types as types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("image_gen_agent")

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "john-poc-453315")
REGION = os.environ.get("GOOGLE_CLOUD_REGION", "global")
REF_IMAGES_DIR = os.environ.get("REF_IMAGES_DIR", "/tmp/nano_images")

CHARACTER_REFS = [
    {"file": "a166e269a528cd48faa4848d9617600a249083.jpeg", "role": "浣熊角色参考"},
    {"file": "2dc690e2357fef10fb6260f6d73bcbad152534.jpeg", "role": "整体风格参考"},
    {"file": "c21f5ca4f07cec444ce5625085c27445132701.jpeg", "role": "小鸡角色参考"},
    {"file": "7c05e45a7abd8391f1fdf34fabdffea6235175.jpeg", "role": "狐狸角色参考"},
    {"file": "cc89ffc7d78ef27166464e799e59b590210591.jpeg", "role": "猫头鹰角色参考"},
    {"file": "7c2824a25aa962bb1eb001495bfecafc1481356.png", "role": "猫头鹰场景参考"},
    {"file": "cf1bbb51d381b19bfc397278854be4b41402876.png", "role": "青蛙角色参考"},
    {"file": "d7222fbde2bb40677be78d9febc26d201230050.png", "role": "青蛙全身参考"},
    {"file": "831ce2416fc684a43a2bd3d68e1784c8218923.jpeg", "role": "熊角色参考"},
    {"file": "8980026972ed3622b9ffdeb7cf50147c1327010.png", "role": "熊场景参考"},
]

OPTIMIZER_SYSTEM = """你是一个专业的AI图像生成提示词优化师。

用户会提供：
1. 若干参考图片（角色参考、风格参考、场景参考等）
2. 一个简单的生成需求描述

你的任务是生成一个结构化的、高质量的图像生成提示词，供 Gemini Pro Image 模型使用。

输出规则：
- 使用中文
- 必须包含：【画面风格】【场景】【角色】【构图】【图片对应】五个部分
- 【画面风格】必须在开头明确声明用户指定的渲染风格，并禁止偏离该风格
- 【场景】描述要简洁，场景风格必须与画面风格一致
- 【角色】部分必须为每个角色详细描述：颜色、比例、五官、服装/配饰、独特标识
- 对颜色相近或容易混淆的角色，必须明确标注区别
- 在【图片对应】中说明 [IMG_N] 对应什么
- 直接输出提示词，不要输出解释性文字
- 提示词中用 [IMG_N] 引用对应图片

关键约束（必须在提示词中体现）：
- 画面风格严格遵循用户指定的风格方向，不要擅自更改
- 所有角色风格必须统一，像出自同一款作品
- 角色外观必须严格参照参考图片
- 在【角色】描述中，必须逐一列出每个角色从参考图中观察到的所有视觉特征，不可遗漏或简化
- 必须在提示词中强调"角色外观与参考图完全一致，不可自行创作或修改任何细节"

【动态感约束】：
- 每个角色必须有不同的动态姿势，禁止排成一排站立
- 角色之间要有互动关系
- 角色在画面中错落分布，有前后层次关系
- 添加动态点缀（飘落的花瓣/树叶、飞舞的蝴蝶等）

【构图约束】：
- 视角：略低于平视（eye-level），禁止俯视/鸟瞰视角
- 比例：角色整体只占画面高度的30-40%，背景占60-70%
- 禁止：角色撑满画面、近景特写、俯视拍摄"""

CHAR_REF_PREFIX = """[IMPORTANT] The following reference images are the EXACT design standards for each character. When generating:
1. Each character's colors, patterns, accessories, clothing must EXACTLY match the corresponding reference image — no creative deviation
2. Body proportions, facial features, unique markings must be reproduced 1:1
3. These reference images are the character "design sheets" — the output must look like different action poses of the SAME characters

"""

EVALUATOR_SYSTEM = """你是一个极其严格的图像质量评估师。对比这张生成图与角色参考图，评估角色一致性。

你必须非常严格地评分。以下任何一项不匹配都必须扣分：

逐一检查图中每个角色：
1. 颜色：主体色必须完全一致（浣熊是棕橙色不是灰色，狐狸是红橙色等）。颜色偏差直接扣2分。
2. 配饰/服装：浣熊必须有绿色小礼帽+绿色背带裤+黄色蝴蝶结。缺任何一个扣1分。
3. 体型比例：必须是Q版大头小身（头身比约1:1）。如果变成写实比例直接扣3分。
4. 独特标识：浣熊必须有环形条纹尾巴，小鸡必须有红色鸡冠，狐狸尾巴尖必须是深色，猫头鹰必须是紫色，青蛙脸颊必须有黄色条纹斑。缺少任何一个扣1分。
5. 角色数量：必须有6个角色全部出现。少一个扣2分。
6. 风格统一：所有角色必须是同一3D卡通风格，不能有的写实有的卡通。不统一扣2分。

评分标准（满分10分，从10开始扣）：
- 10分：所有角色完美匹配参考图
- 8-9分：minor偏差（如某角色姿态略不同但颜色配饰都对）
- 6-7分：有明显偏差（颜色错误、缺少配饰、比例不对）
- 4-5分：多个角色不匹配
- 0-3分：严重偏离参考图

输出JSON：
{"score": 0-10, "pass": true/false(>=8为pass), "issues": [{"character": "名", "issue": "问题"}], "suggestions": "修改建议"}

你要像甲方审稿一样严格！只输出JSON。"""


def _get_token():
    r = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
    return r.stdout.strip()


def _load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _get_mime(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")


def _build_ref_parts():
    """Build reference image parts for API calls."""
    parts = []
    for img_info in CHARACTER_REFS:
        path = os.path.join(REF_IMAGES_DIR, img_info["file"])
        if os.path.exists(path):
            parts.append({"inline_data": {"mime_type": _get_mime(path), "data": _load_image_b64(path)}})
    return parts


async def save_memories_callback(callback_context: CallbackContext):
    """After each agent turn, save session events to Memory Bank."""
    logger.info("💾 Saving session events to Memory Bank...")
    try:
        await callback_context.add_events_to_memory(events=callback_context.session.events[-5:-1])
        logger.info("✅ Memory saved successfully")
    except Exception as e:
        logger.warning(f"⚠️ Memory Bank not available (skipping): {e}")
    return None


# === Tools ===

async def optimize_prompt(scene_description: str, tool_context: ToolContext) -> dict:
    """Optimize a scene description into a structured image generation prompt using Flash 3.5.

    Args:
        scene_description: The user's scene description, e.g. "3D卡通风格，小动物们在长城上玩耍"
    """
    logger.info(f"🔧 [Stage 1] optimize_prompt: {scene_description[:100]}")
    t0 = time.time()
    token = _get_token()

    parts = [{"text": OPTIMIZER_SYSTEM}]
    parts.extend(_build_ref_parts())
    image_desc = "\n".join(f"- IMG_{i} 是{img['role']}" for i, img in enumerate(CHARACTER_REFS, 1))
    parts.append({"text": f"以上{len(CHARACTER_REFS)}张图片：\n{image_desc}\n\n请生成提示词，要求：{scene_description}"})

    url = (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
           f"/locations/{REGION}/publishers/google/models/gemini-3.5-flash:generateContent")

    resp = requests.post(url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"contents": [{"role": "USER", "parts": parts}],
              "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192}})

    logger.info(f"  ⏱️ Flash: {resp.status_code} ({time.time()-t0:.1f}s)")
    if resp.status_code != 200:
        return {"status": "error", "message": f"Flash API failed: {resp.status_code}"}

    optimized = ""
    for c in resp.json().get("candidates", []):
        for p in c.get("content", {}).get("parts", []):
            if "text" in p:
                optimized += p["text"]

    tool_context.state["optimized_prompt"] = optimized
    tool_context.state["generation_round"] = 0
    tool_context.state["total_generated"] = 0
    logger.info(f"  ✅ Prompt: {len(optimized)} chars")
    return {"status": "success", "prompt_length": len(optimized), "prompt_preview": optimized[:300]}


async def generate_images(num_images: int, tool_context: ToolContext) -> dict:
    """Generate images using the optimized prompt. Saves as artifacts.

    Args:
        num_images: Number of images to generate (1-6)
    """
    optimized_prompt = tool_context.state.get("optimized_prompt")
    if not optimized_prompt:
        return {"status": "error", "message": "No optimized prompt. Run optimize_prompt first."}

    num_images = min(max(num_images, 1), 6)
    round_num = tool_context.state.get("generation_round", 0) + 1
    tool_context.state["generation_round"] = round_num
    total = tool_context.state.get("total_generated", 0)

    logger.info(f"🎨 [Generate] Round {round_num}: generating {num_images} images...")
    token = _get_token()

    parts = [{"text": CHAR_REF_PREFIX}]
    parts.extend(_build_ref_parts())
    parts.append({"text": optimized_prompt})

    url = (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
           f"/locations/{REGION}/publishers/google/models/gemini-3-pro-image:generateContent")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    request_body = {
        "contents": [{"role": "USER", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "temperature": 0.3, "topP": 0.5,
            "imageConfig": {"aspectRatio": "1:1"}
        }
    }

    generated = []
    for i in range(num_images):
        idx = total + i + 1
        t0 = time.time()
        logger.info(f"  🖼️ Image {idx}...")
        resp = requests.post(url, headers=headers, json=request_body)
        logger.info(f"  ⏱️ {resp.status_code} ({time.time()-t0:.1f}s)")
        if resp.status_code != 200:
            continue
        for c in resp.json().get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                if "inlineData" in p:
                    img_bytes = base64.b64decode(p["inlineData"]["data"])
                    artifact = types.Part(inline_data=types.Blob(data=img_bytes, mime_type="image/png"))
                    version = await tool_context.save_artifact(filename=f"candidate_{idx}.png", artifact=artifact)
                    logger.info(f"  ✅ candidate_{idx}.png ({len(img_bytes)} bytes, v{version})")
                    generated.append({"filename": f"candidate_{idx}.png", "index": idx})

    tool_context.state["total_generated"] = total + len(generated)
    return {"status": "success", "images_generated": len(generated), "round": round_num, "artifacts": generated}


async def evaluate_and_select(tool_context: ToolContext) -> dict:
    """Evaluate each candidate image against reference images.
    Returns which passed (score>=7) and which failed. Goal: collect 3 passing images.
    """
    logger.info("🔍 [Evaluate] Checking candidate images...")
    token = _get_token()
    total = tool_context.state.get("total_generated", 0)

    # Load all candidate images
    candidates = []
    for i in range(1, total + 1):
        fname = f"candidate_{i}.png"
        try:
            art = await tool_context.load_artifact(filename=fname)
            if art and art.inline_data and art.inline_data.data:
                candidates.append((i, base64.b64encode(art.inline_data.data).decode()))
        except Exception:
            pass

    if not candidates:
        return {"status": "error", "message": "No candidate images found."}

    # Skip already-evaluated ones
    already_passed = tool_context.state.get("passed_indices", [])
    already_failed = tool_context.state.get("failed_indices", [])
    to_evaluate = [(i, d) for i, d in candidates if i not in already_passed and i not in already_failed]

    if not to_evaluate:
        return {"status": "done", "passed_count": len(already_passed), "passed_indices": already_passed}

    logger.info(f"  📷 Evaluating {len(to_evaluate)} new candidates (already passed: {len(already_passed)})")

    # Build ref parts for evaluation
    ref_parts = []
    for img_info in CHARACTER_REFS:
        path = os.path.join(REF_IMAGES_DIR, img_info["file"])
        if os.path.exists(path):
            ref_parts.append({"text": f"[{img_info['role']}]:"})
            ref_parts.append({"inline_data": {"mime_type": _get_mime(path), "data": _load_image_b64(path)}})

    url = (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
           f"/locations/{REGION}/publishers/google/models/gemini-3.5-flash:generateContent")
    auth_headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    new_passed = []
    new_failed = []
    all_suggestions = []

    for idx, img_b64 in to_evaluate:
        t0 = time.time()
        parts = [{"text": EVALUATOR_SYSTEM}, {"text": "=== 角色参考图 ==="}]
        parts.extend(ref_parts)
        parts.append({"text": "=== 生成图 ==="})
        parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
        parts.append({"text": "评估这张图。输出JSON。"})

        resp = requests.post(url, headers=auth_headers,
            json={"contents": [{"role": "USER", "parts": parts}],
                  "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}})

        elapsed = time.time() - t0
        if resp.status_code != 200:
            logger.error(f"  ❌ Eval candidate_{idx} API error")
            new_failed.append(idx)
            continue

        eval_text = ""
        for c in resp.json().get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                if "text" in p:
                    eval_text += p["text"]

        try:
            clean = eval_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            ev = json.loads(clean)
        except json.JSONDecodeError:
            ev = {"score": 5, "pass": False, "suggestions": eval_text[:200]}

        score = ev.get("score", 0)
        passed = ev.get("pass", score >= 8)
        logger.info(f"  📊 candidate_{idx}: score={score}/10 {'✅' if passed else '❌'} ({elapsed:.1f}s)")

        if passed:
            new_passed.append(idx)
        else:
            new_failed.append(idx)
            if ev.get("suggestions"):
                all_suggestions.append(ev["suggestions"])

    # Update state
    all_passed = already_passed + new_passed
    all_failed_list = already_failed + new_failed
    tool_context.state["passed_indices"] = all_passed
    tool_context.state["failed_indices"] = all_failed_list
    tool_context.state["eval_suggestions"] = "\n".join(all_suggestions) if all_suggestions else ""

    # Save final selected images as final_1, final_2, final_3
    if len(all_passed) >= 3:
        for out_i, src_idx in enumerate(all_passed[:3], 1):
            art = await tool_context.load_artifact(filename=f"candidate_{src_idx}.png")
            if art:
                await tool_context.save_artifact(filename=f"final_{out_i}.png", artifact=art)
        logger.info(f"  🎉 Got 3 passing images! Saved as final_1/2/3.png")

    still_needed = max(0, 3 - len(all_passed))
    logger.info(f"  📋 Total: {len(all_passed)} passed, {len(all_failed_list)} failed, need {still_needed} more")

    return {
        "status": "success",
        "passed_count": len(all_passed),
        "passed_indices": all_passed,
        "failed_count": len(all_failed_list),
        "still_needed": still_needed,
        "goal_reached": len(all_passed) >= 3,
        "suggestions": tool_context.state.get("eval_suggestions", "")[:300] if still_needed > 0 else ""
    }


async def refine_prompt(tool_context: ToolContext) -> dict:
    """Refine the prompt based on evaluation feedback to fix character consistency issues.
    Call this before generating more images when evaluation found issues.
    """
    suggestions = tool_context.state.get("eval_suggestions", "")
    original_prompt = tool_context.state.get("optimized_prompt", "")
    if not suggestions:
        return {"status": "skip", "message": "No suggestions to refine."}

    logger.info("📝 [Refine] Refining prompt based on evaluation...")
    token = _get_token()

    parts = [{"text": f"""修改以下提示词来解决角色一致性问题。只修正问题部分，保持整体结构不变。

=== 原始提示词 ===
{original_prompt}

=== 评估发现的问题和修改建议 ===
{suggestions}

直接输出修改后的完整提示词。"""}]
    parts.extend(_build_ref_parts())

    url = (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
           f"/locations/{REGION}/publishers/google/models/gemini-3.5-flash:generateContent")

    resp = requests.post(url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"contents": [{"role": "USER", "parts": parts}],
              "generationConfig": {"temperature": 0.5, "maxOutputTokens": 8192}})

    if resp.status_code != 200:
        return {"status": "error", "message": f"Refine API failed: {resp.status_code}"}

    refined = ""
    for c in resp.json().get("candidates", []):
        for p in c.get("content", {}).get("parts", []):
            if "text" in p:
                refined += p["text"]

    tool_context.state["optimized_prompt"] = refined
    tool_context.state["eval_suggestions"] = ""  # Clear after refining
    logger.info(f"  ✅ Prompt refined: {len(refined)} chars")
    return {"status": "success", "prompt_length": len(refined)}


# === Agent Definition ===
root_agent = Agent(
    model="gemini-3.5-flash",
    name="image_gen_agent",
    description="Generates 3D cartoon images with automatic quality evaluation loop.",
    instruction="""你是 Happy Element 的图像生成助手。生成6个动物角色在中国文化地标的3D卡通图片，确保角色与参考图一致。

工作流程（严格按顺序）：
1. optimize_prompt — 优化用户的场景描述为结构化提示词
2. generate_images(4) — 先生成4张候选图
3. evaluate_and_select — 逐张评估，筛选出>=7分的合格图
4. 检查结果：
   - 如果 goal_reached=true（已有3张合格）→ 完成，告知用户
   - 如果 still_needed > 0（不够3张）→ refine_prompt 修正提示词 → generate_images(still_needed+1) → evaluate_and_select
5. 最多重复步骤4两次。如果最终仍不够3张，把已有的合格图交付并说明。

重要规则：
- 目标：凑齐3张评分>=8的图片（严格标准）
- 合格图片最终保存为 final_1.png, final_2.png, final_3.png
- 每次评估后要告知用户当前进度（X/3 张合格，分数情况）
- 如果有不合格的，简要说明问题所在""",
    tools=[optimize_prompt, generate_images, evaluate_and_select, refine_prompt, PreloadMemoryTool()],
    after_agent_callback=save_memories_callback,
)
