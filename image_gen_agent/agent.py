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
1. 若干参考图片（角色参考、风格参考、场景参考等）——也可能没有
2. 一个生成需求描述

你的任务是生成一个结构化的、高质量的图像生成提示词，供 Gemini Pro Image 模型使用。

输出规则：
- 使用中文
- 必须包含：【画面风格】【场景】【角色】【构图】【图片对应】（如有参考图）
- 【画面风格】严格遵循用户指定的风格，如用户未指定则根据参考图推断
- 【场景】描述要简洁具体
- 【角色】如有参考图，必须为每个角色详细描述视觉特征；如无参考图则根据用户文字描述
- 直接输出提示词，不要输出解释性文字
- 提示词中用 [IMG_N] 引用对应图片

关键约束：
- 风格方向由用户决定，不要擅自更改
- 如有角色参考图，外观必须严格参照
- 所有角色风格必须统一

【动态感约束】：
- 角色必须有动态姿势，禁止呆板站立
- 角色之间有互动关系
- 画面有前后层次感
- 添加动态点缀元素

【构图约束】：
- 视角：略低于平视，禁止俯视
- 比例：角色占30-40%，背景占60-70%
- 禁止：角色撑满画面、近景特写"""

CHAR_REF_PREFIX = """[IMPORTANT] The following reference images are the EXACT design standards for each character. When generating:
1. Each character's colors, patterns, accessories, clothing must EXACTLY match the corresponding reference image — no creative deviation
2. Body proportions, facial features, unique markings must be reproduced 1:1
3. These reference images are the character "design sheets" — the output must look like different action poses of the SAME characters

"""

EVALUATOR_SYSTEM = """你是一个极其严格的图像质量评估师。对比这张生成图与角色参考图，评估角色一致性和动作合理性。

你必须非常严格地评分。以下任何一项不匹配都必须扣分：

【角色一致性】逐一检查图中每个角色：
1. 颜色：主体色必须完全一致（浣熊是棕橙色不是灰色，狐狸是红橙色等）。颜色偏差直接扣2分。
2. 配饰/服装：浣熊必须有绿色小礼帽+绿色背带裤+黄色蝴蝶结。缺任何一个扣1分。
3. 体型比例：必须是Q版大头小身（头身比约1:1）。如果变成写实比例直接扣3分。
4. 独特标识：浣熊必须有环形条纹尾巴，小鸡必须有红色鸡冠，狐狸尾巴尖必须是深色，猫头鹰必须是紫色，青蛙脸颊必须有黄色条纹斑。缺少任何一个扣1分。
5. 角色数量：必须有6个角色全部出现。少一个扣2分。
6. 风格统一：所有角色必须是同一3D卡通风格，不能有的写实有的卡通。不统一扣2分。

【动作合理性】检查每个角色的姿态和动作：
7. 肢体合理：手脚位置、关节弯曲是否自然合理。出现断肢、多余肢体、肢体穿模扣2分。
8. 动作协调：角色动作是否符合其体型特征（如青蛙蹲跳、猫头鹰飞翔、小鸡用翅膀而非手）。不协调扣1分。
9. 互动逻辑：角色之间的互动是否合理（如目光方向、手势指向是否一致）。逻辑矛盾扣1分。
10. 场景融合：角色的动作是否与场景匹配（如在寺庙里做合理的游玩动作，而非不相关的动作）。不匹配扣1分。

评分标准（满分10分，从10开始扣）：
- 10分：所有角色完美匹配且动作自然合理
- 8-9分：minor偏差（如某角色姿态略僵硬但整体合理）
- 6-7分：有明显偏差（颜色错误、肢体异常）
- 4-5分：多个角色不匹配或动作严重不合理
- 0-3分：严重偏离参考图

输出JSON：
{"score": 0-10, "pass": true/false(>=8为pass), "issues": [{"character": "名", "issue": "问题", "type": "consistency/action"}], "suggestions": "修改建议"}

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
        num_images: Number of images to generate (default 1, max 6)
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


async def evaluate_and_select(num_target: int, tool_context: ToolContext) -> dict:
    """Evaluate each candidate image against reference images.
    Returns which passed (score>=8) and which failed.

    Args:
        num_target: How many passing images are needed (e.g. 1 for single image, 3 for batch)
    """
    num_target = max(num_target, 1)
    logger.info(f"🔍 [Evaluate] Target: {num_target} passing images")
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

    # Save final selected images as final_1, final_2, ...
    if len(all_passed) >= num_target:
        for out_i, src_idx in enumerate(all_passed[:num_target], 1):
            art = await tool_context.load_artifact(filename=f"candidate_{src_idx}.png")
            if art:
                await tool_context.save_artifact(filename=f"final_{out_i}.png", artifact=art)
        logger.info(f"  🎉 Got {num_target} passing images! Saved as final_1...{num_target}.png")

    still_needed = max(0, num_target - len(all_passed))
    logger.info(f"  📋 Total: {len(all_passed)} passed, {len(all_failed_list)} failed, need {still_needed} more")

    return {
        "status": "success",
        "passed_count": len(all_passed),
        "passed_indices": all_passed,
        "failed_count": len(all_failed_list),
        "still_needed": still_needed,
        "goal_reached": len(all_passed) >= num_target,
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


async def edit_image(image_name: str, edit_instruction: str, tool_context: ToolContext) -> dict:
    """Edit a specific final image based on user feedback. Generates 2 candidates, evaluates, and picks the best one.

    Args:
        image_name: Which image to edit, e.g. "final_1.png"
        edit_instruction: What to change, e.g. "熊的腮红要更明显，颜色要更粉"
    """
    logger.info(f"✏️ [Edit] {image_name}: {edit_instruction[:80]}")
    token = _get_token()

    # Load the original image
    try:
        original_art = await tool_context.load_artifact(filename=image_name)
        if not original_art or not original_art.inline_data:
            return {"status": "error", "message": f"Image {image_name} not found in artifacts."}
    except Exception as e:
        return {"status": "error", "message": f"Cannot load {image_name}: {e}"}

    original_b64 = base64.b64encode(original_art.inline_data.data).decode()

    # Build edit prompt: original image + reference images + edit instruction
    edit_prompt = f"""请基于以下原图进行修改。

修改要求：{edit_instruction}

重要：
- 只修改用户要求的部分，其他保持不变
- 角色外观必须严格参照角色参考图
- 保持整体3D卡通风格和构图不变
"""

    parts = [{"text": CHAR_REF_PREFIX}]
    parts.extend(_build_ref_parts())
    parts.append({"text": "=== 需要修改的原图 ==="})
    parts.append({"inline_data": {"mime_type": "image/png", "data": original_b64}})
    parts.append({"text": edit_prompt})

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

    # Generate 2 edit candidates
    edit_candidates = []
    for i in range(2):
        t0 = time.time()
        logger.info(f"  🖼️ Generating edit candidate {i+1}/2...")
        resp = requests.post(url, headers=headers, json=request_body)
        logger.info(f"  ⏱️ {resp.status_code} ({time.time()-t0:.1f}s)")
        if resp.status_code != 200:
            continue
        for c in resp.json().get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                if "inlineData" in p:
                    img_bytes = base64.b64decode(p["inlineData"]["data"])
                    edit_candidates.append(img_bytes)
                    # Save as edit candidate artifact
                    artifact = types.Part(inline_data=types.Blob(data=img_bytes, mime_type="image/png"))
                    await tool_context.save_artifact(filename=f"edit_candidate_{i+1}.png", artifact=artifact)
                    logger.info(f"  ✅ edit_candidate_{i+1}.png ({len(img_bytes)} bytes)")

    if not edit_candidates:
        return {"status": "error", "message": "Failed to generate edit candidates."}

    # Evaluate each candidate
    logger.info("  🔍 Evaluating edit candidates...")
    ref_parts = []
    for img_info in CHARACTER_REFS:
        path = os.path.join(REF_IMAGES_DIR, img_info["file"])
        if os.path.exists(path):
            ref_parts.append({"text": f"[{img_info['role']}]:"})
            ref_parts.append({"inline_data": {"mime_type": _get_mime(path), "data": _load_image_b64(path)}})

    eval_url = (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
                f"/locations/{REGION}/publishers/google/models/gemini-3.5-flash:generateContent")
    auth_headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    best_score = -1
    best_idx = -1
    results = []

    for i, img_bytes in enumerate(edit_candidates):
        img_b64 = base64.b64encode(img_bytes).decode()
        parts = [{"text": EVALUATOR_SYSTEM}, {"text": "=== 角色参考图 ==="}]
        parts.extend(ref_parts)
        parts.append({"text": "=== 生成图 ==="})
        parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
        parts.append({"text": "评估这张图。输出JSON。"})

        resp = requests.post(eval_url, headers=auth_headers,
            json={"contents": [{"role": "USER", "parts": parts}],
                  "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}})

        if resp.status_code != 200:
            results.append({"index": i+1, "score": 0})
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
            ev = {"score": 5, "pass": False}

        score = ev.get("score", 0)
        logger.info(f"  📊 edit_candidate_{i+1}: score={score}/10")
        results.append({"index": i+1, "score": score, "pass": score >= 8})

        if score > best_score:
            best_score = score
            best_idx = i

    # If best passes threshold, replace the original
    if best_score >= 8:
        best_artifact = types.Part(inline_data=types.Blob(data=edit_candidates[best_idx], mime_type="image/png"))
        await tool_context.save_artifact(filename=image_name, artifact=best_artifact)
        logger.info(f"  🎉 Edit passed! Replaced {image_name} with candidate {best_idx+1} (score={best_score})")
        return {
            "status": "success",
            "replaced": True,
            "image": image_name,
            "best_score": best_score,
            "candidates": results
        }
    else:
        logger.info(f"  ⚠️ Edit candidates didn't pass (best={best_score}). Keeping original.")
        return {
            "status": "success",
            "replaced": False,
            "image": image_name,
            "best_score": best_score,
            "candidates": results,
            "message": f"Edit candidates scored {best_score}/10, below threshold 8. Original kept. Try different instructions."
        }


async def upload_reference(image_role: str, tool_context: ToolContext) -> dict:
    """Upload a user-provided reference image to replace or add to the default character references.
    The user should attach an image in their message before calling this tool.

    Args:
        image_role: Description of what this image is, e.g. "新角色参考" or "替换狐狸角色参考"
    """
    logger.info(f"📤 [Upload] Reference image: {image_role}")

    # Check if user attached an image in the current session events
    # Look through recent events for inline image data
    session_events = tool_context.state.get("_user_images", [])

    # Store the role for this reference - actual image comes from user message
    custom_refs = tool_context.state.get("custom_refs", [])
    custom_refs.append({"role": image_role})
    tool_context.state["custom_refs"] = custom_refs

    logger.info(f"  ✅ Added custom reference: {image_role} (total custom refs: {len(custom_refs)})")
    return {
        "status": "success",
        "message": f"已添加自定义参考图 '{image_role}'。请在消息中直接附带图片，生成时会自动使用。",
        "total_custom_refs": len(custom_refs)
    }


# === Agent Definition ===
root_agent = Agent(
    model="gemini-3.5-flash",
    name="image_gen_agent",
    description="Generates images based on user-provided reference images and prompts, with automatic quality evaluation.",
    instruction="""你是一个通用的图像生成助手。根据用户提供的参考图和描述生成高质量图片。

=== 核心原则 ===
- 风格完全由用户的提示词决定（3D卡通、写实、2D插画等都可以）
- 角色/内容完全依赖用户提供的 reference image
- 你不预设任何风格限制，用户说什么风格就什么风格

=== 素材引导 ===
在开始生成前，先评估用户提供的素材是否充分：
1. 如果用户没有提供任何参考图 → 询问是否需要提供角色/风格参考图
2. 如果用户只提供了部分角色参考 → 提醒"目前有X个角色参考，是否还需要补充其他角色？"
3. 如果用户没有指定风格 → 询问期望的风格方向（3D卡通/写实/扁平等）
4. 如果用户明确说"就这些，开始生成" → 直接开始，不再追问

风格补全规则：
- 如果用户只说了场景没说风格 → 建议补充风格描述
- 如果用户说了风格但不够具体（如只说"卡通"）→ 可以追问"Q版大头还是正常比例？色彩明亮还是柔和？"
- 如果用户给了完整描述 → 直接执行，不要多问

=== 生成流程 ===
1. 确认素材充分后 → optimize_prompt（基于用户提供的参考图和描述）
2. generate_images（数量 = 用户要求的最终图片数 + 1，如用户要1张则生成2张用于筛选）
3. evaluate_and_select（num_target = 用户要求的最终数量）
4. 不通过则 refine_prompt → 再生成(still_needed+1) → 再评估（最多2轮）

⚠️ 严格规则：generate_images 的数量 = 用户要求数量 + 1，不要多生成！用户要1张就生成2张，要3张就生成4张。

=== 编辑流程 ===
用户说"修改第X张图" → edit_image

=== 自定义参考图 ===
用户随时可以在对话中附带图片作为参考素材，调用 upload_reference 记录

=== 重要 ===
- 不要预设"必须是6个动物角色"或"必须在中国地标"——这些由用户决定
- 评估标准基于用户提供的参考图，没有参考图则只评估画面质量和动作合理性
- 灵活适应各种生图需求""",
    tools=[optimize_prompt, generate_images, evaluate_and_select, refine_prompt, edit_image, upload_reference, PreloadMemoryTool()],
    after_agent_callback=save_memories_callback,
)
