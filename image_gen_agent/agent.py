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

# No default refs — all reference images come from user uploads in the session

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

EVALUATOR_SYSTEM = """你是一个极其严格的图像质量评估师。对比这张生成图与用户提供的角色参考图，评估角色一致性和动作合理性。

你必须非常严格地评分。以下任何一项不匹配都必须扣分：

【角色一致性】逐一检查图中每个角色（对照参考图）：
1. 颜色：主体色必须与参考图完全一致。颜色偏差直接扣2分。
2. 配饰/服装：参考图中有的配饰必须出现。缺任何一个扣1分。
3. 体型比例：必须与参考图一致（如参考图是Q版大头就必须是Q版大头）。比例不对扣3分。
4. 独特标识：参考图中的独特特征（花纹、斑纹、特殊颜色区域等）必须保留。缺少扣1分。
5. 角色数量：参考图中有几个角色，生成图就必须有几个。少一个扣2分。
6. 风格统一：所有角色风格必须统一。不统一扣2分。

【动作合理性】检查每个角色的姿态和动作：
7. 肢体合理：手脚位置、关节弯曲是否自然。出现断肢、多余肢体、穿模扣2分。
8. 动作协调：角色动作是否符合其体型特征。不协调扣1分。
9. 互动逻辑：角色之间的互动是否合理（目光、手势方向）。逻辑矛盾扣1分。
10. 场景融合：角色动作是否与场景匹配。不匹配扣1分。

如果没有提供参考图，则只评估画面质量、动作合理性和风格一致性。

评分标准（满分10分，从10开始扣）：
- 10分：完美匹配参考图且动作自然
- 8-9分：minor偏差
- 6-7分：有明显偏差
- 4-5分：多处不匹配
- 0-3分：严重偏离

输出JSON：
{"score": 0-10, "pass": true/false(>=8为pass), "issues": [{"character": "名", "issue": "问题", "type": "consistency/action"}], "suggestions": "修改建议"}

严格评分！只输出JSON。"""


def _get_token():
    r = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
    return r.stdout.strip()


def _build_session_ref_parts(session_refs):
    """Build reference image parts from user-uploaded images in session state."""
    parts = []
    for ref in session_refs:
        parts.append({"inline_data": {"mime_type": ref["mime_type"], "data": ref["data"]}})
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

FEATURE_EXTRACT_PROMPT = """请仔细观察这张角色参考图，详细提取所有视觉特征。要求极其精确和详细：

请输出以下信息：
1. 整体形状和比例（头身比、体型胖瘦、大小）
2. 主体颜色（精确描述每个部位的颜色，如"身体是橙红色，腹部是奶白色"）
3. 面部特征（眼睛颜色/大小/形状、嘴巴、鼻子、表情特点）
4. 独特标识（花纹、斑纹位置和形状、特殊颜色区域）
5. 配饰/服装（帽子、衣服、蝴蝶结等的颜色、样式、位置）
6. 四肢特征（手脚颜色、爪子形状）
7. 尾巴/翅膀等附属特征（颜色、形状、花纹）
8. 材质/风格感觉（如3D粘土质感、光滑塑料感等）

用简洁精确的中文描述，不要遗漏任何视觉细节。"""


async def add_reference_image(role: str, tool_context: ToolContext) -> dict:
    """Register a user-uploaded image as a reference. Automatically extracts detailed visual features using Flash.

    Args:
        role: What this image represents, e.g. "机器人角色参考" or "赛博朋克风格参考"
    """
    logger.info(f"📤 [AddRef] Adding reference: {role}")

    # Search recent session events for user-uploaded images
    img_data = None
    for event in reversed(tool_context.session.events):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                    img_data = {
                        "role": role,
                        "data": base64.b64encode(part.inline_data.data).decode(),
                        "mime_type": part.inline_data.mime_type or "image/png"
                    }
                    break
        if img_data:
            break

    if not img_data:
        return {"status": "error", "message": "未找到上传的图片。请在消息中附带图片后重试。"}

    # Use Flash to extract detailed visual features
    logger.info(f"  🔍 Extracting visual features with Flash...")
    token = _get_token()
    parts = [
        {"text": FEATURE_EXTRACT_PROMPT},
        {"inline_data": {"mime_type": img_data["mime_type"], "data": img_data["data"]}}
    ]

    url = (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
           f"/locations/{REGION}/publishers/google/models/gemini-3.5-flash:generateContent")

    resp = requests.post(url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"contents": [{"role": "USER", "parts": parts}],
              "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}})

    description = ""
    if resp.status_code == 200:
        for c in resp.json().get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                if "text" in p:
                    description += p["text"]
        logger.info(f"  ✅ Extracted features: {len(description)} chars")
    else:
        logger.warning(f"  ⚠️ Feature extraction failed: {resp.status_code}")

    img_data["description"] = description

    session_refs = tool_context.state.get("session_refs", [])
    session_refs.append(img_data)
    tool_context.state["session_refs"] = session_refs

    logger.info(f"  ✅ Added ref: {role} (total: {len(session_refs)})")
    return {"status": "success", "role": role, "features": description[:300], "total_user_refs": len(session_refs)}


async def optimize_prompt(scene_description: str, tool_context: ToolContext) -> dict:
    """Optimize a scene description into a structured image generation prompt using Flash 3.5.

    Args:
        scene_description: The user's scene description, e.g. "3D卡通风格，小动物们在长城上玩耍"
    """
    logger.info(f"🔧 [Stage 1] optimize_prompt: {scene_description[:100]}")
    t0 = time.time()
    token = _get_token()

    parts = [{"text": OPTIMIZER_SYSTEM}]

    # Use only user-uploaded session refs
    session_refs = tool_context.state.get("session_refs", [])
    parts.extend(_build_session_ref_parts(session_refs))

    if session_refs:
        image_desc = "\n".join(
            f"- IMG_{i} 是{r['role']}" + (f"\n  详细特征：{r['description']}" if r.get('description') else "")
            for i, r in enumerate(session_refs, 1)
        )
        parts.append({"text": f"以上{len(session_refs)}张参考图片：\n{image_desc}\n\n请生成提示词（角色描述必须严格基于上面的详细特征），要求：{scene_description}"})
    else:
        parts.append({"text": f"（无参考图片）\n\n请生成提示词，要求：{scene_description}"})

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
    session_refs = tool_context.state.get("session_refs", [])
    parts.extend(_build_session_ref_parts(session_refs))
    parts.append({"text": optimized_prompt})

    url = (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
           f"/locations/{REGION}/publishers/google/models/gemini-3.1-flash-image:generateContent")
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

    # Build ref parts for evaluation from session refs
    ref_parts = []
    session_refs = tool_context.state.get("session_refs", [])
    for ref in session_refs:
        ref_parts.append({"text": f"[{ref['role']}]:"})
        ref_parts.append({"inline_data": {"mime_type": ref["mime_type"], "data": ref["data"]}})

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

        # Track all scores
        all_scores = tool_context.state.get("all_scores", {})
        all_scores[str(idx)] = score
        tool_context.state["all_scores"] = all_scores

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

    # Find best scoring candidate overall
    all_scores = tool_context.state.get("all_scores", {})
    best_idx = max(all_scores, key=all_scores.get) if all_scores else None
    best_score = all_scores.get(best_idx, 0) if best_idx else 0

    # Save final selected images as final_1, final_2, ...
    final_parts = []
    if len(all_passed) >= num_target:
        for out_i, src_idx in enumerate(all_passed[:num_target], 1):
            art = await tool_context.load_artifact(filename=f"candidate_{src_idx}.png")
            if art:
                await tool_context.save_artifact(filename=f"final_{out_i}.png", artifact=art)
                final_parts.append(art)
        logger.info(f"  🎉 Got {num_target} passing images! Saved as final_1...{num_target}.png")

    still_needed = max(0, num_target - len(all_passed))
    logger.info(f"  📋 Total: {len(all_passed)} passed, {len(all_failed_list)} failed, need {still_needed} more (best={best_score}/10)")

    # If goal reached, output images directly in chat
    if final_parts:
        response_parts = [types.Part(text=f"✅ 已选出 {len(final_parts)} 张合格图片：")]
        response_parts.extend(final_parts)
        tool_context.actions.skip_summarization = True
        tool_context.actions.escalate = True

    return {
        "status": "success",
        "passed_count": len(all_passed),
        "passed_indices": all_passed,
        "failed_count": len(all_failed_list),
        "still_needed": still_needed,
        "goal_reached": len(all_passed) >= num_target,
        "best_candidate_index": int(best_idx) if best_idx else None,
        "best_score": best_score,
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
    session_refs = tool_context.state.get("session_refs", [])
    parts.extend(_build_session_ref_parts(session_refs))

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
    session_refs = tool_context.state.get("session_refs", [])
    parts.extend(_build_session_ref_parts(session_refs))
    parts.append({"text": "=== 需要修改的原图 ==="})
    parts.append({"inline_data": {"mime_type": "image/png", "data": original_b64}})
    parts.append({"text": edit_prompt})

    url = (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
           f"/locations/{REGION}/publishers/google/models/gemini-3.1-flash-image:generateContent")
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
    session_refs = tool_context.state.get("session_refs", [])
    for ref in session_refs:
        ref_parts.append({"text": f"[{ref['role']}]:"})
        ref_parts.append({"inline_data": {"mime_type": ref["mime_type"], "data": ref["data"]}})

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
        # Show edited image inline in chat
        tool_context.actions.skip_summarization = True
        tool_context.actions.escalate = True
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




# === Agent Definition ===
root_agent = Agent(
    model="gemini-3.5-flash",
    name="image_gen_agent",
    description="Generates images based on user-provided reference images and prompts, with automatic quality evaluation.",
    instruction="""你是一个通用的图像生成助手。根据用户提供的参考图和描述生成高质量图片。

=== 核心原则 ===
- 风格完全由用户的提示词决定（3D卡通、写实、2D插画等都可以）
- 角色/内容完全依赖用户提供的 reference image
- 你不预设任何风格限制，没有任何预置角色

=== 素材引导 ===
在开始生成前，评估用户素材是否充分：
1. 用户没有提供参考图也没有指定风格 → 询问"请描述期望的风格，或上传参考图"
2. 用户没有提供参考图但指定了风格 → 直接按文字描述生成，不需要参考图
3. 用户上传了参考图但没指定风格 → 根据参考图推断风格，向用户确认"从参考图看，风格是XX，确认用这个风格吗？"
4. 用户上传了图但没说明用途 → 询问"这张图是作为角色参考还是风格参考？"

=== 生成流程 ===
1. 用户上传参考图时 → add_reference_image 记录每张图的用途
2. optimize_prompt（基于用户上传的参考图和描述）
3. generate_images（数量 = 用户要求数 + 1）
4. evaluate_and_select（num_target = 用户要求数量）
5. 不通过则 refine_prompt → 再生成 → 再评估（最多重试5轮）
6. 如果5轮后仍没有合格图片（score>=8），选所有候选中分数最高的作为最终结果

⚠️ 严格规则：
- generate_images 数量 = 用户要求数量 + 1，不要多生成！
- 只使用用户上传的参考图，没有任何预置角色
- 最多重试5轮，5轮后选最高分的交付

=== 参考图管理 ===
- 用户附图 + 说明 → add_reference_image(role="描述")
- 支持任何类型：角色、风格、场景、构图参考等
- 用户可多次上传逐步补充

=== 编辑流程 ===
用户说"修改第X张图" → edit_image

=== 重要 ===
- 评估标准基于用户提供的参考图，没有参考图则只评估画面质量和动作合理性
- 灵活适应各种生图需求""",
    tools=[add_reference_image, optimize_prompt, generate_images, evaluate_and_select, refine_prompt, edit_image, PreloadMemoryTool()],
    after_agent_callback=save_memories_callback,
)
