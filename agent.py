"""
Happy Element 图像生成 ADK Agent (v3)
- 评估循环：生成 → 逐张评估 → 不够则补生成，直到凑齐3张合格图
- Memory Bank 集成
- 详细日志
"""
import base64, json, os, subprocess, time, requests
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
import google.genai.types as types
from google import genai
from google.adk.models.google_llm import Gemini
from pydantic import Field

class GeminiWithLocation(Gemini):
    location: str = Field(default="global")

    @property
    def api_client(self) -> genai.Client:
        key = "_cached_client"
        if not hasattr(self, key) or getattr(self, key) is None:
            client = genai.Client(
                location=self.location,
                http_options=genai.types.HttpOptions(
                    headers=self._tracking_headers(),
                    retry_options=self.retry_options,
                    base_url=self.base_url,
                ),
            )
            object.__setattr__(self, key, client)
        return getattr(self, key)

    def __getstate__(self):
        state = super().__getstate__() if hasattr(super(), "__getstate__") else self.__dict__.copy()
        if isinstance(state, dict):
            state.pop("_cached_client", None)
        return state

    def __setstate__(self, state):
        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)
        else:
            self.__dict__.update(state)





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

EVALUATOR_SYSTEM = """你是一个极其严格的图像质量评估师。评估这张生成图的整体质量、角色一致性和动作合理性。

你必须非常严格地评分。以下任何一项不匹配都必须扣分：

【角色一致性】（如有参考图则逐一对照）：
1. 颜色：主体色必须与参考图完全一致。颜色偏差扣2分。
2. 配饰/服装：参考图中有的配饰必须出现且样式正确。缺少或错误扣1分。
3. 体型比例：必须与参考图一致。比例不对扣3分。
4. 独特标识：参考图中的独特特征必须保留。缺少扣1分。
5. 角色数量：应有的角色必须全部出现。少一个扣2分。
6. 风格统一：整体风格必须一致。不统一扣2分。

【人体/角色解剖合理性】（非常重要！）：
7. 手部：手指数量必须正确（人类5根），握持姿势合理。多指/少指/畸形手扣3分。
8. 面部：五官位置正常，表情协调，不能出现扭曲变形。面部异常扣2分。
9. 身体结构：四肢长度比例正常，关节弯曲方向正确，无断肢/多余肢体/穿模。结构异常扣3分。
10. 对称性：左右对称部位（如双眼、双手）大小和位置应基本对称。严重不对称扣1分。

【服装和道具合理性】：
11. 服装物理：衣服、披风、头发的飘动方向应一致（同一风向）。方向矛盾扣1分。
12. 武器/道具：持握姿势正确，道具形态完整不变形。握持异常或变形扣2分。
13. 层次关系：前后遮挡关系正确（如手在衣服前面/后面）。遮挡错误扣1分。

【构图与画面质量】：
14. 构图美学：画面构图是否专业、角色位置是否合理。构图混乱扣1分。
15. 清晰度：画面是否清晰无模糊伪影。有明显伪影扣1分。

【提示词符合度】（对照提示词要求）：
16. 场景：生成图的场景是否符合提示词描述的场景要求。场景不符扣2分。
17. 动作/姿态：角色的动作是否符合提示词中描述的姿态要求。动作不符扣2分。
18. 服装/外观：角色穿着和外观是否符合提示词要求。不符扣1分。

如果没有提供参考图，跳过【角色一致性】部分，只评估其他维度。

评分标准（满分10分，从10开始扣）：
- 10分：完美，无任何问题
- 8-9分：极minor偏差（如发丝细节略有不同）
- 6-7分：有明显但可接受的问题
- 4-5分：有严重的解剖/结构问题
- 0-3分：多处严重错误

输出JSON：
{"score": 0-10, "pass": true/false(>=8为pass), "issues": [{"character": "名/部位", "issue": "具体问题", "type": "consistency/anatomy/physics/composition"}], "suggestions": "修改建议"}

特别注意：手指数量、肢体结构、武器握持是AI生图最常见的错误，务必仔细检查！只输出JSON。"""


def _get_token():
    """Get access token via ADC (works in Agent Engine and locally with gcloud auth)."""
    try:
        import google.auth
        import google.auth.transport.requests
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token
    except Exception:
        # Fallback to gcloud CLI for local dev
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
    print("💾 Saving session events to Memory Bank...")
    try:
        await callback_context.add_events_to_memory(events=callback_context.session.events[-5:-1])
        print("✅ Memory saved successfully")
    except Exception as e:
        print(f"⚠️ Memory Bank not available (skipping): {e}")
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


async def add_reference_image(role: str, image_url: str, tool_context: ToolContext) -> dict:
    """Register an image as a reference. Provide image_url or leave empty to find from upload.

    Args:
        role: What this image represents, e.g. "角色参考" or "风格参考"
        image_url: Image URL (http/https/gs://). Pass empty string if image was uploaded in message.
    """
    print(f"📤 [AddRef] Adding reference: {role}")

    # Try URL first
    img_data = None
    if image_url and image_url.strip():
        try:
            url_str = image_url.strip()
            if url_str.startswith("gs://"):
                from google.cloud import storage as gcs_storage
                client = gcs_storage.Client()
                bucket_name = url_str.split("/")[2]
                blob_name = "/".join(url_str.split("/")[3:])
                file_bytes = client.bucket(bucket_name).blob(blob_name).download_as_bytes()
            else:
                file_bytes = requests.get(url_str, timeout=30).content
            mime = "image/jpeg" if any(x in url_str for x in [".jpg", ".jpeg"]) else "image/png"
            img_data = {"role": role, "data": base64.b64encode(file_bytes).decode(), "mime_type": mime}
        except Exception:
            pass

    # Search recent session events for user-uploaded images
    if not img_data:
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
                    # Handle file_data (GE/Agent Engine - GCS URI)
                    if hasattr(part, 'file_data') and part.file_data and part.file_data.file_uri:
                        import google.auth
                        import google.auth.transport.requests
                        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
                        creds.refresh(google.auth.transport.requests.Request())
                        uri = part.file_data.file_uri
                        if uri.startswith("gs://"):
                            from google.cloud import storage
                            bucket_name = uri.split("/")[2]
                            blob_name = "/".join(uri.split("/")[3:])
                            client = storage.Client()
                            file_bytes = client.bucket(bucket_name).blob(blob_name).download_as_bytes()
                        else:
                            file_bytes = requests.get(uri, headers={"Authorization": f"Bearer {creds.token}"}).content
                        img_data = {
                            "role": role,
                            "data": base64.b64encode(file_bytes).decode(),
                            "mime_type": part.file_data.mime_type or "image/png"
                        }
                        break
            if img_data:
                break

    if not img_data:
        captured = tool_context.state.get("captured_images", [])
        if captured:
            latest = captured[-1]
            img_data = {"role": role, "data": latest["data"], "mime_type": latest["mime_type"]}

    if not img_data:
        return {"status": "error", "message": "未找到上传的图片。请在消息中附带图片后重试。"}

    # Use Flash to extract detailed visual features
    print(f"  🔍 Extracting visual features with Flash...")
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
        print(f"  ✅ Extracted features: {len(description)} chars")
    else:
        print(f"  ⚠️ Feature extraction failed: {resp.status_code}")

    img_data["description"] = description

    session_refs = tool_context.state.get("session_refs", [])
    session_refs.append(img_data)
    tool_context.state["session_refs"] = session_refs

    print(f"  ✅ Added ref: {role} (total: {len(session_refs)})")
    return {"status": "success", "role": role, "features": description[:300], "total_user_refs": len(session_refs)}


async def optimize_prompt(scene_description: str, tool_context: ToolContext) -> dict:
    """Optimize a scene description into a structured image generation prompt using Flash 3.5.

    Args:
        scene_description: The user's scene description, e.g. "3D卡通风格，小动物们在长城上玩耍"
    """
    print(f"🔧 [Stage 1] optimize_prompt: {scene_description[:100]}")
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

    print(f"  ⏱️ Flash: {resp.status_code} ({time.time()-t0:.1f}s)")
    if resp.status_code != 200:
        return {"status": "error", "message": f"Flash API failed: {resp.status_code}"}

    optimized = ""
    for c in resp.json().get("candidates", []):
        for p in c.get("content", {}).get("parts", []):
            if "text" in p:
                optimized += p["text"]

    tool_context.state["optimized_prompt"] = optimized
    tool_context.state["generation_round"] = 0
    tool_context.state["total_generated"] = tool_context.state.get("total_generated", 0)  # preserve across refines
    tool_context.state.setdefault("passed_indices", [])
    tool_context.state.setdefault("failed_indices", [])
    tool_context.state.setdefault("all_scores", {})
    tool_context.state["eval_suggestions"] = ""
    print(f"  ✅ Prompt: {len(optimized)} chars")
    return {"status": "success", "prompt_length": len(optimized), "prompt_preview": optimized[:300]}


async def generate_and_evaluate(num_candidates: int, tool_context: ToolContext) -> dict:
    """Generate candidate images, evaluate each one, and save the best as final.png.
    Only final.png artifact will be visible to the user.

    Args:
        num_candidates: Number of candidate images to generate and evaluate (typically 2)
    """
    optimized_prompt = tool_context.state.get("optimized_prompt")
    if not optimized_prompt:
        return {"status": "error", "message": "No optimized prompt. Run optimize_prompt first."}

    num_candidates = min(max(num_candidates, 1), 4)
    round_num = tool_context.state.get("generation_round", 0) + 1
    tool_context.state["generation_round"] = round_num

    print(f"🎨 [Round {round_num}] Generating {num_candidates} candidates...")
    token = _get_token()

    # Build generation request
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

    # Generate candidates (keep in memory only)
    candidates_b64 = []
    for i in range(num_candidates):
        t0 = time.time()
        resp = requests.post(url, headers=headers, json=request_body)
        print(f"  🖼️ Candidate {i+1}: {resp.status_code} ({time.time()-t0:.1f}s)")
        if resp.status_code != 200:
            continue
        for c in resp.json().get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                if "inlineData" in p:
                    candidates_b64.append(p["inlineData"]["data"])

    if not candidates_b64:
        return {"status": "error", "message": "Image generation failed."}

    # Evaluate each candidate
    print(f"  🔍 Evaluating {len(candidates_b64)} candidates...")
    eval_url = (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
                f"/locations/{REGION}/publishers/google/models/gemini-3.5-flash:generateContent")

    # Build ref parts for eval
    ref_parts = []
    for ref in session_refs:
        ref_parts.append({"text": "[" + ref["role"] + "]:"})
        ref_parts.append({"inline_data": {"mime_type": ref["mime_type"], "data": ref["data"]}})

    scores = []
    issues_list = []
    for i, img_b64 in enumerate(candidates_b64):
        t0 = time.time()
        eval_parts = [{"text": EVALUATOR_SYSTEM}, {"text": "=== 角色参考图 ==="}]
        eval_parts.extend(ref_parts)
        eval_parts.append({"text": "=== 生成目标 ===\n" + optimized_prompt[:800]})
        eval_parts.append({"text": "=== 生成图 ==="})
        eval_parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
        eval_parts.append({"text": "对照参考图和提示词要求，评估这张图。输出JSON。"})

        resp = requests.post(eval_url, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"contents": [{"role": "USER", "parts": eval_parts}],
                  "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}})

        score = 0
        issue_text = ""
        if resp.status_code == 200:
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
                score = ev.get("score", 0)
                issue_text = "; ".join([x.get("issue","") for x in ev.get("issues",[])])
            except Exception:
                score = 5

        scores.append(score)
        issues_list.append(issue_text)
        print(f"  📊 Candidate {i+1}: {score}/10 ({time.time()-t0:.1f}s)")

    # Pick the best
    best_idx = scores.index(max(scores))
    best_score = scores[best_idx]

    # Track best score across rounds (no image data to avoid 10MB state limit)
    tool_context.state["best_overall_score"] = max(tool_context.state.get("best_overall_score", 0), best_score)

    # If passes threshold, save as final
    if best_score >= 8:
        img_bytes = base64.b64decode(candidates_b64[best_idx])
        art = types.Part(inline_data=types.Blob(data=img_bytes, mime_type="image/png"))
        await tool_context.save_artifact(filename="final.png", artifact=art)
        print(f"  🎉 Passed! Saved final.png (score={best_score})")
        return {
            "status": "success",
            "goal_reached": True,
            "final_image": "final.png",
            "best_score": best_score,
            "round": round_num,
            "all_scores": [{"candidate": i+1, "score": s, "issues": iss} for i, (s, iss) in enumerate(zip(scores, issues_list))]
        }

    # If 3 rounds exhausted, use current round's best candidate
    if round_num >= 3:
        img_bytes = base64.b64decode(candidates_b64[best_idx])
        art = types.Part(inline_data=types.Blob(data=img_bytes, mime_type="image/png"))
        await tool_context.save_artifact(filename="final.png", artifact=art)
        print(f"  ⚠️ 3 rounds done. Best this round: {best_score}/10. Saved final.png")
        return {
            "status": "success",
            "goal_reached": True,
            "final_image": "final.png",
            "best_score": best_score,
            "round": round_num,
            "fallback": True,
            "all_scores": [{"candidate": i+1, "score": s, "issues": iss} for i, (s, iss) in enumerate(zip(scores, issues_list))]
        }

    # Need more rounds
    suggestions = "; ".join([iss for iss in issues_list if iss])
    tool_context.state["eval_suggestions"] = suggestions
    return {
        "status": "needs_refinement",
        "goal_reached": False,
        "best_score": best_score,
        "round": round_num,
        "all_scores": [{"candidate": i+1, "score": s, "issues": iss} for i, (s, iss) in enumerate(zip(scores, issues_list))],
        "suggestions": suggestions[:300]
    }


async def refine_prompt(tool_context: ToolContext) -> dict:
    """Refine the prompt based on evaluation feedback to fix character consistency issues.
    Call this before generating more images when evaluation found issues.
    """
    suggestions = tool_context.state.get("eval_suggestions", "")
    original_prompt = tool_context.state.get("optimized_prompt", "")
    if not suggestions:
        return {"status": "skip", "message": "No suggestions to refine."}

    print("📝 [Refine] Refining prompt based on evaluation...")
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
    print(f"  ✅ Prompt refined: {len(refined)} chars")
    return {"status": "success", "prompt_length": len(refined)}


async def edit_image(image_name: str, edit_instruction: str, tool_context: ToolContext) -> dict:
    """Edit a specific final image based on user feedback. Generates 2 candidates, evaluates, and picks the best one.

    Args:
        image_name: Which image to edit, e.g. "final_1.png"
        edit_instruction: What to change, e.g. "熊的腮红要更明显，颜色要更粉"
    """
    print(f"✏️ [Edit] {image_name}: {edit_instruction[:80]}")
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
        print(f"  🖼️ Generating edit candidate {i+1}/2...")
        resp = requests.post(url, headers=headers, json=request_body)
        print(f"  ⏱️ {resp.status_code} ({time.time()-t0:.1f}s)")
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
                    print(f"  ✅ edit_candidate_{i+1}.png ({len(img_bytes)} bytes)")

    if not edit_candidates:
        return {"status": "error", "message": "Failed to generate edit candidates."}

    # Evaluate each candidate
    print("  🔍 Evaluating edit candidates...")
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
        parts.append({"text": f"=== 生成目标（提示词要求）===\n{tool_context.state.get('optimized_prompt', '')[:1000]}"})
        parts.append({"text": "=== 生成图 ==="})
        parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
        parts.append({"text": "对照参考图和提示词要求，评估这张图。输出JSON。"})

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
        print(f"  📊 edit_candidate_{i+1}: score={score}/10")
        results.append({"index": i+1, "score": score, "pass": score >= 8})

        if score > best_score:
            best_score = score
            best_idx = i

    # If best passes threshold, replace the original
    if best_score >= 8:
        best_artifact = types.Part(inline_data=types.Blob(data=edit_candidates[best_idx], mime_type="image/png"))
        await tool_context.save_artifact(filename=image_name, artifact=best_artifact)
        print(f"  🎉 Edit passed! Replaced {image_name} with candidate {best_idx+1} (score={best_score})")
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
        print(f"  ⚠️ Edit candidates didn't pass (best={best_score}). Keeping original.")
        return {
            "status": "success",
            "replaced": False,
            "image": image_name,
            "best_score": best_score,
            "candidates": results,
            "message": f"Edit candidates scored {best_score}/10, below threshold 8. Original kept. Try different instructions."
        }




# === Agent Definition ===
from google.adk.agents.callback_context import CallbackContext as _CBCtx
from google.adk.models.llm_request import LlmRequest as _LReq

def capture_user_images(callback_context: _CBCtx, llm_request: _LReq) -> None:
    """Before model callback: extract user-uploaded images from request into state."""
    for content in llm_request.contents:
        if content.role == "user" and content.parts:
            for part in content.parts:
                if hasattr(part, "inline_data") and part.inline_data and getattr(part.inline_data, "data", None):
                    img_b64 = base64.b64encode(part.inline_data.data).decode()
                    mime = part.inline_data.mime_type or "image/png"
                    captured = callback_context.state.get("captured_images", [])
                    if not any(c.get("data", "")[:50] == img_b64[:50] for c in captured):
                        captured.append({"data": img_b64, "mime_type": mime, "role": "user_upload"})
                        callback_context.state["captured_images"] = captured


def _capture_user_images(callback_context, llm_request) -> None:
    """Before model callback: extract user images from LLM request into state."""
    for content in llm_request.contents:
        if content.role == "user" and content.parts:
            for part in content.parts:
                idata = getattr(part, "inline_data", None)
                if idata and getattr(idata, "data", None):
                    img_b64 = base64.b64encode(idata.data).decode()
                    mime = idata.mime_type or "image/png"
                    captured = callback_context.state.get("captured_images", [])
                    if not any(c.get("data", "")[:50] == img_b64[:50] for c in captured):
                        captured.append({"data": img_b64, "mime_type": mime, "role": "user_upload"})
                        callback_context.state["captured_images"] = captured


root_agent = Agent(
    model=GeminiWithLocation(model="gemini-3.5-flash", location="global"),
    name="ref2image_agent",
    description="Upload reference images, generate matching images with automatic quality evaluation and iterative refinement.",
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
3. generate_and_evaluate(2) — 生成2张候选并立即评估打分
4. 如果返回 goal_reached=true → 最终图已保存为 final.png
5. 如果返回 goal_reached=false → refine_prompt → generate_and_evaluate(2)（最多3轮）

⚠️ 严格规则：
- 你不能凭空描述图片！必须调用 generate_images 工具实际生成图片！
- 绝对禁止：不调用工具就告诉用户"图已生成"
- 只使用用户上传的参考图
- 最多3轮重试后必须给出结果

=== 输出格式（严格遵守）===
每一步都告知用户进度，最终回复按以下顺序输出：

1. 各轮次过程（分数+问题+优化措施）
2. 最终选定说明（如"综合3轮共6张候选，候选3得分最高"）
3. 画面亮点描述
4. 修改建议
5. 最后一行写："✅ 最终选定图片（得分 X/10）："
6. 紧接着展示图片

⚠️ 图片必须放在所有文字的最后面！先文字再图片！禁止图片出现在文字前面或中间。

=== 参考图管理 ===
- 用户附图 + 说明 → add_reference_image(role="描述")
- 支持任何类型：角色、风格、场景、构图参考等
- 用户可多次上传逐步补充

=== 编辑流程 ===
用户说"修改第X张图" → edit_image

=== 重要 ===
- 评估标准基于用户提供的参考图，没有参考图则只评估画面质量和动作合理性
- 灵活适应各种生图需求""",
    tools=[add_reference_image, optimize_prompt, generate_and_evaluate, refine_prompt, edit_image],
    before_model_callback=_capture_user_images,

)
