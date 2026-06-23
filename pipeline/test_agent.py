"""
Test the ADK agent locally using Runner (no deployment needed).
"""
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from image_gen_agent.agent import root_agent


async def main():
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()

    runner = Runner(
        agent=root_agent,
        app_name="image_gen_test",
        session_service=session_service,
        artifact_service=artifact_service,
    )

    # Create session
    session = await session_service.create_session(app_name="image_gen_test", user_id="test_user")

    print("=== Sending request to agent ===")
    from google.genai import types
    user_msg = types.Content(
        role="user",
        parts=[types.Part(text="3D卡通风格，水平视角，小动物们在山西双林寺里玩耍，有古建，古塔，古树")]
    )

    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=user_msg,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"[{event.author}]: {part.text[:200]}")
                if part.function_call:
                    print(f"[TOOL CALL]: {part.function_call.name}({dict(part.function_call.args) if part.function_call.args else {}})")
                if part.function_response:
                    print(f"[TOOL RESULT]: {str(part.function_response)[:200]}")
        if event.actions and event.actions.artifact_delta:
            print(f"[ARTIFACTS]: {event.actions.artifact_delta}")

    # Check artifacts
    artifacts = await artifact_service.list_artifact_keys(
        app_name="image_gen_test", user_id="test_user", session_id=session.id
    )
    print(f"\n=== Generated artifacts: {artifacts} ===")

    # Save artifacts to disk
    for fname in artifacts:
        art = await artifact_service.load_artifact(
            app_name="image_gen_test", user_id="test_user", session_id=session.id, filename=fname
        )
        if art and art.inline_data and art.inline_data.data:
            out_path = f"/tmp/nano_images/output/adk_{fname}"
            with open(out_path, "wb") as f:
                f.write(art.inline_data.data)
            print(f"  Saved: {out_path} ({len(art.inline_data.data)} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
