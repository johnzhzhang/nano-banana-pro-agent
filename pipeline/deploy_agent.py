"""
Deploy the image generation agent to Gemini Enterprise Agent Platform.

Prerequisites:
  pip install google-cloud-aiplatform[agent_engines,adk]>=1.112
  gcloud auth application-default login
"""
import vertexai
from vertexai import agent_engines
from google.cloud.aiplatform import types as platform_types
from image_gen_agent.agent import root_agent

PROJECT_ID = "john-poc-453315"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://john-poc-453315-adk-staging"

# Initialize
client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

# Wrap agent
app = agent_engines.AdkApp(agent=root_agent)

# Test locally first
print("=== Local test ===")
import asyncio
async def test_local():
    async for event in app.async_stream_query(
        user_id="test_user",
        message="3D卡通风格，小动物们在长城上玩耍",
    ):
        print(event)

# asyncio.run(test_local())  # Uncomment to test locally

# Deploy to Agent Runtime
print("=== Deploying to Agent Runtime ===")
remote_agent = client.agent_engines.create(
    agent=app,
    config={
        "requirements": [
            "google-cloud-aiplatform[agent_engines,adk]>=1.112",
            "requests",
            "pyyaml",
        ],
        "staging_bucket": STAGING_BUCKET,
        "identity_type": platform_types.IdentityType.AGENT_IDENTITY,
    }
)

print(f"✅ Deployed! Resource name: {remote_agent.resource_name}")

# Test deployed agent
print("=== Testing deployed agent ===")
async def test_remote():
    async for event in remote_agent.async_stream_query(
        user_id="test_user",
        message="3D卡通风格，水平视角，小动物们在山西双林寺里玩耍，有古建，古塔，古树",
    ):
        print(event)

# asyncio.run(test_remote())  # Uncomment to test
