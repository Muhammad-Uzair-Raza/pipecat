#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import sys
from deepgram import LiveOptions
from dotenv import load_dotenv

from loguru import logger
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.openai.llm import OpenAILLMContext, OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

load_dotenv(override=True)

class IntakeProcessor:
    def __init__(self, context: OpenAILLMContext):
        print("Initializing context from IntakeProcessor")

        # Inject Jessica's system instruction
        context.add_message(
            {
                "role": "system",
                "content": (
                    "You are Jessica, an agent for a company called Tri-County Health Services. "
                    "Your job is to collect important information from the user before their doctor visit. "
                    "You're talking to Chad Bailey. You should address the user by their first name and be polite and professional. "
                    "You're not a medical professional, so you shouldn't provide any advice. Keep your responses short. "
                    "Your job is to collect information to give to a doctor. Don't make assumptions about what values to plug into functions. "
                    "Ask for clarification if a user response is ambiguous. Start by introducing yourself. "
                    "Then, ask the user to tell you their full name. When they answer with their name, call the verify_name function."
                ),
            }
        )

        context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "verify_name",
                        "description": "Use this function to verify the user has provided their correct full name.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The user's full name.",
                                }
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "verify_age",
                        "description": "Use this function to verify the user's age.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "age": {
                                    "type": "integer",
                                    "description": "The user's age in years.",
                                }
                            },
                        },
                    },
                },
            ]
        )

    async def verify_name(self, params):
        if params.arguments["name"].lower() == "chad bailey":
            await params.result_callback(
                [
                    {
                        "role": "system",
                        "content": "Thank you for confirming your name. Now, please tell me your age in years.",
                    }
                ]
            )
        else:
            await params.result_callback(
                [
                    {
                        "role": "system",
                        "content": "The name you provided doesn't match our records. Please try again.",
                    }
                ]
            )

    async def verify_age(self, params):
        if int(params.arguments["age"]) == 42:
            await params.result_callback(
                [
                    {
                        "role": "system",
                        "content": "Thank you for confirming your age. Please list any prescription medications you're currently taking, including medication name and dosage.",
                    }
                ]
            )
        else:
            await params.result_callback(
                [
                    {
                        "role": "system",
                        "content": "The age you provided doesn't match our records. Please try again.",
                    }
                ]
            )


async def run_bot(webrtc_connection):
    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=2,
        ),
    )

    print("openai key", os.getenv("OPENAI_API_KEY"))

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o",
        # ❌ system_instruction removed
    )

    context = OpenAILLMContext()
    context_aggregator = llm.create_context_aggregator(context)

    # Register function logic
    intake = IntakeProcessor(context)
    llm.register_function("verify_name", intake.verify_name)
    llm.register_function("verify_age", intake.verify_age)

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(vad_events=True, utterance_end_ms="1000"),
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            pipecat_transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
        ),
    )

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")

    @pipecat_transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info("Pipecat Client closed")

        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
