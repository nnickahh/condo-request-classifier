"""
Cloud entrypoint for 24/7 hosting on Render, Koyeb, Railway, or Docker.
Runs the CMMS REST API server on $PORT for healthchecks,
while concurrently running the Condo Management Telegram Bot long-polling loop.
"""

import logging
import os
import sys
import threading
from dotenv import load_dotenv

from condo_jev.telegram_bot import CondoManagementTelegramBot
from condo_jev.work_order_api import CMMSApiServer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("condo_cloud")


def main() -> None:
    port = int(os.getenv("PORT", os.getenv("WORK_ORDER_API_PORT", "8000")))
    host = os.getenv("HOST", "0.0.0.0")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    staff_chat = os.getenv("TELEGRAM_STAFF_CHAT_ID")

    logger.info("=" * 60)
    logger.info(" CONDO JEV 24/7 CLOUD SERVICE & TELEGRAM BOT")
    logger.info("=" * 60)

    # 1. Start HTTP Healthcheck & CMMS API server in background thread
    api_server = CMMSApiServer(host=host, port=port)
    api_server.start_background()
    logger.info(f"Healthcheck and CMMS API live on http://{host}:{port}/api/health")

    # 2. Start Telegram Bot polling
    if not token:
        logger.warning("No TELEGRAM_BOT_TOKEN found in environment. Telegram bot polling disabled.")
        logger.info("Service will keep HTTP API alive. Add TELEGRAM_BOT_TOKEN to enable the bot.")
        try:
            threading.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down API server...")
            api_server.shutdown()
        return

    logger.info("Connecting to Telegram Bot and starting 24/7 polling loop...")
    bot = CondoManagementTelegramBot(
        token=token,
        staff_chat_id=staff_chat,
        work_order_store=api_server.store,
        triage_service=api_server.triage_service,
    )
    try:
        bot.run_polling()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down 24/7 cloud service...")
    finally:
        api_server.shutdown()


if __name__ == "__main__":
    main()
