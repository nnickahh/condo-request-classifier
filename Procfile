web: python run_cloud.py
worker: python -m condo_jev.telegram_bot
dashboard: python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0
api: python -m condo_jev.work_order_api --port $PORT
