from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from condo_jev.backends import LocalJevSimulatorBackend
from condo_jev.schema import ResidentRequest
from condo_jev.work_order import (
    ESTATE_TECHNICIANS,
    WorkOrderStatus,
    WorkOrderStore,
    dispatch_outbound_webhook,
)
from condo_jev.workflow import JevTriageService

logger = logging.getLogger("condo_jev.work_order_api")


class CMMSApiHandler(BaseHTTPRequestHandler):
    """REST API Request Handler for the Condominium Work Order Management System."""

    server_version = "CondoCMMS-API/1.0"

    def __init__(self, *args, **kwargs):
        # We access store and service from server attributes
        super().__init__(*args, **kwargs)

    @property
    def store(self) -> WorkOrderStore:
        return getattr(self.server, "work_order_store", None) or WorkOrderStore()

    @property
    def triage_service(self) -> JevTriageService:
        return getattr(self.server, "triage_service", None) or JevTriageService(
            backend=LocalJevSimulatorBackend()
        )

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Requested-With, X-Condo-Signature",
        )

    def _send_json_response(self, data: Any, status_code: int = 200) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_json(self, message: str, status_code: int = 400) -> None:
        self._send_json_response({"error": message, "status": status_code}, status_code=status_code)

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip("/")
        if not path:
            path = "/"
        params = parse_qs(parsed_url.query)

        # 1. Root / Interactive API Documentation
        if path in ("/", "/api", "/docs"):
            self._handle_docs_get()
            return

        # 2. Health check
        if path in ("/api/health", "/health"):
            stats = self.store.get_cmms_stats()
            self._send_json_response(
                {
                    "status": "healthy",
                    "service": "condo-cmms-work-order-api",
                    "version": "1.0.0",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_work_orders": stats.get("total_work_orders", 0),
                    "open_work_orders": stats.get("open_work_orders", 0),
                }
            )
            return

        # 3. Operational KPIs & Statistics
        if path == "/api/stats":
            self._send_json_response(self.store.get_cmms_stats())
            return

        # 4. Technician & Contractor Roster
        if path == "/api/technicians":
            self._send_json_response({"technicians": ESTATE_TECHNICIANS})
            return

        # 5. List all Work Orders (with query filtering)
        if path == "/api/work-orders":
            limit = int(params.get("limit", ["100"])[0])
            status_filter = params.get("status", [None])[0]
            dept_filter = params.get("category", [None])[0]
            prio_filter = params.get("priority", [None])[0]
            resident_id = params.get("resident_id", [None])[0]

            orders = self.store.list_all(limit=limit)

            if resident_id:
                orders = [o for o in orders if o.resident_id == resident_id.strip()]
            if status_filter:
                orders = [o for o in orders if status_filter.lower() in o.status.lower()]
            if dept_filter:
                orders = [o for o in orders if dept_filter.lower() in o.category_title.lower() or dept_filter.lower() == str(o.category).lower()]
            if prio_filter:
                orders = [o for o in orders if prio_filter.lower() == str(o.priority.value).lower()]

            self._send_json_response(
                {
                    "count": len(orders),
                    "total_in_store": len(self.store.list_all(limit=1000)),
                    "work_orders": [o.model_dump() for o in orders],
                }
            )
            return

        # 6. Single Work Order lookup: /api/work-orders/{id}
        if path.startswith("/api/work-orders/"):
            wo_id = path.replace("/api/work-orders/", "").strip()
            wo = self.store.get(wo_id)
            if not wo:
                self._send_error_json(f"Work order '{wo_id}' not found", status_code=404)
                return
            self._send_json_response(wo.model_dump())
            return

        # 7. Telegram Bot status endpoint
        if path == "/api/telegram/status":
            token = os.getenv("TELEGRAM_BOT_TOKEN")
            staff_chat = os.getenv("TELEGRAM_STAFF_CHAT_ID")
            has_token = bool(token)
            bot_info = None
            if has_token:
                try:
                    import urllib.request
                    res = urllib.request.urlopen(f"https://api.telegram.org/bot{token}/getMe", timeout=3)
                    bot_info = json.loads(res.read().decode()).get("result")
                except Exception as exc:
                    bot_info = {"error": str(exc)}
            self._send_json_response(
                {
                    "telegram_configured": has_token,
                    "staff_chat_configured": bool(staff_chat),
                    "staff_chat_id": staff_chat,
                    "bot": bot_info,
                }
            )
            return

        # 8. CSV Export endpoint
        if path == "/api/export/csv":
            csv_data = self.store.export_csv().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="work_orders.csv"')
            self.send_header("Content-Length", str(len(csv_data)))
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(csv_data)
            return

        self._send_error_json(f"Endpoint '{path}' not found", status_code=404)

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip("/")

        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_error_json("Empty request body. JSON payload required.", status_code=400)
            return

        try:
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self._send_error_json(f"Invalid JSON: {exc}", status_code=400)
            return

        # 1. Create a new Work Order via POST /api/work-orders
        if path == "/api/work-orders":
            text = data.get("text", "").strip()
            if not text:
                self._send_error_json("Field 'text' (resident issue report) is required", status_code=400)
                return

            channel = data.get("channel", "REST API")
            location = data.get("location")
            resident_id = str(data.get("resident_id", "api_client"))
            resident_name = data.get("resident_name", "API User")
            resident_username = data.get("resident_username")

            try:
                req = ResidentRequest(text=text, channel=channel, location=location or None)
                decision = self.triage_service.classify(req)
                wo = self.store.create_work_order(
                    request=req,
                    classification=decision,
                    resident_id=resident_id,
                    resident_username=resident_username,
                    resident_name=resident_name,
                )
                self._send_json_response(
                    {
                        "message": "Work Order created and dispatched successfully",
                        "work_order": wo.model_dump(),
                    },
                    status_code=201,
                )
            except Exception as exc:
                self._send_error_json(f"Failed to triage and create work order: {exc}", status_code=500)
            return

        # 2. Test an outbound webhook via POST /api/webhooks/test
        if path == "/api/webhooks/test":
            target_url = data.get("webhook_url") or os.getenv("WORK_ORDER_WEBHOOK_URL")
            if not target_url:
                self._send_error_json("Field 'webhook_url' is required", status_code=400)
                return

            sample_order = self.store.list_all(limit=1)
            test_payload = (
                sample_order[0].model_dump()
                if sample_order
                else {
                    "work_order_id": "WO-EST-TEST",
                    "text": "Simulated webhook test from CMMS API",
                    "status": "Test Ping",
                }
            )

            result = dispatch_outbound_webhook(
                event_type="test.ping",
                work_order_data=test_payload,
                webhook_url=target_url,
                secret=data.get("secret"),
            )
            self._send_json_response(
                {
                    "message": "Webhook test executed",
                    "target_url": target_url,
                    "result": result,
                }
            )
            return

        # 3. Direct concierge message to resident on Telegram: POST /api/work-orders/{id}/notify
        if path.startswith("/api/work-orders/") and path.endswith("/notify"):
            wo_id = path.replace("/api/work-orders/", "").replace("/notify", "").strip()
            msg = data.get("message", "").strip()
            if not msg:
                self._send_error_json("Field 'message' is required", status_code=400)
                return
            sender = data.get("sender_name", "Estate Management Office")
            res = self.store.send_resident_message(wo_id, message=msg, sender_name=sender)
            if res.get("success"):
                self._send_json_response({"message": "Notification dispatched to resident via Telegram", "result": res})
            else:
                self._send_error_json(f"Failed to deliver notification: {res.get('error')}", status_code=400)
            return

        # 4. Reopen work order: POST /api/work-orders/{id}/reopen
        if path.startswith("/api/work-orders/") and path.endswith("/reopen"):
            wo_id = path.replace("/api/work-orders/", "").replace("/reopen", "").strip()
            reason = data.get("reason", "Reopened via REST API")
            updated = self.store.reopen_work_order(wo_id, reason=reason)
            if not updated:
                self._send_error_json(f"Work order '{wo_id}' not found", status_code=404)
                return
            self._send_json_response(
                {
                    "message": f"Work order {updated.work_order_id} reopened successfully",
                    "work_order": updated.model_dump(),
                }
            )
            return

        self._send_error_json(f"POST endpoint '{path}' not found", status_code=404)

    def do_PATCH(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip("/")

        if not path.startswith("/api/work-orders/"):
            self._send_error_json(f"PATCH endpoint '{path}' not found", status_code=404)
            return

        wo_id = path.replace("/api/work-orders/", "").strip()
        wo = self.store.get(wo_id)
        if not wo:
            self._send_error_json(f"Work order '{wo_id}' not found", status_code=404)
            return

        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_error_json("Empty request body. JSON payload required.", status_code=400)
            return

        try:
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self._send_error_json(f"Invalid JSON: {exc}", status_code=400)
            return

        updated_wo = wo
        notify_res = bool(data.get("notify_resident", True))

        # Assign technician
        if "assigned_technician" in data:
            tech_name = data["assigned_technician"]
            phone = data.get("technician_phone")
            vendor = data.get("technician_vendor")
            note = data.get("note")
            res = self.store.assign_technician(
                wo_id,
                tech_name,
                phone=phone,
                vendor=vendor,
                note=note,
                notify_resident=notify_res,
            )
            if res:
                updated_wo = res
            if "status" in data:
                res_status = self.store.update_status(
                    wo_id,
                    new_status=data["status"],
                    notify_resident=notify_res,
                )
                if res_status:
                    updated_wo = res_status

        # Resolve work order
        elif data.get("status") in (WorkOrderStatus.RESOLVED.value, "Resolved") or "resolution_summary" in data:
            res_summary = data.get("resolution_summary", "Work completed by technician.")
            labor = data.get("labor_hours")
            cost = data.get("cost_sgd")
            parts = data.get("parts_or_materials")
            note = data.get("note")
            res = self.store.resolve_work_order(
                wo_id,
                resolution_summary=res_summary,
                labor_hours=labor,
                cost_sgd=cost,
                parts=parts,
                note=note,
                notify_resident=notify_res,
            )
            if res:
                updated_wo = res

        # Close work order
        elif data.get("status") in (WorkOrderStatus.CLOSED.value, "Closed"):
            res = self.store.close_work_order(
                wo_id,
                note=data.get("note"),
                notify_resident=notify_res,
            )
            if res:
                updated_wo = res

        # General status or note update
        elif "status" in data or "note" in data:
            new_status = data.get("status", updated_wo.status)
            note = data.get("note")
            res = self.store.update_status(
                wo_id,
                new_status=new_status,
                note=note,
                notify_resident=notify_res,
            )
            if res:
                updated_wo = res

        # Location update
        if "location" in data:
            loc = data["location"]
            res = self.store.update_location(wo_id, loc)
            if res:
                updated_wo = res

        self._send_json_response(
            {
                "message": f"Work Order {updated_wo.work_order_id} updated successfully",
                "work_order": updated_wo.model_dump(),
            }
        )


    def _handle_docs_get(self) -> None:
        """Render a clean, modern HTML documentation page for the Work Order REST API."""
        stats = self.store.get_cmms_stats()
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Condo CMMS Work Order REST API</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0; padding: 30px; background: #0f172a; color: #f8fafc; line-height: 1.6;
    }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    h1 {{ color: #38bdf8; font-size: 2rem; margin-bottom: 0.2rem; }}
    .subtitle {{ color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem; }}
    .kpi-row {{ display: flex; gap: 15px; margin-bottom: 30px; flex-wrap: wrap; }}
    .kpi-card {{
      background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px 20px;
      flex: 1; min-width: 150px; text-align: center;
    }}
    .kpi-val {{ font-size: 1.8rem; font-weight: 700; color: #38bdf8; }}
    .kpi-label {{ font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
    .endpoint-card {{
      background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 20px;
    }}
    .method {{
      display: inline-block; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 0.85rem; margin-right: 8px;
    }}
    .get {{ background: #0284c7; color: white; }}
    .post {{ background: #16a34a; color: white; }}
    .patch {{ background: #d97706; color: white; }}
    .path {{ font-family: monospace; font-size: 1.1rem; font-weight: 600; color: #f1f5f9; }}
    pre {{
      background: #090d16; padding: 12px; border-radius: 6px; overflow-x: auto; color: #a5f3fc; font-size: 0.9rem;
    }}
    .tag {{ background: #334155; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; color: #cbd5e1; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🛠️ Condominium Work Order System (CMMS)</h1>
    <div class="subtitle">REST API & Webhook Dispatch Service · Real-Time Estate Operations Integration</div>

    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-val">{stats.get("total_work_orders", 0)}</div>
        <div class="kpi-label">Total Work Orders</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{stats.get("open_work_orders", 0)}</div>
        <div class="kpi-label">Active / Open</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{stats.get("assigned", 0)}</div>
        <div class="kpi-label">Assigned to Tech</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{stats.get("resolved", 0)}</div>
        <div class="kpi-label">Resolved / Closed</div>
      </div>
    </div>

    <h2>API Endpoints Reference</h2>

    <div class="endpoint-card">
      <span class="method get">GET</span> <span class="path">/api/work-orders</span>
      <p>List all work orders with optional filtering by status, category, priority, or resident.</p>
      <pre>curl -X GET "http://localhost:8000/api/work-orders?status=In+Progress&limit=10"</pre>
    </div>

    <div class="endpoint-card">
      <span class="method get">GET</span> <span class="path">/api/work-orders/&#123;id&#125;</span>
      <p>Retrieve full details of a specific work order by ID (e.g. <code>WO-EST-1042</code> or <code>1042</code>).</p>
      <pre>curl -X GET "http://localhost:8000/api/work-orders/WO-EST-1042"</pre>
    </div>

    <div class="endpoint-card">
      <span class="method post">POST</span> <span class="path">/api/work-orders</span>
      <p>Create a new work order from an external system. Automatically runs Jev System-One triage.</p>
      <pre>curl -X POST "http://localhost:8000/api/work-orders" \
  -H "Content-Type: application/json" \
  -d '{{
    "text": "The gym air conditioner is blowing hot air.",
    "channel": "External CMMS",
    "location": "Clubhouse Gym",
    "resident_name": "Facilities Officer"
  }}'</pre>
    </div>

    <div class="endpoint-card">
      <span class="method patch">PATCH</span> <span class="path">/api/work-orders/&#123;id&#125;</span>
      <p>Update work order status, assign a duty technician, or submit resolution and labor cost.</p>
      <pre>curl -X PATCH "http://localhost:8000/api/work-orders/WO-EST-1042" \
  -H "Content-Type: application/json" \
  -d '{{
    "status": "In Progress",
    "assigned_technician": "John Tan",
    "technician_phone": "+65 9123 4567",
    "note": "Technician on site inspecting corridor light."
  }}'</pre>
    </div>

    <div class="endpoint-card">
      <span class="method get">GET</span> <span class="path">/api/technicians</span>
      <p>Fetch list of available estate technicians and specialized contractors.</p>
      <pre>curl -X GET "http://localhost:8000/api/technicians"</pre>
    </div>

    <div class="endpoint-card">
      <span class="method get">GET</span> <span class="path">/api/stats</span>
      <p>Operational KPIs: SLA rates, volume by department, and priority distribution.</p>
      <pre>curl -X GET "http://localhost:8000/api/stats"</pre>
    </div>

    <div class="endpoint-card">
      <span class="method get">GET</span> <span class="path">/api/export/csv</span>
      <p>Download full Work Order registry as standard CSV.</p>
      <pre>curl -X GET "http://localhost:8000/api/export/csv" -o work_orders.csv</pre>
    </div>

    <div class="endpoint-card">
      <span class="method post">POST</span> <span class="path">/api/webhooks/test</span>
      <p>Send a test event payload to your configured external webhook URL (Jira, ServiceNow, Slack, Zapier).</p>
      <pre>curl -X POST "http://localhost:8000/api/webhooks/test" \
  -H "Content-Type: application/json" \
  -d '{{"webhook_url": "https://webhook.site/your-id"}}'</pre>
    </div>
  </div>
</body>
</html>"""
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(encoded)


class CMMSApiServer:
    """Multi-threaded HTTP Server for the Condo CMMS REST API."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        store: WorkOrderStore | None = None,
        triage_service: JevTriageService | None = None,
    ):
        self.host = host
        self.port = port
        self.store = store or WorkOrderStore()
        self.triage_service = triage_service or JevTriageService(backend=LocalJevSimulatorBackend())
        self.server = ThreadingHTTPServer((self.host, self.port), CMMSApiHandler)
        self.server.work_order_store = self.store  # type: ignore[attr-defined]
        self.server.triage_service = self.triage_service  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    def start_background(self) -> None:
        """Start the server in a background daemon thread."""
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Condo CMMS REST API started on http://{self.host}:{self.port}")

    def serve_forever(self) -> None:
        """Start the server in blocking mode."""
        print("=" * 60, flush=True)
        print("  CONDOMINIUM WORK ORDER SYSTEM (CMMS) REST API SERVICE", flush=True)
        print("=" * 60, flush=True)
        print(f"API listening at: http://{self.host}:{self.port}/", flush=True)
        print(f"Interactive Docs: http://localhost:{self.port}/docs", flush=True)
        print(f"Health Check    : http://localhost:{self.port}/api/health", flush=True)
        print("Press Ctrl+C to terminate.", flush=True)
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down CMMS API server...", flush=True)
            self.server.shutdown()

    def shutdown(self) -> None:
        self.server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Condominium CMMS Work Order REST API Server")
    parser.add_argument("--host", default=os.getenv("WORK_ORDER_API_HOST", "0.0.0.0"), help="Host address to bind")
    parser.add_argument("--port", type=int, default=int(os.getenv("WORK_ORDER_API_PORT", "8000")), help="Port to listen on")
    args = parser.parse_args()

    server = CMMSApiServer(host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
