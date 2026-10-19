# Management Agent: Condo Emergency Triage & Dispatch Platform
## Central Knowledge Base & Project Roadmap

Welcome to the **Knowledge Base** for the AI Condominium Management and Emergency Triage System. This folder contains the complete system architecture, operational specs, and phased roadmap to take this project from its current working prototype to a production-ready property management platform.

---

### Navigation Index

| Document | Description |
| :--- | :--- |
| **[00. Executive Summary & Vision](00_EXECUTIVE_SUMMARY.md)** | Core product vision, problem statement, key differentiators, and high-level workflow. |
| **[01. System Architecture](01_SYSTEM_ARCHITECTURE.md)** | End-to-end technical architecture, component diagrams, and data flow. |
| **[02. Phased Roadmap & Milestones](02_PHASED_ROADMAP.md)** | Comprehensive 5-phase execution plan from PoC to enterprise scale. |
| **[03. Data Models & Schemas](03_DATA_MODELS_AND_SCHEMAS.md)** | SQLite database schemas, Pydantic validation models, and state transitions. |
| **[04. Vendor Directory & SLA Specifications](04_CONTRACTOR_DIRECTORY_AND_SLA.md)** | Incident classification rules, vendor matching matrix, and response SLAs. |
| **[05. Telegram Bot Specification](05_TELEGRAM_BOT_SPECIFICATION.md)** | UX/UI design, commands, multimodal photo handling, and inline interactive controls. |

---

### Project Quick Facts
* **Primary Interface:** Telegram Bot ([`@CondoManagementAgentBot`](https://t.me/CondoManagementAgentBot)) + Streamlit Web Command Center.
* **Core AI Engine:** Google Gemini 3.8 Flash (Multimodal vision + Pydantic structured output).
* **Target Audience:** Condominium Residents, On-site Security/Guardhouse, Property Managers, and Contracted Technicians.
