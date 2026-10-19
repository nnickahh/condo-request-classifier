# 04. Vendor Directory & SLA Specifications

This document defines the default contractor mapping matrix, priority triggers, contact protocols, and service level agreements (SLAs).

---

## 1. Vendor Routing Matrix

| Category / Domain | Primary Contractor Entity | Phone Contact | Emergency Trigger | SLA Target |
| :--- | :--- | :--- | :--- | :--- |
| **Plumbing & Flooding** | **Apex 24/7 Rapid Plumbing** | `+65 6789-0123` | Burst pipe, active flooding, sewage overflow | **< 20–30 mins** |
| **Lifts & Escalators** | **Schindler Express Response** | `+65 6888-4321` | Trapped passengers, elevator stoppage | **< 15–20 mins** |
| **Electrical & Fire** | **VoltCare Electrical Hotline** | `+65 6555-9999` | Electrical sparks, burning smell, total blackout | **< 25 mins** |
| **Security & Trespass** | **Main Guardhouse & Patrol** | `+65 6444-2222` | Intruder, tailgating, fight, fire alarm | **Immediate (< 5 mins)** |
| **Noise & Nuisance** | **Resident Relations Patrol** | `+65 6444-2222` | Late night parties (>10:30 PM), loud music | **< 45 mins** |
| **Cleaning & Hygiene** | **CleanSweep Estate Services** | `+65 6333-1111` | Spill in lift, garbage chute choke | **< 2 hours** |
| **Facilities & Booking**| **Clubhouse Management Office**| `+65 6222-0000` | Pool, BBQ pit, gym access queries | **< 4 hours** |
| **Billing & Payments**  | **Condo Accounts Dept**        | `+65 6111-5555` | Maintenance fee double billing, refunds | **Next business day** |

---

## 2. Priority Escalation Rules

### Priority 1: Emergency (Auto-Dispatch Enabled)
* **Conditions:**
  * Active water flooding or pipe rupture.
  * Elevator entrapment with people inside.
  * Electrical sparking, smoke, or fire hazards.
  * Structural damage or uncontained security breaches.
* **Autonomous Action:**
  1. Instant contractor SMS/alert dispatched.
  2. Guardhouse access clearance pre-authorized for contractor vehicle.
  3. Actionable interim safety advice rendered to resident.
  4. Flagged with blinking red alert in the management dashboard.

### Priority 2: High (Urgent Staff Triage)
* **Conditions:**
  * Intermittent elevator jerky motion (no one trapped).
  * Main driveway barrier arm stuck open.
  * Ongoing loud noise past permitted hours.
* **Autonomous Action:**
  1. Alert logged in staff queue.
  2. Guardhouse dispatched for on-site inspection.
  3. Resident given ticket confirmation.

### Priority 3: Normal / Routine
* **Conditions:**
  * Non-urgent facility bookings, feedback, minor bulb replacements in common corridors.
* **Autonomous Action:**
  1. Standard ticket generated.
  2. Added to daytime maintenance roster.

---

## 3. Contractor Notification Payloads

### SMS Dispatch Template (Emergency)
```text
🚨 [URGENT DISPATCH - {condo_name}]
Ticket: {ticket_id} | Unit: {unit_number}
Fault: {issue_summary}
Priority: EMERGENCY (SLA: {sla_minutes} mins)
Resident: {contact_number}
Access: Pre-cleared at Main Guardhouse.
Reply 'CONFIRM' with your ETA.
```

### Resident Safety Advice Matrix
* **Pipe Burst:** *"Turn off the blue/red water valve inside your kitchen riser immediately to prevent floor and cabinet damage."*
* **Electrical Spark:** *"Do not touch the switch or appliance. Switch off the main breaker at your distribution board if safe."*
* **Stuck Lift:** *"Do not attempt to pry open the doors. Emergency ventilation is active; elevator technician has been auto-dispatched."*
