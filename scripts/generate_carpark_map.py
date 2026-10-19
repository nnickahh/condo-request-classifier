"""
Generate an architectural schematic map of the condominium basement carparks (B1 & B2).
Used by the Telegram bot when a resident mentions carpark without specifying the level or lot.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_carpark_map(output_path: Path) -> Path:
    width = 1200
    height = 800
    image = Image.new("RGB", (width, height), color="#0f172a")  # Slate 900 background
    draw = ImageDraw.Draw(image)

    # Try default fonts
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 28)
        subtitle_font = ImageFont.truetype("arial.ttf", 16)
        section_font = ImageFont.truetype("arialbd.ttf", 22)
        body_bold = ImageFont.truetype("arialbd.ttf", 15)
        body_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        title_font = subtitle_font = section_font = body_bold = body_font = small_font = ImageFont.load_default()

    # Header bar
    draw.rectangle([0, 0, width, 80], fill="#1e293b")
    draw.text((40, 16), "THE VERDANT RESIDENCES · BASEMENT CARPARK DIRECTORY", fill="#38bdf8", font=title_font)
    draw.text((40, 50), "Estate Layout Guide: Basement 1 (B1) & Basement 2 (B2) | Please state Level & Lot / Pillar Number", fill="#94a3b8", font=subtitle_font)

    # Two Main Cards: Left for B1, Right for B2
    card_y = 105
    card_h = 660
    card_w = 540

    # ==================== B1 CARD ====================
    b1_x = 40
    draw.rounded_rectangle([b1_x, card_y, b1_x + card_w, card_y + card_h], radius=16, fill="#1e293b", outline="#334155", width=2)
    # B1 Header badge
    draw.rounded_rectangle([b1_x + 20, card_y + 20, b1_x + card_w - 20, card_y + 70], radius=8, fill="#0284c7")
    draw.text((b1_x + 36, card_y + 30), "BASEMENT 1 (B1) — VISITOR & RESIDENT", fill="#ffffff", font=section_font)

    # B1 Content Areas
    # Area 1: Entrance Ramp & Guardhouse Access
    draw.rounded_rectangle([b1_x + 25, card_y + 90, b1_x + card_w - 25, card_y + 145], radius=8, fill="#0f172a", outline="#38bdf8", width=1)
    draw.text((b1_x + 40, card_y + 102), "MAIN VEHICLE RAMP (Entrance & Exit)", fill="#38bdf8", font=body_bold)
    draw.text((b1_x + 40, card_y + 122), "Direct ramp leading up to Main Guardhouse / Drop-off", fill="#94a3b8", font=body_font)

    # Area 2: Visitor Lots & EV Charging
    draw.rounded_rectangle([b1_x + 25, card_y + 160, b1_x + card_w - 25, card_y + 240], radius=8, fill="#0f172a", outline="#475569", width=1)
    draw.text((b1_x + 40, card_y + 172), "VISITOR PARKING & EV CHARGING (B1)", fill="#f59e0b", font=body_bold)
    draw.text((b1_x + 40, card_y + 194), "• Visitor Lots: V01 to V20 (Near Guardhouse Ramp)", fill="#cbd5e1", font=body_font)
    draw.text((b1_x + 40, card_y + 214), "• 4x Fast EV Charging Bays (Lots EV-01 to EV-04)", fill="#a3e635", font=body_font)

    # Area 3: Resident Lots
    draw.rounded_rectangle([b1_x + 25, card_y + 255, b1_x + card_w - 25, card_y + 335], radius=8, fill="#0f172a", outline="#475569", width=1)
    draw.text((b1_x + 40, card_y + 267), "RESIDENT PARKING LOTS (B1)", fill="#e2e8f0", font=body_bold)
    draw.text((b1_x + 40, card_y + 289), "• Lots #01 to #80 (Season Parking Holders)", fill="#cbd5e1", font=body_font)
    draw.text((b1_x + 40, card_y + 309), "• Pillars: A1 - A6, B1 - B6, C1 - C6", fill="#94a3b8", font=body_font)

    # Area 4: Lift Lobbies
    draw.rounded_rectangle([b1_x + 25, card_y + 350, b1_x + card_w - 25, card_y + 510], radius=8, fill="#0f172a", outline="#475569", width=1)
    draw.text((b1_x + 40, card_y + 362), "LIFT LOBBIES & RESIDENTIAL TOWERS (B1 Access)", fill="#60a5fa", font=body_bold)
    
    # Tower badges
    lobby_box_y = card_y + 395
    towers = [
        ("TOWER A", "Lobby A (Near Lots 01-25)", "#3b82f6"),
        ("TOWER B", "Lobby B (Near Lots 26-55)", "#10b981"),
        ("TOWER C", "Lobby C (Near Lots 56-80)", "#f59e0b"),
    ]
    for i, (t_name, t_desc, color) in enumerate(towers):
        ty = lobby_box_y + (i * 36)
        draw.rounded_rectangle([b1_x + 40, ty, b1_x + 130, ty + 28], radius=6, fill=color)
        draw.text((b1_x + 48, ty + 5), t_name, fill="#ffffff", font=body_bold)
        draw.text((b1_x + 145, ty + 5), t_desc, fill="#e2e8f0", font=body_font)

    # Tip in B1
    draw.rounded_rectangle([b1_x + 25, card_y + 525, b1_x + card_w - 25, card_y + 635], radius=8, fill="#0c4a6e", outline="#0284c7")
    draw.text((b1_x + 40, card_y + 540), "IDENTIFYING YOUR LOCATION ON B1:", fill="#38bdf8", font=body_bold)
    draw.text((b1_x + 40, card_y + 565), "Check the nearest yellow pillar letter/number (e.g. Pillar B3)", fill="#e0f2fe", font=body_font)
    draw.text((b1_x + 40, card_y + 585), "or note the numbered parking lot stencil on the ground (e.g. Lot 42).", fill="#e0f2fe", font=body_font)
    draw.text((b1_x + 40, card_y + 608), "Example reply: 'Basement 1 near Pillar B3' or 'B1 Lot 28'", fill="#7dd3fc", font=small_font)


    # ==================== B2 CARD ====================
    b2_x = 620
    draw.rounded_rectangle([b2_x, card_y, b2_x + card_w, card_y + card_h], radius=16, fill="#1e293b", outline="#334155", width=2)
    # B2 Header badge
    draw.rounded_rectangle([b2_x + 20, card_y + 20, b2_x + card_w - 20, card_y + 70], radius=8, fill="#6366f1")
    draw.text((b2_x + 36, card_y + 30), "BASEMENT 2 (B2) — RESIDENT SEASON ONLY", fill="#ffffff", font=section_font)

    # B2 Content Areas
    # Area 1: Internal Connecting Ramp
    draw.rounded_rectangle([b2_x + 25, card_y + 90, b2_x + card_w - 25, card_y + 145], radius=8, fill="#0f172a", outline="#818cf8", width=1)
    draw.text((b2_x + 40, card_y + 102), "INTERNAL VEHICLE RAMP TO B1", fill="#a5b4fc", font=body_bold)
    draw.text((b2_x + 40, card_y + 122), "Direct ramp connection between Basement 1 and Basement 2", fill="#94a3b8", font=body_font)

    # Area 2: Resident Season Lots
    draw.rounded_rectangle([b2_x + 25, card_y + 160, b2_x + card_w - 25, card_y + 240], radius=8, fill="#0f172a", outline="#475569", width=1)
    draw.text((b2_x + 40, card_y + 172), "RESIDENT PARKING LOTS (B2)", fill="#e2e8f0", font=body_bold)
    draw.text((b2_x + 40, card_y + 194), "• Lots #81 to #180 (Designated Resident Season Parking)", fill="#cbd5e1", font=body_font)
    draw.text((b2_x + 40, card_y + 214), "• Pillars: AA1 - AA6, BB1 - BB6, CC1 - CC6", fill="#94a3b8", font=body_font)

    # Area 3: Facilities (Car Wash & Bicycle Bay)
    draw.rounded_rectangle([b2_x + 25, card_y + 255, b2_x + card_w - 25, card_y + 335], radius=8, fill="#0f172a", outline="#475569", width=1)
    draw.text((b2_x + 40, card_y + 267), "SPECIAL AMENITIES (B2)", fill="#38bdf8", font=body_bold)
    draw.text((b2_x + 40, card_y + 289), "• Car Wash Bays: 2x Dedicated water wash points (Near CC2)", fill="#cbd5e1", font=body_font)
    draw.text((b2_x + 40, card_y + 309), "• Residents Bicycle Parking Shelters (Near Lobby A)", fill="#94a3b8", font=body_font)

    # Area 4: Lift Lobbies
    draw.rounded_rectangle([b2_x + 25, card_y + 350, b2_x + card_w - 25, card_y + 510], radius=8, fill="#0f172a", outline="#475569", width=1)
    draw.text((b2_x + 40, card_y + 362), "LIFT LOBBIES & RESIDENTIAL TOWERS (B2 Access)", fill="#818cf8", font=body_bold)
    
    # Tower badges B2
    towers_b2 = [
        ("TOWER A", "Lobby A (Near Lots 81-110)", "#3b82f6"),
        ("TOWER B", "Lobby B (Near Lots 111-145)", "#10b981"),
        ("TOWER C", "Lobby C (Near Lots 146-180)", "#f59e0b"),
    ]
    for i, (t_name, t_desc, color) in enumerate(towers_b2):
        ty = lobby_box_y + (i * 36)
        draw.rounded_rectangle([b2_x + 40, ty, b2_x + 130, ty + 28], radius=6, fill=color)
        draw.text((b2_x + 48, ty + 5), t_name, fill="#ffffff", font=body_bold)
        draw.text((b2_x + 145, ty + 5), t_desc, fill="#e2e8f0", font=body_font)

    # Tip in B2
    draw.rounded_rectangle([b2_x + 25, card_y + 525, b2_x + card_w - 25, card_y + 635], radius=8, fill="#312e81", outline="#6366f1")
    draw.text((b2_x + 40, card_y + 540), "IDENTIFYING YOUR LOCATION ON B2:", fill="#a5b4fc", font=body_bold)
    draw.text((b2_x + 40, card_y + 565), "Check the double-letter pillar code (e.g. Pillar BB4 or CC1)", fill="#e0e7ff", font=body_font)
    draw.text((b2_x + 40, card_y + 585), "or note the parking lot stencil (e.g. Lot 125 near Tower B).", fill="#e0e7ff", font=body_font)
    draw.text((b2_x + 40, card_y + 608), "Example reply: 'Basement 2 Lot 118' or 'B2 Car Wash Bay'", fill="#c7d2fe", font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "JPEG", quality=92)
    return output_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "output" / "estate_carpark_map.jpg"
    res = create_carpark_map(out)
    print(f"Generated carpark map at: {res}")
