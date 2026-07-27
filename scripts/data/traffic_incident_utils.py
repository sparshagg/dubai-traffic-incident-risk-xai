from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = ROOT / "traffic_incidents_2026-06-01_10-44-37_1.csv"
AUDIT_DIR = ROOT / "data" / "audit"
MAPPINGS_DIR = ROOT / "data" / "mappings"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_COLUMNS = ["acci_id", "acci_time", "acci_name", "acci_x", "acci_y", "load_timestamp"]
SEVERITY_BY_AR = {
    "بسيط": ("minor", "minor", 1),
    "متوسط": ("moderate", "moderate", 2),
    "بليغ": ("severe", "severe", 3),
}
KNOWN_SEVERITIES = set(SEVERITY_BY_AR)


@dataclass(frozen=True)
class TypeTranslation:
    incident_type_en: str
    incident_type_code: str
    review_status: str = "translated_reviewed"
    notes: str = ""
    include_in_eda: bool = True
    exclude_reason: str = ""


@dataclass(frozen=True)
class NormalizedCoordinate:
    longitude: str
    latitude: str
    coordinate_status: str


def ensure_dirs() -> None:
    for path in [AUDIT_DIR, MAPPINGS_DIR, PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_acci_name(value: str) -> str:
    text = (value or "").strip()
    text = text.replace("ـ", "")
    text = text.replace("\u200f", "").replace("\u200e", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*[-–—]\s*", " - ", text)
    return text.strip()


def strip_redundant_prefix(type_ar: str) -> str:
    text = normalize_acci_name(type_ar)
    while text.startswith("حادث "):
        text = text[len("حادث ") :].strip()
    return text


def split_incident_name(acci_name_ar: str) -> tuple[str, str, str]:
    normalized = normalize_acci_name(acci_name_ar)
    parts = [p.strip() for p in normalized.rsplit(" - ", 1)]
    if len(parts) == 2 and parts[1] in KNOWN_SEVERITIES:
        return strip_redundant_prefix(parts[0]), parts[1], normalized

    # A few source labels encode "minor" as part of the phrase without a hyphen.
    if normalized.endswith(" بسيط"):
        without_suffix = normalized[: -len(" بسيط")].strip()
        if "حادث مروري" in normalized:
            return strip_redundant_prefix(without_suffix), "بسيط", normalized

    return strip_redundant_prefix(normalized), "", normalized


def severity_fields(severity_ar: str) -> tuple[str, str, int]:
    if severity_ar in SEVERITY_BY_AR:
        return SEVERITY_BY_AR[severity_ar]
    return "not specified", "unknown", 0


def looks_garbled(text: str) -> bool:
    known_terms = [
        "مركبة",
        "مركبه",
        "صدم",
        "اصطدام",
        "تعطل",
        "تدهور",
        "دهس",
        "حريق",
        "سقوط",
        "وجود",
        "ازدحام",
        "حادث",
        "حافلة",
        "باص",
        "شاحنة",
        "دراجة",
        "مروري",
        "الوقوف",
        "التسابق",
        "الاستعراضات",
        "حيوانات",
        "قيادة",
        "عبور",
        "دخول",
        "مسيرات",
        "إصابة",
        "توقف",
        "انفصال",
    ]
    return not any(term in text for term in known_terms)


def make_translation(
    en: str,
    code: str,
    status: str = "translated_reviewed",
    notes: str = "",
    include_in_eda: bool = True,
    exclude_reason: str = "",
) -> TypeTranslation:
    return TypeTranslation(en, code, status, notes, include_in_eda, exclude_reason)


def in_dubai_lon_lat(longitude: float, latitude: float) -> bool:
    return 54.5 <= longitude <= 56.5 and 24.5 <= latitude <= 26.5


def normalize_coordinates(acci_x: str, acci_y: str) -> NormalizedCoordinate:
    x_raw = (acci_x or "").strip()
    y_raw = (acci_y or "").strip()
    if not x_raw or not y_raw:
        return NormalizedCoordinate("", "", "missing_coordinate")

    try:
        x = float(x_raw)
        y = float(y_raw)
    except ValueError:
        return NormalizedCoordinate("", "", "invalid_coordinate")

    if x == 0 or y == 0:
        return NormalizedCoordinate("", "", "zero_coordinate")
    if in_dubai_lon_lat(x, y):
        return NormalizedCoordinate(f"{x:.8f}", f"{y:.8f}", "as_provided_lon_lat")
    if in_dubai_lon_lat(y, x):
        return NormalizedCoordinate(f"{y:.8f}", f"{x:.8f}", "swapped_lat_lon")
    return NormalizedCoordinate("", "", "out_of_bounds")


def translate_incident_type(type_ar: str) -> TypeTranslation:
    text = strip_redundant_prefix(type_ar)
    compact = text.replace(" ", "")

    if not text:
        return make_translation("", "unknown", "reviewed_excluded", "Missing incident type.", False, "missing_incident_type")
    if "تم الغاء نوع الحدث" in text:
        return make_translation(
            "cancelled event type",
            "cancelled_event_type",
            "reviewed_excluded",
            "Administrative/cancelled source category; excluded from category-level EDA.",
            False,
            "cancelled_event_type",
        )
    if text == "مروري":
        return make_translation("generic traffic event", "generic_traffic_event", "reviewed_conservative", "Very broad source category; retained as generic traffic event.")
    if "حوادث مرورية اخرى" in text:
        return make_translation("other traffic incident", "other_traffic_incident")

    if "الوقوف خلف المركبات" in text:
        return make_translation("double parking behind vehicles", "double_parking")
    if "مركبات مخالفة" in text:
        return make_translation("violating vehicles", "vehicle_violation")
    if "قيادة مركبة بدون رخصة" in text:
        return make_translation("driving without a license", "driving_without_license")
    if "الاستعراضات والتفحيط" in text:
        return make_translation("stunt driving and drifting", "stunt_driving_drifting")
    if "التسابق في الشارع العام" in text:
        return make_translation("street racing", "street_racing")
    if "مسيرات احتفالية غير مرخصة" in text:
        return make_translation("unauthorized celebratory parade", "unauthorized_celebratory_parade")
    if "طلب ثبات دورية" in text:
        return make_translation("request for patrol presence at roadwork site", "patrol_presence_roadwork_site")

    if "ازدحام في المنافذ الحدودية" in text:
        return make_translation("border crossing congestion", "border_crossing_congestion")
    if "ازدحام مروري" in text:
        return make_translation("traffic congestion", "traffic_congestion")

    if "تعطل اشارة ضوئية" in text:
        return make_translation("traffic signal malfunction", "traffic_signal_malfunction")
    if "تعطل مثبت السرعة" in text:
        return make_translation("cruise control malfunction", "cruise_control_malfunction")

    if "تعطل مركبة نقل موقوفين" in text or "تعطل مركبة نقل مساجين" in text:
        return make_translation("detainee transport vehicle breakdown", "detainee_transport_vehicle_breakdown")
    if "تعطل مركبة نقل أموال" in text:
        return make_translation("cash transport vehicle breakdown", "cash_transport_vehicle_breakdown")
    if "تعطل مركبة عسكرية" in text:
        return make_translation("military vehicle breakdown", "military_vehicle_breakdown")
    if "تعطل مركبة ثقيلة" in text:
        return make_translation("heavy vehicle breakdown", "heavy_vehicle_breakdown")
    if "تعطل مركبة خفيفة" in text:
        return make_translation("light vehicle breakdown", "light_vehicle_breakdown")
    if "تعطل مركبة على طريق عام" in text:
        return make_translation("vehicle breakdown on public road", "vehicle_breakdown_public_road")
    if "تعطل مركبة نتيجة حادث مروري" in text:
        return make_translation("vehicle breakdown due to traffic accident", "vehicle_breakdown_due_to_accident")
    if "مركبه عطلانه في الشارع" in text or "مركبة عطلانة في الشارع" in text:
        return make_translation("disabled vehicle on road", "disabled_vehicle_on_road")

    if "ضد مجهول" in text:
        return make_translation("incident involving unknown party", "unknown_party_incident", "translated_reviewed", "Dubai Police category; verify exact operational meaning if used in final report.")
    if "الصدم والهروب" in text or "صدم و هروب" in text or "صدم وهروب" in text:
        return make_translation("hit and run", "hit_and_run")
    if "دهس وهروب" in text:
        return make_translation("pedestrian hit and run", "pedestrian_hit_and_run")

    if "دهس طفل" in text:
        return make_translation("pedestrian collision involving child", "pedestrian_collision_child")
    if "دهس امراة" in text or "دهس امرأة" in text:
        return make_translation("pedestrian collision involving woman", "pedestrian_collision_woman")
    if "دهس رجل" in text:
        return make_translation("pedestrian collision involving man", "pedestrian_collision_man")
    if text == "دهس":
        return make_translation("pedestrian collision", "pedestrian_collision")

    if "اصطدام بين حافلة نقل عمال ومركبة" in text:
        return make_translation("worker bus and vehicle collision", "worker_bus_vehicle_collision")
    if "اصطدام بين حافلة مدرسية ومركبة" in text:
        return make_translation("school bus and vehicle collision", "school_bus_vehicle_collision")
    if "اصطدام بين شاحنة ومركبة" in text:
        return make_translation("truck and vehicle collision", "truck_vehicle_collision")
    if "اصطدام بين شاحنتين" in text:
        return make_translation("collision between two trucks", "two_truck_collision")
    if "اصطدام بين عدة" in text:
        return make_translation("multi-vehicle collision", "multi_vehicle_collision")
    if "اصطدام بين سيارتين" in text or "اصطدام بين مركبتين" in text:
        return make_translation("two-vehicle collision", "two_vehicle_collision")

    if "صدم صهريج مواد كيماوية" in text:
        return make_translation("hit chemical tanker", "hit_chemical_tanker")
    if "صدم صهريج محروقات" in text:
        return make_translation("hit fuel tanker", "hit_fuel_tanker")
    if "صدم جسر من شاحنة عالية الارتفاع" in text:
        return make_translation("bridge strike by over-height truck", "bridge_strike_overheight_truck")
    if "صدم مركبة نقل أموال" in text:
        return make_translation("hit cash transport vehicle", "hit_cash_transport_vehicle")
    if "صدم مركبة عسكرية" in text:
        return make_translation("hit military vehicle", "hit_military_vehicle")
    if "صدم دراجة نارية" in text:
        return make_translation("hit motorcycle", "hit_motorcycle")
    if "صدم دراجة هوائية" in text:
        return make_translation("hit bicycle", "hit_bicycle")
    if text == "صدم دراجة":
        return make_translation(
            "hit two-wheeler, unspecified",
            "hit_two_wheeler_unspecified",
            "reviewed_conservative",
            "Arabic source says two-wheeler generically; retained without forcing bicycle vs motorcycle.",
        )
    if "صدم حيوان" in text:
        return make_translation("hit animal", "hit_animal")
    if "صدم جسم في الشارع" in text:
        return make_translation("hit object in road", "hit_object_in_road")
    if "صدم علامة مرورية" in text or "صدم اشارة مرورية" in text:
        return make_translation("hit traffic sign", "hit_traffic_sign")
    if "صدم لوحة إرشادية" in text:
        return make_translation("hit guide sign", "hit_guide_sign")
    if "صدم إشارة ضوئية" in text:
        return make_translation("hit traffic signal", "hit_traffic_signal")
    if "صدم ترام" in text:
        return make_translation("hit tram", "hit_tram")
    if "صدم قطار" in text:
        return make_translation("hit train", "hit_train")
    if "صدم حواجز" in text or "صدم حاجز" in text:
        return make_translation("hit barrier", "hit_barrier")
    if "صدم عمود" in text:
        return make_translation("hit pole", "hit_pole")
    if "صدم جدار" in text:
        return make_translation("hit wall", "hit_wall")
    if "صدم رصيف" in text:
        return make_translation("hit curb", "hit_curb")
    if "صدم شجرة" in text:
        return make_translation("hit tree", "hit_tree")
    if "صدم مبنى او بيوت او معرض" in text:
        return make_translation("hit building, house, or showroom", "hit_building_house_showroom")
    if "صدم مبنى" in text:
        return make_translation("hit building", "hit_building")
    if "صدم باب" in text:
        return make_translation("hit door", "hit_door")

    if "تدهور صهريج لمواد قابلة للاشتعال" in text:
        return make_translation("flammable materials tanker rollover", "flammable_tanker_rollover")
    if "تدهور حافلة مدرسية" in text or "تدهور باص مدرسة" in text:
        return make_translation("school bus rollover", "school_bus_rollover")
    if "تدهور حافلة عمال" in text or "تدهور باص عمال" in text:
        return make_translation("worker bus rollover", "worker_bus_rollover")
    if "تدهور مركبة ثقيلة" in text:
        return make_translation("heavy vehicle rollover", "heavy_vehicle_rollover")
    if "تدهور مركبة خفيفة" in text:
        return make_translation("light vehicle rollover", "light_vehicle_rollover")
    if "تدهور دراجة نارية" in text:
        return make_translation("motorcycle rollover", "motorcycle_rollover")
    if "تدهور دراجة هوائية" in text:
        return make_translation("bicycle rollover", "bicycle_rollover")
    if text == "تدهور دراجة":
        return make_translation(
            "two-wheeler rollover, unspecified",
            "two_wheeler_rollover_unspecified",
            "reviewed_conservative",
            "Arabic source says two-wheeler generically; retained without forcing bicycle vs motorcycle.",
        )

    if "حريق مركبة أثناء سيرها" in text:
        return make_translation("vehicle fire while moving", "vehicle_fire_while_moving")
    if "حريق في مركبة" in text:
        return make_translation("vehicle fire", "vehicle_fire")

    if "وجود بقعة زيت" in text:
        return make_translation("oil spill on road", "oil_spill_on_road")
    if "وجود حفرة في الطريق" in text:
        return make_translation("hole in road", "hole_in_road")
    if "وجود جسم في الشارع" in text:
        return make_translation("object in road", "object_in_road")
    if "حيوانات سائبة" in text:
        return make_translation("stray animals on public road", "stray_animals_on_road")
    if "عبور شخص" in text:
        return make_translation("pedestrian crossing at non-designated location", "unauthorized_pedestrian_crossing")

    if "تطاير أجسام على مركبة" in text or "سقوط وتطاير أجسام" in text:
        return make_translation("falling or flying objects hitting vehicle", "falling_flying_objects_hit_vehicle")
    if "سقوط حمولة" in text:
        return make_translation("load falling from moving vehicle", "load_falling_from_vehicle")
    if "سقوط شخص من مركبة" in text:
        return make_translation("person falling from moving vehicle", "person_falling_from_vehicle")
    if "سقوط مركبة من على جسر" in text:
        return make_translation("vehicle falling from bridge", "vehicle_falling_from_bridge")
    if "سقوط مركبة من ارتفاع" in text:
        return make_translation("vehicle falling from height", "vehicle_falling_from_height")
    if "سقوط مركبة في حفرة" in text:
        return make_translation("vehicle falling into pit", "vehicle_falling_into_pit")
    if "سقوط حاوية" in text:
        return make_translation("container fall", "container_fall")
    if "سقوط رافعة" in text:
        return make_translation("crane fall", "crane_fall")
    if "انفصال مقطورة" in text:
        return make_translation("trailer detached from moving vehicle", "trailer_detached_from_vehicle")

    if "دخول شاحنة في مكان ممنوع" in text:
        return make_translation("truck entering prohibited area", "truck_entering_prohibited_area")
    if "عكس اتجاه السير" in text:
        return make_translation("vehicle driving against traffic", "wrong_way_vehicle")
    if "إصابة راكب أثناء حركة المركبة" in text:
        return make_translation("passenger injury while vehicle is moving", "passenger_injury_moving_vehicle")
    if "توقف مركبة في شارع المغادرين" in text:
        return make_translation("vehicle stopped on departures road", "vehicle_stopped_departures_road")

    if looks_garbled(text):
        return make_translation(
            "unclear source category",
            "unclear_source_category",
            "reviewed_excluded",
            "Source Arabic appears garbled or non-meaningful; excluded from category-level EDA.",
            False,
            "garbled_source_category",
        )

    return make_translation(
        "unmapped incident type",
        "unmapped_incident_type",
        "reviewed_excluded",
        f"Unmapped source category excluded until manually reviewed: {text}",
        False,
        "unmapped_category",
    )


def read_csv_rows(path: Path) -> csv.DictReader:
    f = path.open("r", encoding="utf-8-sig", newline="")
    return csv.DictReader(f)
