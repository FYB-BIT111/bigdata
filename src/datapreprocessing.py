"""清洗 Palo Alto 官方 CSV，合并 Hamilton #1、#2，按当地接入日汇总。"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = PROJECT_ROOT / "data" / "raw" / "ChargePoint_Data_CY20Q4_full.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "clean"
STATIONS = ("PALO ALTO CA / HAMILTON #1", "PALO ALTO CA / HAMILTON #2")
LOCAL_TIMEZONE = "America/Los_Angeles"
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def parse_times(values, zones, allow_excel_serial=False):
    wall = pd.to_datetime(values, format="%m/%d/%Y %H:%M", errors="coerce")
    serial_mask = pd.Series(False, index=values.index)
    if allow_excel_serial:
        serial = pd.to_numeric(values, errors="coerce")
        serial_mask = wall.isna() & serial.notna()
        wall.loc[serial_mask] = pd.to_datetime(
            serial.loc[serial_mask].to_numpy(dtype=float), unit="D", origin="1899-12-30"
        )
    offsets = zones.map({"PST": -8, "PDT": -7, "UTC": 0})
    utc = wall.dt.tz_localize("UTC") - pd.to_timedelta(offsets, unit="h")
    return utc.dt.tz_convert(LOCAL_TIMEZONE), serial_mask


def clean_sessions(raw):
    required = {
        "Station Name", "MAC Address", "Port Number", "Plug In Event Id",
        "Start Date", "Start Time Zone", "End Date", "End Time Zone", "Energy (kWh)",
        "Total Duration (hh:mm:ss)", "Charging Time (hh:mm:ss)",
    }
    if required - set(raw):
        raise ValueError("缺少必要字段：" + ", ".join(sorted(required - set(raw))))
    raw_columns = list(raw.columns)
    frame = raw.loc[raw["Station Name"].isin(STATIONS)].copy()
    if frame.empty:
        raise ValueError("没有 Hamilton #1、#2 的记录。")
    frame.insert(0, "source_rows", (frame.index + 1).astype(str))
    selected_count = len(frame)
    null_counts = {key: int(value) for key, value in frame[raw_columns].isna().sum().items()}
    exact_duplicates = frame.duplicated(raw_columns, keep="first")
    exact_duplicate_rows = frame.loc[exact_duplicates, "source_rows"].tolist()
    frame = frame.loc[~exact_duplicates].copy()
    start_local, _ = parse_times(frame["Start Date"], frame["Start Time Zone"])
    mac = frame["MAC Address"].str.replace(":", "", regex=False).str.upper().str.strip()
    keys = pd.DataFrame({
        "mac": mac,
        "port": frame["Port Number"].str.strip(),
        "event": frame["Plug In Event Id"].str.strip(),
        "start": start_local.map(lambda value: value.isoformat() if pd.notna(value) else None),
    }, index=frame.index)
    invalid = start_local.isna() | keys.isna().any(axis=1) | keys.eq("").any(axis=1)
    invalid_records = frame.loc[invalid, ["source_rows", "Station Name", "Start Date"]].to_dict("records")
    frame = frame.loc[~invalid].copy()
    keys = keys.loc[frame.index]
    frame["MAC Address"] = keys["mac"]
    frame["sessionID"] = keys.apply(
        lambda row: "hamilton_" + hashlib.sha256("\x1f".join(row.tolist()).encode("utf-8")).hexdigest(),
        axis=1,
    )
    conflicts = frame.loc[frame["sessionID"].duplicated(keep=False)].copy()
    conflict_details = []
    remove_indices = []
    for session_id, group in conflicts.groupby("sessionID", sort=False):
        first = group.index[0]
        different = [column for column in raw_columns
                     if group[column].fillna("<NULL>").nunique() > 1]
        frame.at[first, "source_rows"] = ";".join(group["source_rows"])
        # 冲突字段不任意选择一条；按已确认规则将会话电量置空。
        # 标识已确认开始时刻相同；PST/PDT/UTC 的等价表达保留首条表示。
        for column in (set(different) - {"Start Date", "Start Time Zone"}) | {"Energy (kWh)"}:
            frame.at[first, column] = pd.NA
        conflict_details.append({
            "sessionID": session_id,
            "source_rows": group["source_rows"].tolist(),
            "different_fields": different,
        })
        remove_indices.extend(group.index[1:].tolist())
    frame = frame.drop(index=remove_indices).copy()
    if frame.empty:
        raise ValueError("没有可以保留的 Hamilton 会话。")
    start_local, _ = parse_times(frame["Start Date"], frame["Start Time Zone"])
    end_local, end_serial_mask = parse_times(frame["End Date"], frame["End Time Zone"], True)
    frame["connection_time_local"] = start_local.map(lambda value: value.isoformat())
    frame["connection_date"] = start_local.dt.strftime("%Y-%m-%d")
    frame["disconnect_time_local"] = end_local.map(
        lambda value: value.isoformat() if pd.notna(value) else None
    )
    frame["site"] = "Hamilton"
    flags = pd.Series("", index=frame.index, dtype="string")
    flag_counts = {}

    def flag(mask, label):
        mask = mask.fillna(False)
        flag_counts[label] = int(mask.sum())
        flags.loc[mask] = flags.loc[mask] + label + ";"

    conflict_ids = set(conflicts["sessionID"])
    conflict_mask = frame["sessionID"].isin(conflict_ids)
    expected_start_zone = start_local.map(lambda value: value.strftime("%Z"))
    expected_end_zone = end_local.map(lambda value: value.strftime("%Z") if pd.notna(value) else None)
    flag(conflict_mask, "conflicting_session_records")
    flag(frame["Start Time Zone"].eq("UTC"), "start_time_was_utc")
    flag(frame["Start Time Zone"].isin(["PST", "PDT"]) & frame["Start Time Zone"].ne(expected_start_zone), "start_timezone_rule_mismatch")
    flag(frame["End Time Zone"].isin(["PST", "PDT"]) & frame["End Time Zone"].ne(expected_end_zone) & end_local.notna(), "end_timezone_rule_mismatch")
    flag(end_serial_mask, "end_time_excel_serial")
    flag(end_local.isna(), "end_time_missing_or_invalid")
    flag(end_local < start_local, "end_before_start")
    total_duration = pd.to_timedelta(frame["Total Duration (hh:mm:ss)"].fillna("").astype(str), errors="coerce")
    charging_duration = pd.to_timedelta(frame["Charging Time (hh:mm:ss)"].fillna("").astype(str), errors="coerce")
    flag(total_duration.isna(), "total_duration_missing_or_invalid")
    flag(charging_duration.isna(), "charging_duration_missing_or_invalid")
    flag(charging_duration > total_duration, "charging_duration_exceeds_total")
    flag((end_local - start_local - total_duration).abs() > pd.Timedelta(minutes=2), "duration_difference_over_2min")
    energy = pd.to_numeric(frame["Energy (kWh)"], errors="coerce").astype(float)
    energy_invalid = ~np.isfinite(energy) | energy.lt(0)
    flag(energy_invalid & ~conflict_mask, "energy_missing_or_invalid")
    frame["kWhDelivered"] = energy.mask(energy_invalid)
    frame["quality_flags"] = flags.str.rstrip(";")
    frame = frame.loc[start_local.sort_values(kind="stable").index].reset_index(drop=True)
    log = {
        "source_records": len(raw),
        "non_hamilton_records_excluded": len(raw) - selected_count,
        "selected_hamilton_records": selected_count,
        "original_null_counts": null_counts,
        "exact_duplicate_extra_rows_removed": len(exact_duplicate_rows),
        "exact_duplicate_source_rows": exact_duplicate_rows,
        "invalid_required_records_excluded": len(invalid_records),
        "invalid_required_record_details": invalid_records,
        "conflict_groups_merged": len(conflict_details),
        "conflicting_extra_rows_merged": len(remove_indices),
        "conflict_details": conflict_details,
        "retained_unique_sessions": len(frame),
        "quality_flag_counts": flag_counts,
        "local_connection_range": [start_local.min().isoformat(), start_local.max().isoformat()],
    }
    conflict_export = raw.loc[conflicts.index].copy()
    conflict_export.insert(0, "source_rows", conflicts["source_rows"])
    conflict_export.insert(1, "sessionID", conflicts["sessionID"])
    return frame, conflict_export, log


def summarize_daily(sessions):
    daily = sessions.groupby("connection_date").agg(
        daily_sessions=("sessionID", "nunique"),
        daily_energy_kwh=("kWhDelivered", lambda values: values.sum(min_count=len(values))),
        sessions_with_unknown_energy=("kWhDelivered", lambda values: int(values.isna().sum())),
        sessions_with_quality_flags=("quality_flags", lambda values: int(values.ne("").sum())),
    )
    calendar = pd.date_range(daily.index.min(), daily.index.max())
    daily = daily.reindex(calendar.strftime("%Y-%m-%d"))
    daily.index.name = "date"
    daily.insert(0, "weekday", [WEEKDAYS[value] for value in calendar.dayofweek])
    daily["day_status"] = "observed"
    daily.loc[daily["daily_sessions"].isna(), "day_status"] = "no_records_unverified"
    daily["is_boundary_date"] = daily.index.isin([daily.index.min(), daily.index.max()])
    for column in ("daily_sessions", "sessions_with_unknown_energy", "sessions_with_quality_flags"):
        daily[column] = daily[column].astype("Int64")
    return daily.reset_index()


def main():
    paths = {
        "sessions": OUTPUT_DIR / "hamilton_sessions_clean.csv",
        "daily": OUTPUT_DIR / "hamilton_daily_summary.csv",
        "conflicts": OUTPUT_DIR / "hamilton_conflicting_records.csv",
        "log": OUTPUT_DIR / "hamilton_preprocessing_log.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("输出已存在，停止以避免覆盖：" + ", ".join(existing))
    source_hash = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    raw = pd.read_csv(SOURCE_PATH, encoding="utf-8-sig", dtype="string")
    sessions, conflicts, processing = clean_sessions(raw)
    daily = summarize_daily(sessions)
    assert sessions["sessionID"].is_unique
    assert int(daily["daily_sessions"].sum()) == len(sessions)
    assert processing["selected_hamilton_records"] == (
        len(sessions) + processing["exact_duplicate_extra_rows_removed"]
        + processing["invalid_required_records_excluded"] + processing["conflicting_extra_rows_merged"]
    )
    assert hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() == source_hash, "原文件在处理过程中发生变化"
    log = {
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(SOURCE_PATH), "bytes": SOURCE_PATH.stat().st_size, "sha256": source_hash},
        "scope": {"site": "Hamilton", "address": "250 Hamilton Ave", "stations": list(STATIONS), "timezone": LOCAL_TIMEZONE, "year_filter": None},
        "definitions": {
            "daily_sessions": "按当地开始日期统计去重后的接入会话数。",
            "daily_energy_kwh": "按接入日归属的会话总电量，包含跨日交付；任一会话电量未知则该日总量留空。",
            "weekday": "Monday 至 Sunday。",
            "timestamps": "先按原字段 PST/PDT/UTC 解释，再转换到 America/Los_Angeles；与当地时区规则冲突时保留标记。",
            "excel_end_dates": "数值形式结束时间按 Excel 1899-12-30 日期原点解析。",
            "session_identity": "规范化 MAC、端口、事件编号、开始时刻组成标识；完全重复去重。",
            "conflicts": "按用户确认每组计一次；不一致字段和会话电量留空，原记录保留于冲突明细。",
            "no_record_days": "会话数和电量留空，不当作零；范围为 Hamilton 自身首末接入日。",
            "boundary_dates": "Hamilton 首末日期标记为边界，全天完整性待核查。",
            "source_rows": "原始 CSV 数据行位置，从 1 开始，不含表头；合并记录用分号分隔。",
        },
        "processing": processing,
        "summary": {
            "start_date": daily["date"].iloc[0], "end_date": daily["date"].iloc[-1],
            "calendar_days": len(daily),
            "observed_days": int(daily["day_status"].eq("observed").sum()),
            "no_record_dates": daily.loc[daily["day_status"].eq("no_records_unverified"), "date"].tolist(),
            "observed_dates_with_unknown_energy": daily.loc[daily["day_status"].eq("observed") & daily["daily_energy_kwh"].isna(), "date"].tolist(),
            "total_sessions": len(sessions),
            "sum_of_known_session_energy_kwh": float(sessions["kWhDelivered"].sum()),
            "energy_total_complete": bool(sessions["kWhDelivered"].notna().all()),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for frame, path in ((sessions, paths["sessions"]), (daily, paths["daily"]), (conflicts, paths["conflicts"])):
        frame.to_csv(path, index=False, encoding="utf-8-sig", mode="x")
    with paths["log"].open("x", encoding="utf-8") as handle:
        json.dump(log, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print("Selected source records:", processing["selected_hamilton_records"])
    print("Unique retained sessions:", len(sessions), "Merged conflict groups:", processing["conflict_groups_merged"])
    print("Calendar days:", len(daily), "No-record days:", len(log["summary"]["no_record_dates"]))
    print("Observed dates with unknown energy:", log["summary"]["observed_dates_with_unknown_energy"])
    for path in paths.values():
        print("Saved:", path)


if __name__ == "__main__":
    main()
