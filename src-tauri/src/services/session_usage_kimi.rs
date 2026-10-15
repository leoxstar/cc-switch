//! Kimi Code CLI 会话日志使用追踪
//!
//! 从 `$KIMI_CODE_HOME/sessions/<workDirKey>/<sessionId>/agents/<agentId>/wire.jsonl`
//! 提取 `usage.record` 事件（每次模型调用的用量），实现无代理模式下的使用统计。
//!
//! ## 数据流
//! ```text
//! ~/.kimi-code/sessions/*/*/agents/*/wire.jsonl → 全量解析 → 费用计算 → proxy_request_logs 表
//! ```
//!
//! ## 与其它解析器的差异
//! - 一条 `usage.record` 就是一次模型调用，token 是独立值，不需要 delta 计算，
//!   也不依赖前一条记录的累计值。
//! - 记录里带 `model`（如 `kimi-code/k3`，命名空间前缀已剥掉）和 `time`（epoch 毫秒），
//!   但**没有**请求 id；request_id 由「会话 + agent + 时间 + token 指纹」合成
//!   （见 [`record_request_id`]）。文件被重写也不影响这个 id，所以重复扫描是幂等的。
//! - 子 agent 的用量写在自己的 `agents/<agentId>/wire.jsonl` 里，与主 agent 的
//!   `wire.jsonl` 不重叠（本机 88 个文件、699 条记录实测无重复指纹），因此每个
//!   agent 目录各记一条，不需要额外去重。
//! - 单次请求的耗时和首字来自同目录的 `logs/kimi-code.log`：每条 `llm response`
//!   行带 `ttftMs` / `streamDurationMs`，时间戳与 `usage.record.time` 是同一个时钟
//!   （实测 480 条全部毫秒级对齐）。所以这两列是**客户端实测**，不是按日志时间戳
//!   估的，前端/后端把它当精确口径（见 `MEASURED_TIMING_SESSION_SOURCES`）。
//!   日志会被轮转或清掉，取不到计时的行仍是 0 / NULL（界面显示「—」）。
//! - 旧家目录（迁移前的 `~/.kimi`）用的是 `StatusUpdate` 记录类型，不在扫描范围内：
//!   `get_kimi_dir()` 指向 `~/.kimi-code`。要补那批历史得单独写解析分支。

use crate::database::{lock_conn, Database};
use crate::error::AppError;
use crate::kimi_config::get_kimi_dir;
use crate::proxy::usage::calculator::{CostCalculator, ModelPricing};
use crate::proxy::usage::parser::TokenUsage;
use crate::services::session_usage::{
    load_sync_cursors, metadata_modified_nanos, update_sync_state, SessionSyncResult,
};
use crate::services::usage_stats::{find_model_pricing, should_skip_session_insert, DedupKey};
use rust_decimal::Decimal;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// 写入 `proxy_request_logs.app_type` 的值（前端用量页的应用筛选项）。
const APP_TYPE: &str = "kimi";
/// `provider_id` / `provider_type` / `data_source` 都用它：这不是代理来源，是会话导入。
const SOURCE: &str = "kimi_session";

/// 一条 `usage.record` 解析出的用量。
#[derive(Debug, Clone, PartialEq, Eq)]
struct KimiUsage {
    agent_id: String,
    model: String,
    input_tokens: u32,
    output_tokens: u32,
    cache_read_tokens: u32,
    cache_creation_tokens: u32,
    /// epoch 毫秒。合成 request_id 和 `created_at` 都用它。
    time_ms: i64,
}

/// 一条请求的实测耗时，取自会话目录里的 `logs/kimi-code.log`。
///
/// `first_token_ms` 是客户端看到首字的时刻（日志里的 `ttftMs`）；`latency_ms` 是
/// `ttftMs + streamDurationMs`，也就是 Kimi 自己的两段分解——两者相减正好是流式
/// 生成窗口，前端的速度列就是按这个窗口算的。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RequestTiming {
    first_token_ms: i64,
    latency_ms: i64,
}

/// 一个会话的实测耗时表，按响应时刻（epoch 毫秒）索引。
///
/// 同一毫秒出现多条响应时按顺序取用，取过的不会再给第二条记录（1:1 配对）。
#[derive(Debug, Default)]
struct SessionTimings {
    by_time: HashMap<i64, Vec<RequestTiming>>,
}

impl SessionTimings {
    /// 取与 `time_ms` 对齐的耗时；找不到（或已被取走）时返回 `None`。
    ///
    /// 容差 ±3ms：日志和 wire 的毫秒值实测一致，留一点余量防时钟取整差异。
    fn take(&mut self, time_ms: i64) -> Option<RequestTiming> {
        for offset in 0..=3i64 {
            for candidate in [time_ms + offset, time_ms - offset] {
                if let Some(bucket) = self.by_time.get_mut(&candidate) {
                    if !bucket.is_empty() {
                        return Some(bucket.remove(0));
                    }
                }
            }
        }
        None
    }
}

/// 解析 `<session>/logs/kimi-code.log`，收集每条 `llm response` 的实测耗时。
///
/// 行样例（字段顺序不做假设，按 `key=value` 找）：
/// ```text
/// 2026-10-13T03:11:58.493Z INFO  llm response  turnStep=0.1 ttftMs=3479 streamDurationMs=2078 ...
/// ```
fn parse_session_log(log_path: &Path) -> SessionTimings {
    let mut timings = SessionTimings::default();
    let Ok(content) = fs::read_to_string(log_path) else {
        return timings; // 日志被轮转掉是常态，不是错误
    };

    for line in content.lines() {
        let mut fields = line.split_whitespace();
        let Some(timestamp) = fields.next() else {
            continue;
        };
        if fields.next() != Some("INFO") || fields.next() != Some("llm") {
            continue;
        }
        if fields.next() != Some("response") {
            continue;
        }

        let mut ttft_ms = None;
        let mut stream_ms = None;
        for field in fields {
            if let Some(value) = field.strip_prefix("ttftMs=") {
                ttft_ms = value.parse::<i64>().ok();
            } else if let Some(value) = field.strip_prefix("streamDurationMs=") {
                stream_ms = value.parse::<i64>().ok();
            }
        }
        let (Some(ttft_ms), Some(stream_ms)) = (ttft_ms, stream_ms) else {
            continue;
        };
        let Some(time_ms) = parse_log_timestamp_ms(timestamp) else {
            continue;
        };

        timings
            .by_time
            .entry(time_ms)
            .or_default()
            .push(RequestTiming {
                first_token_ms: ttft_ms.max(0),
                latency_ms: ttft_ms.max(0) + stream_ms.max(0),
            });
    }

    timings
}

/// `2026-10-13T03:11:58.493Z` → epoch 毫秒。
fn parse_log_timestamp_ms(timestamp: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(timestamp)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

/// 同步 Kimi Code 使用数据。
pub fn sync_kimi_usage(db: &Database) -> Result<SessionSyncResult, AppError> {
    let files = collect_wire_files(&get_kimi_dir());
    let mut result = SessionSyncResult {
        imported: 0,
        skipped: 0,
        files_scanned: files.len() as u32,
        suspected_duplicates: 0,
        deferred_files: 0,
        errors: vec![],
    };

    if files.is_empty() {
        return Ok(result);
    }

    // 缺 model 的记录回落到当前配置的默认模型；每轮只读一次配置。
    let default_model = default_model();
    let cursors = load_sync_cursors(db)?;

    for file_path in &files {
        let last_modified = cursors
            .get(file_path.to_string_lossy().as_ref())
            .map_or(0, |c| c.last_modified);
        match sync_single_wire_file(db, file_path, last_modified, &default_model) {
            Ok((imported, skipped)) => {
                result.imported += imported;
                result.skipped += skipped;
            }
            Err(e) => {
                let msg = format!("Kimi 会话文件解析失败 {}: {e}", file_path.display());
                log::warn!("[KIMI-SYNC] {msg}");
                result.errors.push(msg);
            }
        }
    }

    if result.imported > 0 {
        log::info!(
            "[KIMI-SYNC] 同步完成: 导入 {} 条, 跳过 {} 条, 扫描 {} 个文件",
            result.imported,
            result.skipped,
            result.files_scanned
        );
    }

    Ok(result)
}

/// 收集 `sessions/<workDirKey>/<sessionId>/agents/<agentId>/wire.jsonl`。
///
/// 一级目录是工作目录键、二级是会话、三级是 `agents`。多一层少一层都跳过，
/// 不递归兜底——`sessions/` 下还有 `logs/`、`notify/` 之类的旁支，宽扫会把它们
/// 当会话读。
fn collect_wire_files(kimi_dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let root = kimi_dir.join("sessions");
    let Ok(workdirs) = fs::read_dir(&root) else {
        return files;
    };

    for workdir in workdirs.flatten() {
        let Ok(sessions) = fs::read_dir(workdir.path()) else {
            continue;
        };
        for session in sessions.flatten() {
            let agents_dir = session.path().join("agents");
            let Ok(agents) = fs::read_dir(&agents_dir) else {
                continue;
            };
            for agent in agents.flatten() {
                let wire = agent.path().join("wire.jsonl");
                if wire.is_file() {
                    files.push(wire);
                }
            }
        }
    }

    files.sort();
    files
}

/// 同步单个 wire.jsonl，返回 (imported, skipped)。
///
/// 文件没变就整份跳过；变了就全量重读，靠 request_id 去重。Kimi 是追加写的，
/// 全量重读意味着每步都要把已导入的记录再判一遍，但换来的是不用担心行偏移
/// 与文件重写（`state.json` 里带 `wire_mtime`，说明它自己也会重写 wire）。
fn sync_single_wire_file(
    db: &Database,
    file_path: &Path,
    last_modified: i64,
    default_model: &str,
) -> Result<(u32, u32), AppError> {
    let metadata = fs::metadata(file_path)
        .map_err(|e| AppError::Config(format!("无法读取文件元数据: {e}")))?;
    let file_modified = metadata_modified_nanos(&metadata);
    if file_modified <= last_modified {
        return Ok((0, 0));
    }

    let content = fs::read_to_string(file_path)
        .map_err(|e| AppError::Config(format!("无法读取文件: {e}")))?;

    let scope = FileScope::from_path(file_path);
    // 同一会话的日志里带每次请求的实测耗时；按响应时刻和 usage.record 对齐。
    let mut timings = parse_session_log(&scope.session_dir.join("logs").join("kimi-code.log"));
    let mut imported = 0u32;
    let mut skipped = 0u32;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue; // 空行是写一半的常见形态，跳过不算错
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue; // 截断的尾行等非法 JSON：跳过，下次扫描再取
        };
        let Some(usage) = parse_usage_record(&value, &scope.agent_id, default_model) else {
            continue;
        };

        let request_id = record_request_id(&scope, &usage);
        let timing = timings.take(usage.time_ms);
        match insert_kimi_usage(db, &request_id, &usage, &scope.session_id, timing) {
            Ok(true) => imported += 1,
            Ok(false) => skipped += 1,
            Err(e) => {
                log::warn!("[KIMI-SYNC] 插入失败 ({request_id}): {e}");
                skipped += 1;
            }
        }
    }

    update_sync_state(db, &file_path.to_string_lossy(), file_modified, 0)?;
    Ok((imported, skipped))
}

/// 从文件路径推出来的会话/agent 标识。
struct FileScope {
    session_id: String,
    agent_id: String,
    /// 会话目录（`logs/kimi-code.log` 在它下面）。
    session_dir: PathBuf,
    /// 同一会话内 request_id 的唯一性只需要 session + agent，目录键只是防串台。
    workdir_key: String,
}

impl FileScope {
    /// `.../sessions/<workDirKey>/<sessionId>/agents/<agentId>/wire.jsonl`
    fn from_path(path: &Path) -> Self {
        let agent_id = path
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "main".to_string());
        let session_dir = path
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let session_id = session_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let workdir_key = session_dir
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        Self {
            session_id,
            agent_id,
            session_dir,
            workdir_key,
        }
    }
}

/// 解析一条 `usage.record`；不是用量记录、token 全零、或没有时间戳时返回 `None`。
fn parse_usage_record(
    value: &Value,
    fallback_agent_id: &str,
    fallback_model: &str,
) -> Option<KimiUsage> {
    if value.get("type").and_then(Value::as_str) != Some("usage.record") {
        return None;
    }

    let usage = value.get("usage")?.as_object()?;
    let token = |key: &str| {
        usage
            .get(key)
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(u32::MAX as u64) as u32
    };

    let input_tokens = token("inputOther");
    let output_tokens = token("output");
    let cache_read_tokens = token("inputCacheRead");
    let cache_creation_tokens = token("inputCacheCreation");
    if input_tokens == 0
        && output_tokens == 0
        && cache_read_tokens == 0
        && cache_creation_tokens == 0
    {
        return None; // 空调用（重试被拦、请求失败）不该占一行
    }

    let time_ms = value
        .get("time")
        .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)))?;

    let agent_id = value
        .get("agentId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback_agent_id)
        .to_string();

    let model = value
        .get("model")
        .and_then(Value::as_str)
        .map(normalize_model)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| fallback_model.to_string());

    Some(KimiUsage {
        agent_id,
        model,
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_creation_tokens,
        time_ms,
    })
}

/// 剥掉 provider 命名空间：`kimi-code/k3` → `k3`。
///
/// 定价表按裸模型名索引（`k3` 有价、`kimi-code/k3` 没有），所以入库前统一剥一层；
/// 模型名里若还有第二段斜杠（如 `a/b/c`）原样保留后面全部。
fn normalize_model(raw: &str) -> String {
    let trimmed = raw.trim();
    match trimmed.split_once('/') {
        Some((_, rest)) => rest.trim().to_string(),
        None => trimmed.to_string(),
    }
}

/// 当前配置的默认模型（缺配置时为 `kimi-for-coding`，即 Kimi Code 的套餐模型名）。
///
/// 记录里没有 model 字段时才用它兜底；定价表里没有 `kimi-for-coding` 的行，
/// 这类记录只统计 token、不计费（与用量页的未定价模型口径一致）。
fn default_model() -> String {
    crate::kimi_config::get_default_model()
        .ok()
        .flatten()
        .map(|raw| normalize_model(&raw))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "kimi-for-coding".to_string())
}

/// 合成 request_id：会话 + agent + 时间 + token 指纹。
///
/// `usage.record` 没有请求 id，用时间和 token 组合代替。文件被重写、重复扫描
/// 同一个文件都会得到同一个 id，所以导入幂等；同一毫秒且 token 完全相同的两条
/// 记录会被当成一条（实测 877 条里没有这种碰撞）。
fn record_request_id(scope: &FileScope, usage: &KimiUsage) -> String {
    format!(
        "{SOURCE}:{}:{}:{}:{}:{}-{}-{}",
        scope.workdir_key,
        scope.session_id,
        scope.agent_id,
        usage.time_ms,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens
    )
}

/// 插入一条用量到 `proxy_request_logs`；返回 false 表示判定为重复而跳过。
fn insert_kimi_usage(
    db: &Database,
    request_id: &str,
    usage: &KimiUsage,
    session_id: &str,
    timing: Option<RequestTiming>,
) -> Result<bool, AppError> {
    let conn = lock_conn!(db.conn);
    let created_at = usage.time_ms / 1000;

    let dedup_key = DedupKey {
        app_type: APP_TYPE,
        model: &usage.model,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_read_tokens: usage.cache_read_tokens,
        cache_creation_tokens: usage.cache_creation_tokens,
        created_at,
    };
    if should_skip_session_insert(&conn, request_id, &dedup_key)? {
        return Ok(false);
    }

    let token_usage = TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_read_tokens: usage.cache_read_tokens,
        cache_creation_tokens: usage.cache_creation_tokens,
        cache_creation_1h_tokens: 0,
        model: Some(usage.model.clone()),
        message_id: None,
    };

    let multiplier = Decimal::from(1);
    let (input_cost, output_cost, cache_read_cost, cache_creation_cost, total_cost) =
        match find_kimi_pricing(&conn, &usage.model) {
            Some(pricing) => {
                let cost =
                    CostCalculator::calculate_for_app(APP_TYPE, &token_usage, &pricing, multiplier);
                (
                    cost.input_cost.to_string(),
                    cost.output_cost.to_string(),
                    cost.cache_read_cost.to_string(),
                    cost.cache_creation_cost.to_string(),
                    cost.total_cost.to_string(),
                )
            }
            None => (
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
            ),
        };

    conn.execute(
        "INSERT INTO proxy_request_logs (
            request_id, provider_id, app_type, model, request_model,
            input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens,
            input_cost_usd, output_cost_usd, cache_read_cost_usd, cache_creation_cost_usd, total_cost_usd,
            latency_ms, first_token_ms, status_code, error_message, session_id,
            provider_type, is_streaming, cost_multiplier, created_at, data_source
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24)
        ON CONFLICT(request_id) DO NOTHING",
        rusqlite::params![
            request_id,
            format!("_{SOURCE}"), // provider_id
            APP_TYPE,
            usage.model,
            usage.model,         // request_model：会话记录里两者本来就是一个名
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cache_creation_tokens,
            input_cost,
            output_cost,
            cache_read_cost,
            cache_creation_cost,
            total_cost,
            timing.map_or(0, |t| t.latency_ms), // latency_ms：来自 logs/kimi-code.log 的实测两段之和
            timing.map(|t| t.first_token_ms),   // first_token_ms：日志里的 ttftMs
            200i64,
            Option::<String>::None,
            Some(session_id.to_string()),
            Some(SOURCE),
            1i64,                   // is_streaming
            "1.0",
            created_at,
            SOURCE,
        ],
    )
    .map_err(|e| AppError::Database(format!("插入 Kimi 会话日志失败: {e}")))?;

    Ok(conn.changes() > 0)
}

fn find_kimi_pricing(conn: &rusqlite::Connection, model_id: &str) -> Option<ModelPricing> {
    find_model_pricing(conn, model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_wire(
        root: &Path,
        workdir: &str,
        session: &str,
        agent: &str,
        lines: &[Value],
    ) -> PathBuf {
        let dir = root
            .join("sessions")
            .join(workdir)
            .join(session)
            .join("agents")
            .join(agent);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("wire.jsonl");
        let body = lines
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, format!("{body}\n")).unwrap();
        path
    }

    /// 在会话目录下写一份 `logs/kimi-code.log`（与 wire.jsonl 同级）。
    fn write_session_log(wire_path: &Path, lines: &str) {
        let session_dir = wire_path
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let logs = session_dir.join("logs");
        fs::create_dir_all(&logs).unwrap();
        fs::write(logs.join("kimi-code.log"), lines).unwrap();
    }

    /// 真实日志行（字段顺序、双空格、`INFO` 前缀都照抄本机 `logs/kimi-code.log`）。
    const REAL_RESPONSE_LINE: &str = "2026-10-13T03:11:58.493Z INFO  llm response  turnStep=0.1 ttftMs=3479 streamDurationMs=2078 outputTokens=111 requestBuildMs=3 serverFirstTokenMs=3476 serverDecodeMs=2076 clientConsumeMs=2 clientBlockedMs=46";

    fn usage_line(
        agent: &str,
        model: &str,
        time: i64,
        input: u64,
        output: u64,
        cache_read: u64,
    ) -> Value {
        serde_json::json!({
            "type": "usage.record",
            "agentId": agent,
            "model": model,
            "usage": {
                "inputOther": input,
                "output": output,
                "inputCacheRead": cache_read,
                "inputCacheCreation": 0
            },
            "usageScope": "turn",
            "time": time
        })
    }

    #[test]
    fn parse_usage_record_extracts_tokens_and_model() {
        let value = usage_line("main", "kimi-code/k3", 1791861118492, 13294, 111, 18944);
        let usage = parse_usage_record(&value, "fallback", "fallback-model").unwrap();
        assert_eq!(usage.agent_id, "main");
        assert_eq!(usage.model, "k3");
        assert_eq!(usage.input_tokens, 13294);
        assert_eq!(usage.output_tokens, 111);
        assert_eq!(usage.cache_read_tokens, 18944);
        assert_eq!(usage.cache_creation_tokens, 0);
        assert_eq!(usage.time_ms, 1791861118492);
    }

    #[test]
    fn parse_usage_record_falls_back_for_missing_model_and_agent() {
        let value = serde_json::json!({
            "type": "usage.record",
            "usage": { "inputOther": 10, "output": 1 },
            "time": 1000
        });
        let usage = parse_usage_record(&value, "agent-0", "kimi-for-coding").unwrap();
        assert_eq!(usage.agent_id, "agent-0");
        assert_eq!(usage.model, "kimi-for-coding");
        assert_eq!(usage.cache_read_tokens, 0);
    }

    #[test]
    fn parse_usage_record_skips_other_types_and_empty_usage() {
        let other = serde_json::json!({ "type": "llm.request", "model": "kimi-code/k3" });
        assert!(parse_usage_record(&other, "main", "m").is_none());

        let zero = serde_json::json!({
            "type": "usage.record",
            "agentId": "main",
            "model": "kimi-code/k3",
            "usage": { "inputOther": 0, "output": 0, "inputCacheRead": 0, "inputCacheCreation": 0 },
            "time": 5
        });
        assert!(parse_usage_record(&zero, "main", "m").is_none());

        let no_time = serde_json::json!({
            "type": "usage.record",
            "agentId": "main",
            "usage": { "inputOther": 5, "output": 1 }
        });
        assert!(parse_usage_record(&no_time, "main", "m").is_none());
    }

    #[test]
    fn cache_only_record_is_kept() {
        let value = usage_line("main", "kimi-code/k3", 7, 0, 0, 5000);
        let usage = parse_usage_record(&value, "main", "m").unwrap();
        assert_eq!(usage.cache_read_tokens, 5000);
    }

    #[test]
    fn normalize_model_strips_namespace_only() {
        assert_eq!(normalize_model("kimi-code/k3"), "k3");
        assert_eq!(
            normalize_model("kimi-code/kimi-for-coding"),
            "kimi-for-coding"
        );
        assert_eq!(normalize_model("k3"), "k3");
        assert_eq!(normalize_model("  kimi-code/k3  "), "k3");
    }

    #[test]
    fn collect_wire_files_walks_agents_dirs_only() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_wire(
            root,
            "wd_a_1",
            "session_1",
            "main",
            &[usage_line("main", "kimi-code/k3", 1, 1, 1, 0)],
        );
        write_wire(
            root,
            "wd_a_1",
            "session_1",
            "agent-0",
            &[usage_line("agent-0", "kimi-code/k3", 2, 1, 1, 0)],
        );
        write_wire(
            root,
            "wd_b_2",
            "session_2",
            "main",
            &[usage_line("main", "kimi-code/k3", 3, 1, 1, 0)],
        );
        // 旁支目录不该被当成会话
        fs::create_dir_all(root.join("sessions").join("wd_a_1").join("logs")).unwrap();

        let files = collect_wire_files(root);
        assert_eq!(files.len(), 3);
        assert!(files
            .iter()
            .all(|p| p.file_name().is_some_and(|n| n == "wire.jsonl")));
    }

    #[test]
    fn sync_imports_once_then_skips_on_unchanged_file() -> Result<(), AppError> {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_wire(
            tmp.path(),
            "wd_a_1",
            "session_1",
            "main",
            &[
                usage_line("main", "kimi-code/k3", 1791861118000, 100, 10, 0),
                usage_line("main", "kimi-code/k3", 1791861120000, 200, 20, 50),
            ],
        );
        let db = Database::memory().unwrap();
        let (imported, skipped) = sync_single_wire_file(&db, &path, 0, "kimi-for-coding").unwrap();
        assert_eq!((imported, skipped), (2, 0));

        // 同一文件再扫：游标拦下（last_modified 已推进）→ 0/0，不重复入库
        let meta = fs::metadata(&path).unwrap();
        let (imported, skipped) = sync_single_wire_file(
            &db,
            &path,
            metadata_modified_nanos(&meta) + 1,
            "kimi-for-coding",
        )
        .unwrap();
        assert_eq!((imported, skipped), (0, 0));

        let conn = lock_conn!(db.conn);
        let rows: Vec<(String, String, i64, i64, i64, i64, i64)> = conn
            .prepare(
                "SELECT app_type, model, input_tokens, output_tokens, cache_read_tokens, latency_ms,
                        first_token_ms IS NULL
                   FROM proxy_request_logs ORDER BY created_at",
            )
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, "kimi");
        assert_eq!(rows[0].1, "k3");
        assert_eq!(rows[0].2, 100);
        assert_eq!(rows[0].3, 10);
        assert_eq!(rows[1].4, 50);
        assert_eq!(rows[0].5, 0, "wire 里没有耗时，不该编一个");
        assert_eq!(rows[0].6, 1, "first_token_ms 必须是 NULL");
        Ok(())
    }

    #[test]
    fn sync_is_idempotent_when_file_is_reread() -> Result<(), AppError> {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_wire(
            tmp.path(),
            "wd_a_1",
            "session_1",
            "agent-0",
            &[usage_line(
                "agent-0",
                "kimi-code/k3",
                1791861200000,
                5,
                6,
                7,
            )],
        );
        let db = Database::memory().unwrap();

        // 直接以 last_modified = 0 调两次，模拟文件被重写后再扫
        let first = sync_single_wire_file(&db, &path, 0, "kimi-for-coding").unwrap();
        let second = sync_single_wire_file(&db, &path, 0, "kimi-for-coding").unwrap();
        assert_eq!(first, (1, 0));
        assert_eq!(second, (0, 1));

        let conn = lock_conn!(db.conn);
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM proxy_request_logs WHERE app_type = 'kimi'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
        Ok(())
    }

    #[test]
    fn kimi_fresh_input_is_not_double_subtracted() {
        // `inputOther` 本身不含缓存读，所以入库的 input_tokens 已经是新鲜输入。
        // 展示层的 `fresh_input_sql` 只对 cache-inclusive 名单做减法；kimi 一旦被
        // 误加进那个名单，这里就会减出负数。用真实 SQL 读一遍更保险。
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE proxy_request_logs (
                request_id TEXT PRIMARY KEY,
                app_type TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
                input_token_semantics INTEGER NOT NULL DEFAULT 0
            );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO proxy_request_logs
                (request_id, app_type, input_tokens, output_tokens, cache_read_tokens)
             VALUES ('kimi-1', 'kimi', 100, 20, 5000)",
            [],
        )
        .unwrap();

        let sql = format!(
            "SELECT SUM({}) FROM proxy_request_logs l",
            crate::services::sql_helpers::fresh_input_sql("l")
        );
        let fresh: i64 = conn.query_row(&sql, [], |row| row.get(0)).unwrap();
        assert_eq!(fresh, 100, "新鲜输入应当保持 100，不被缓存读减掉");
    }

    #[test]
    fn parse_session_log_reads_ttft_and_stream() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("kimi-code.log");
        fs::write(
            &log,
            format!(
                "{REAL_RESPONSE_LINE}\n\
                 2026-10-13T03:11:52.934Z INFO  llm request  turnStep=0.1\n\
                 2026-10-13T03:11:52.934Z INFO  llm config  turnStep=0.1 model=kimi-for-coding\n"
            ),
        )
        .unwrap();

        let mut timings = parse_session_log(&log);
        let time_ms = parse_log_timestamp_ms("2026-10-13T03:11:58.493Z").unwrap();
        let timing = timings.take(time_ms).expect("响应行应当被解析出来");
        assert_eq!(timing.first_token_ms, 3479);
        // 两段之和 = 5557ms：与同一行的时间戳差（52.934 → 58.493 = 5559ms）相差 2ms
        assert_eq!(timing.latency_ms, 5557);
        assert!(timings.take(time_ms).is_none(), "同一条只能取一次");
    }

    #[test]
    fn session_timings_take_tolerates_few_ms_and_keeps_order() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("kimi-code.log");
        fs::write(
            &log,
            "2026-10-13T03:11:58.493Z INFO  llm response  turnStep=0.1 ttftMs=1000 streamDurationMs=500\n\
             2026-10-13T03:11:58.493Z INFO  llm response  turnStep=0.2 ttftMs=2000 streamDurationMs=600\n",
        )
        .unwrap();

        let base = parse_log_timestamp_ms("2026-10-13T03:11:58.493Z").unwrap();
        let mut timings = parse_session_log(&log);
        // 同一毫秒两条：先到先得，不是覆盖
        assert_eq!(timings.take(base).unwrap().first_token_ms, 1000);
        assert_eq!(timings.take(base + 2).unwrap().first_token_ms, 2000);
        assert!(timings.take(base + 10).is_none(), "超出容差就不再凑数");
    }

    #[test]
    fn sync_attaches_measured_timing_from_session_log() -> Result<(), AppError> {
        let tmp = tempfile::tempdir().unwrap();
        let response_time = parse_log_timestamp_ms("2026-10-13T03:11:58.493Z").unwrap();
        let path = write_wire(
            tmp.path(),
            "wd_a_1",
            "session_1",
            "main",
            &[usage_line(
                "main",
                "kimi-code/kimi-for-coding",
                response_time - 1, // 日志与 wire 的毫秒值实测差 0～1ms
                13294,
                111,
                18944,
            )],
        );
        write_session_log(&path, &format!("{REAL_RESPONSE_LINE}\n"));

        let db = Database::memory().unwrap();
        assert_eq!(
            sync_single_wire_file(&db, &path, 0, "kimi-for-coding").unwrap(),
            (1, 0)
        );

        let conn = lock_conn!(db.conn);
        let (latency, first_token): (i64, Option<i64>) = conn
            .query_row(
                "SELECT latency_ms, first_token_ms FROM proxy_request_logs WHERE app_type = 'kimi'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(latency, 5557);
        assert_eq!(first_token, Some(3479));
        Ok(())
    }

    #[test]
    fn sync_leaves_timing_empty_without_session_log() -> Result<(), AppError> {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_wire(
            tmp.path(),
            "wd_a_1",
            "session_1",
            "main",
            &[usage_line(
                "main",
                "kimi-code/k3",
                1791861118492,
                100,
                10,
                0,
            )],
        );
        let db = Database::memory().unwrap();
        assert_eq!(
            sync_single_wire_file(&db, &path, 0, "kimi-for-coding").unwrap(),
            (1, 0)
        );

        let conn = lock_conn!(db.conn);
        let (latency, first_token): (i64, Option<i64>) = conn
            .query_row(
                "SELECT latency_ms, first_token_ms FROM proxy_request_logs WHERE app_type = 'kimi'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(latency, 0, "没有日志就不编耗时");
        assert_eq!(first_token, None, "没有日志就不编首字");
        Ok(())
    }
}
