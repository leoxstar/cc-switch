//! DeepSeek Harness (DSH) 配置文件读写模块
//!
//! DeepSeek Harness (DSH) configuration read/write module.
//!
//! 处理 DSH 的两代配置面。
//!
//! DSH 使用累加式供应商管理：provider 都挂在 `llm-pi-ai` 的 `providers.<id>`
//! 之下，当前激活路由存在 `agent-default-model` 里。这两个位置在两代文件里形状不同：
//!
//! 1. **老版：`~/.dsh/settings.yaml`**（顶层小节）
//!
//! ```yaml
//! llm-pi-ai:
//!   providers:
//!     a6api-under-0.1-cny:
//!       api: openai-completions
//!       baseURL: https://api.a6api.com/v1
//! agent-default-model:
//!   provider: a6api-under-0.1-cny
//!   model: glm-5.3
//! ```
//!
//! 2. **当前：profile patch 层** `~/.dsh/profiles/<profile>/cordis.patch.yml`
//!    （顶层是 loader patch 条目数组，同一份内容多两层壳）
//!
//! ```yaml
//! - id: llm-pi-ai
//!   name: "@deepseek-ai/dsh-llm-pi-ai"
//!   config:
//!     providers:
//!       a6api-under-0.1-cny:
//!         api: openai-completions
//!         baseURL: https://api.a6api.com/v1
//! - id: agent-default-model
//!   name: "@deepseek-ai/dsh-agent-default-model"
//!   config:
//!     provider: a6api-under-0.1-cny
//!     model: glm-5.3
//! ```
//!
//! `get_dsh_settings_path()` 挑的是**当前真正生效**的那个（见该函数注释）。
//!
//! ## 写入策略
//!
//! 与 Hermes 模块一致，采用「区块级替换」而非整文件 serde 往返：
//! - 老版的 `agent-default-model` 是顶层小节，复用顶层小节定位/替换算法；
//! - patch 层是数组条目，按条目范围替换其 `config:` 块；
//! - provider 是嵌套块（老版缩进 4/patch 层缩进 6），用嵌套块扫描器只替换目标 provider，
//!   保留文件其余部分的注释与格式——patch 层里有用户注释和 Orca 托管块，不能整文件重写；
//! - 每次写入前先备份到 `~/.cc-switch/backups/dsh/`，再原子写回。
//!
//! DSH 自己的配置文件由它以跨进程写锁 + 原子重命名维护，本模块无法参与其锁协议，
//! 因此仅在单次读-改-写内保持一致性，并通过备份支持人工回滚。

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;

use crate::config::{atomic_write, get_app_config_dir};
use crate::error::AppError;
use crate::settings::effective_backup_retain_count;

// ============================================================================
// Path Functions
// ============================================================================

/// 获取 DSH 配置目录
///
/// 解析顺序：
///   1. CCS 设置 `dsh_config_dir`（显式覆盖）
///   2. `DSH_HOME` 环境变量（trim 后非空）
///   3. 平台默认 `~/.dsh`
pub fn get_dsh_dir() -> PathBuf {
    if let Some(override_dir) = crate::settings::get_dsh_override_dir() {
        return override_dir;
    }

    if let Some(raw) = std::env::var_os("DSH_HOME") {
        let value = raw.to_string_lossy();
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    crate::config::get_home_dir().join(".dsh")
}

/// 获取 profile patch 层的文件名（DSH 自己就是这个名字）。
pub const DSH_PATCH_FILE_NAME: &str = "cordis.patch.yml";

/// 当前 DSH profile 名。
///
/// 解析顺序：`DSH_PROFILE` 环境变量（trim 后非空）→ `profiles/desktop` 目录存在就用它
/// （本机 dsh 桌面版就是这个名字，且它只由 Electron 应用启动，CLI 看不到）→
/// profiles 下第一个目录 → `desktop`。
///
/// 为什么不读配置文件：dsh 没有落盘的「当前 profile」记录（桌面应用自己在进程里记住），
/// `dsh --profile <name>` 每次显式传，所以只能靠约定 + 环境变量覆盖。
pub fn get_dsh_profile() -> String {
    if let Some(raw) = std::env::var_os("DSH_PROFILE") {
        let value = raw.to_string_lossy();
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    let profiles = get_dsh_dir().join("profiles");
    if profiles.join("desktop").is_dir() {
        return "desktop".to_string();
    }
    if let Ok(entries) = fs::read_dir(&profiles) {
        let mut dirs: Vec<String> = entries
            .flatten()
            .filter(|entry| entry.path().is_dir())
            .filter_map(|entry| entry.file_name().to_str().map(str::to_owned))
            .collect();
        dirs.sort();
        if let Some(first) = dirs.into_iter().next() {
            return first;
        }
    }
    "desktop".to_string()
}

/// 当前 profile 的 patch 层路径：`<dsh_dir>/profiles/<profile>/cordis.patch.yml`。
pub fn get_dsh_profile_patch_path() -> PathBuf {
    get_dsh_dir()
        .join("profiles")
        .join(get_dsh_profile())
        .join(DSH_PATCH_FILE_NAME)
}

/// 老版 settings.yaml 的路径（仅作为回退与兼容读取）。
pub fn get_dsh_legacy_settings_path() -> PathBuf {
    get_dsh_dir().join("settings.yaml")
}

/// 获取当前生效的 DSH 配置路径。
///
/// 这不是「猜」：本机实测 dsh 0.1.5-rc.2 不读 `~/.dsh/settings.yaml`（该文件已被 dsh
/// 自己改名为 `settings.yaml.imported`），provider 实际存在
/// `profiles/<profile>/cordis.patch.yml` 里。所以优先级：
///
///   1. profile patch 层已存在 → 用它（当前形态）；
///   2. 老版 settings.yaml 已存在 → 用它（旧版 dsh）；
///   3. 两者都没有：profiles 目录已存在 → 按新形态在 patch 层建；
///      否则按老形态建 settings.yaml。
pub fn get_dsh_settings_path() -> PathBuf {
    let patch = get_dsh_profile_patch_path();
    if patch.exists() {
        return patch;
    }

    let legacy = get_dsh_legacy_settings_path();
    if legacy.exists() {
        return legacy;
    }

    if patch
        .parent()
        .map(|profile_dir| profile_dir.is_dir())
        .unwrap_or(false)
        || get_dsh_dir().join("profiles").is_dir()
    {
        return patch;
    }

    legacy
}

fn dsh_write_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

// ============================================================================
// Types
// ============================================================================

/// DSH 写入结果
#[derive(Debug, Clone, Default, Serialize)]
pub struct DshWriteOutcome {
    pub backup_path: Option<String>,
}

/// `agent-default-model` 顶层小节的形状
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DshDefaultModel {
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

// ============================================================================
// Reading
// ============================================================================

/// `~/.dsh/settings.yaml` 的读写（旧版 dsh 的配置面）
fn read_dsh_settings_at(path: &Path) -> Result<YamlValue, AppError> {
    if !path.exists() {
        return Ok(YamlValue::Mapping(serde_yaml::Mapping::new()));
    }
    let raw = fs::read_to_string(path).map_err(|e| AppError::io(path, e))?;
    serde_yaml::from_str(&raw).map_err(|e| {
        AppError::Config(format!(
            "Failed to parse DSH settings at {}: {e}",
            path.display()
        ))
    })
}

/// DSH 配置文件的两种形态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DshLayout {
    /// 旧版：顶层小节（`llm-pi-ai:` / `agent-default-model:`）。
    SettingsFile,
    /// 当前：profile patch 层，顶层是 `- id: <section>` 的条目数组。
    ProfilePatch,
}

impl DshLayout {
    /// `providers:` 所在缩进（旧版 2，patch 层在 `config:` 下为 4）。
    fn providers_indent(self) -> usize {
        match self {
            DshLayout::SettingsFile => 2,
            DshLayout::ProfilePatch => 4,
        }
    }

    /// provider id 一行所在缩进。
    fn provider_indent(self) -> usize {
        self.providers_indent() + 2
    }

    /// 从内容猜形态：顶层 `- id:` 就是 patch 层，顶层 key 就是老版 settings。
    fn detect(raw: &str) -> Option<Self> {
        for line in raw.lines() {
            if is_blank_or_comment(line) {
                continue;
            }
            let trimmed = line.trim_end_matches('\r');
            if line_indent(trimmed) != 0 {
                continue;
            }
            if trimmed.starts_with("- ") {
                return Some(DshLayout::ProfilePatch);
            }
            if trimmed.starts_with("---") || trimmed.starts_with("...") {
                continue;
            }
            if trimmed.contains(':') {
                return Some(DshLayout::SettingsFile);
            }
        }
        None
    }

    /// 解析某个文件该按哪种形态写：能看内容就看内容，空文件/新文件看文件名。
    fn for_path(path: &Path, raw: &str) -> Self {
        if let Some(layout) = Self::detect(raw) {
            return layout;
        }
        if path.file_name().and_then(|name| name.to_str()) == Some(DSH_PATCH_FILE_NAME) {
            return DshLayout::ProfilePatch;
        }
        DshLayout::SettingsFile
    }

    /// 定位一个「节」的行号：老版是 `llm-pi-ai:` 顶层 key，patch 层是 `- id: llm-pi-ai`。
    fn find_section_line(self, lines: &[&str], section: &str) -> Option<usize> {
        match self {
            DshLayout::SettingsFile => find_key_line(lines, 0, lines.len(), 0, section),
            DshLayout::ProfilePatch => find_array_entry_line(lines, 0, lines.len(), "id", section),
        }
    }
}

/// 取出某个节的 `config`（或内容）映射：
/// 老版是顶层 `section:` 的值，patch 层是数组里 `id == section` 那条的 `config`。
fn section_value<'a>(root: &'a YamlValue, section: &str) -> Option<&'a YamlValue> {
    if root.is_mapping() {
        return root.get(section);
    }
    root.as_sequence()?
        .iter()
        .find(|item| item.get("id").and_then(YamlValue::as_str) == Some(section))
        .and_then(|item| item.get("config"))
}

/// 读取 `llm-pi-ai.providers` 下的全部 provider（id → JSON config）。
///
/// 返回的 config 形状即为 DSH YAML 中 provider 块的原样 JSON：
/// displayName / api / baseURL / apiKeyEnv / models（数组）等字段原样透传。
pub fn get_providers_at(path: &Path) -> Result<serde_json::Map<String, JsonValue>, AppError> {
    let config = read_dsh_settings_at(path)?;
    let mut map = serde_json::Map::new();

    let providers = section_value(&config, "llm-pi-ai")
        .and_then(|v| v.get("providers"))
        .and_then(|v| v.as_mapping());

    if let Some(providers) = providers {
        for (key, value) in providers {
            let Some(name) = key.as_str() else {
                continue;
            };
            if name.trim().is_empty() {
                continue;
            }
            match crate::hermes_config::yaml_to_json(value) {
                Ok(json_val) => {
                    map.insert(name.to_string(), json_val);
                }
                Err(e) => {
                    log::warn!("Failed to convert DSH provider '{name}' to JSON: {e}");
                }
            }
        }
    }

    Ok(map)
}

/// 便捷封装：使用默认路径读取 providers。
pub fn get_providers() -> Result<serde_json::Map<String, JsonValue>, AppError> {
    get_providers_at(&get_dsh_settings_path())
}

/// 读取单个 provider。
pub fn get_provider_at(path: &Path, id: &str) -> Result<Option<JsonValue>, AppError> {
    Ok(get_providers_at(path)?.get(id).cloned())
}

/// 读取 `agent-default-model` 小节（老版顶层小节 / patch 层数组条目均可）。
pub fn get_default_model_at(path: &Path) -> Result<Option<DshDefaultModel>, AppError> {
    let config = read_dsh_settings_at(path)?;
    match section_value(&config, "agent-default-model") {
        Some(value) => {
            let json = crate::hermes_config::yaml_to_json(value)?;
            // 容忍未知字段：只提取 provider/model/reasoningEffort。
            let provider = json
                .get("provider")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let model = json
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            if provider.is_empty() || model.is_empty() {
                return Ok(None);
            }
            Ok(Some(DshDefaultModel {
                provider,
                model,
                reasoning_effort: json
                    .get("reasoningEffort")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
            }))
        }
        None => Ok(None),
    }
}

/// 便捷封装：使用默认路径读取 agent-default-model。
pub fn get_default_model() -> Result<Option<DshDefaultModel>, AppError> {
    get_default_model_at(&get_dsh_settings_path())
}

// ============================================================================
// Top-level section helpers（与 hermes_config 相同算法的私有副本）
// ============================================================================

fn is_top_level_key_line(line: &str) -> bool {
    !line.starts_with(' ')
        && !line.starts_with('\t')
        && !line.starts_with('#')
        && !line.starts_with("---")
        && !line.starts_with("...")
        && line.contains(':')
}

/// 查找顶层小节的字节范围（含小节头行，直到下一个顶层 key 或 EOF）。
fn find_yaml_section_range(raw: &str, section_key: &str) -> Option<(usize, usize)> {
    let target = format!("{section_key}:");
    let mut section_start = None;
    let mut offset = 0;

    for line in raw.split('\n') {
        if section_start.is_none() && is_top_level_key_line(line) && line.starts_with(&target) {
            let after_target = &line[target.len()..];
            if after_target.is_empty()
                || after_target.starts_with(' ')
                || after_target.starts_with('\t')
                || after_target.starts_with('\r')
                || after_target.starts_with('#')
            {
                section_start = Some(offset);
            }
        } else if section_start.is_some() && is_top_level_key_line(line) {
            return Some((section_start.unwrap(), offset));
        }
        offset += line.len() + 1;
    }

    section_start.map(|start| (start, raw.len()))
}

fn serialize_yaml_section(key: &str, value: &YamlValue) -> Result<String, AppError> {
    let mut section = serde_yaml::Mapping::new();
    section.insert(YamlValue::String(key.to_string()), value.clone());
    serde_yaml::to_string(&YamlValue::Mapping(section))
        .map_err(|e| AppError::Config(format!("Failed to serialize YAML section '{key}': {e}")))
}

fn remove_all_sections(raw: &str, section_key: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some((start, end)) = find_yaml_section_range(rest, section_key) {
        result.push_str(&rest[..start]);
        rest = &rest[end..];
    }
    result.push_str(rest);
    result
}

fn replace_yaml_section(
    raw: &str,
    section_key: &str,
    value: &YamlValue,
) -> Result<String, AppError> {
    let serialized = serialize_yaml_section(section_key, value)?;

    if let Some((start, end)) = find_yaml_section_range(raw, section_key) {
        let mut result = String::with_capacity(raw.len());
        result.push_str(&raw[..start]);
        result.push_str(&serialized);
        let remainder = remove_all_sections(&raw[end..], section_key);
        if !serialized.ends_with('\n') && !remainder.is_empty() && !remainder.starts_with('\n') {
            result.push('\n');
        }
        result.push_str(&remainder);
        Ok(result)
    } else {
        let mut result = raw.to_string();
        if !result.is_empty() && !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str(&serialized);
        if !result.ends_with('\n') {
            result.push('\n');
        }
        Ok(result)
    }
}

// ============================================================================
// Nested provider block scanner（DSH 专属：llm-pi-ai.providers.<id>）
// ============================================================================

/// 一行的缩进宽度（tab 计 1；DSH settings 由机器生成，只用空格）。
fn line_indent(line: &str) -> usize {
    let trimmed = line.trim_end_matches('\r');
    trimmed.len() - trimmed.trim_start_matches(' ').len()
}

/// 行是否为空行或注释行（不参与块边界判定）。
fn is_blank_or_comment(line: &str) -> bool {
    let t = line.trim();
    t.is_empty() || t.starts_with('#')
}

/// 在 `lines` 的 `[from, to)` 范围内查找缩进为 `indent` 且 key 精确匹配
/// `key` 的行，返回行号。
///
/// 支持 `key:`、`"key":`、`'key':` 三种 YAML key 写法。
fn find_key_line(
    lines: &[&str],
    from: usize,
    to: usize,
    indent: usize,
    key: &str,
) -> Option<usize> {
    for (idx, line) in lines.iter().enumerate().skip(from).take(to - from) {
        if is_blank_or_comment(line) {
            continue;
        }
        let line_indent = line_indent(line);
        if line_indent < indent {
            // 已经离开目标层：继续外层调用方负责的边界。
            return None;
        }
        if line_indent != indent {
            continue;
        }
        let body = line.trim_start_matches(' ');
        let body = body.trim_end_matches('\r');
        let matches = body == format!("{key}:")
            || body == format!("\"{key}\":")
            || body == format!("'{key}':")
            || body == format!("{key}: ")
            || body.starts_with(&format!("{key}: "))
            || body.starts_with(&format!("\"{key}\": "))
            || body.starts_with(&format!("'{key}': "));
        if matches {
            return Some(idx);
        }
    }
    None
}

/// 从 key 行开始，找到该映射块的结束行（exclusive）：
/// 下一行非空、非注释且缩进 <= key 缩进。
fn block_end(lines: &[&str], key_line: usize, key_indent: usize) -> usize {
    let mut end = lines.len();
    for (idx, line) in lines.iter().enumerate().skip(key_line + 1) {
        if is_blank_or_comment(line) {
            continue;
        }
        if line_indent(line) <= key_indent {
            end = idx;
            break;
        }
    }
    end
}

/// 查找顶层数组条目 `- <key>: <value>`（patch 层形态），返回行号。
///
/// 只认缩进 0 的 `- key: value`；`insert:` 列表里那些缩进的 `- id:` 不会误中。
fn find_array_entry_line(
    lines: &[&str],
    from: usize,
    to: usize,
    key: &str,
    value: &str,
) -> Option<usize> {
    for (idx, line) in lines.iter().enumerate().skip(from).take(to - from) {
        if is_blank_or_comment(line) {
            continue;
        }
        if line_indent(line) != 0 {
            continue;
        }
        let trimmed = line.trim_end_matches('\r');
        let Some(rest) = trimmed.strip_prefix("- ") else {
            continue;
        };
        let rest = rest.trim_start();
        let matches = rest == format!("{key}: {value}")
            || rest == format!("{key}: '{value}'")
            || rest == format!("{key}: \"{value}\"");
        if matches {
            return Some(idx);
        }
    }
    None
}

/// 在一个 patch 条目里找 `  config:` 行。
fn find_patch_config_line(lines: &[&str], entry_line: usize) -> Option<usize> {
    let entry_end = block_end(lines, entry_line, 0);
    find_key_line(lines, entry_line + 1, entry_end, 2, "config")
}

/// 在 `llm-pi-ai` 里查找 `providers:` 行（两种形态）。
///
/// 返回 (providers 行号, providers 块结束行)。
fn find_providers_section(lines: &[&str], layout: DshLayout) -> Option<(usize, usize)> {
    let providers_indent = layout.providers_indent();
    match layout {
        DshLayout::SettingsFile => {
            let llm_start = find_key_line(lines, 0, lines.len(), 0, "llm-pi-ai")?;
            let llm_end = block_end(lines, llm_start, 0);
            let providers_line = find_key_line(lines, llm_start + 1, llm_end, 2, "providers")?;
            Some((providers_line, block_end(lines, providers_line, 2)))
        }
        DshLayout::ProfilePatch => {
            let entry_line = find_array_entry_line(lines, 0, lines.len(), "id", "llm-pi-ai")?;
            let config_line = find_patch_config_line(lines, entry_line)?;
            let config_end = block_end(lines, config_line, 2);
            let providers_line = find_key_line(
                lines,
                config_line + 1,
                config_end,
                providers_indent,
                "providers",
            )?;
            Some((
                providers_line,
                block_end(lines, providers_line, providers_indent),
            ))
        }
    }
}

/// 把 provider config（JSON 对象）序列化为带 `indent` 缩进的 YAML 块。
///
/// serde_yaml 输出的属性行从 column 0 开始；本函数统一加上 `indent + 2` 空格，
/// 使属性落在 provider key 的下一层。key 行本身给 `indent` 空格。
fn serialize_provider_block(
    id: &str,
    config: &JsonValue,
    indent: usize,
) -> Result<String, AppError> {
    let obj = config.as_object().ok_or_else(|| {
        AppError::Config(format!("DSH provider '{id}' config must be a JSON object"))
    })?;
    if obj.get("baseURL").and_then(|v| v.as_str()).is_none() {
        return Err(AppError::Config(format!(
            "DSH provider '{id}' config is missing required string field 'baseURL'"
        )));
    }

    let yaml_value = crate::hermes_config::json_to_yaml(config)?;
    let serialized = serde_yaml::to_string(&yaml_value)
        .map_err(|e| AppError::Config(format!("Failed to serialize DSH provider '{id}': {e}")))?;

    let key_pad = " ".repeat(indent);
    let field_pad = " ".repeat(indent + 2);
    let mut block = String::new();
    block.push_str(&key_pad);
    block.push_str(&yaml_key_literal(id));
    block.push_str(":\n");
    for line in serialized.lines() {
        if line.trim().is_empty() {
            block.push('\n');
        } else {
            block.push_str(&field_pad);
            block.push_str(line);
            block.push('\n');
        }
    }
    Ok(block)
}

/// 拼一个完整的 patch 条目文本（`- id:` + `  name:` + `  config:` + 内容）。
///
/// `name` 按 DSH 自己的命名惯例推：`@deepseek-ai/dsh-<id>`（实测桌面 profile 里的
/// `llm-pi-ai` / `agent-default-model` 都是这个形），所以新条目不会缺插件名。
fn serialize_patch_entry(id: &str, config: &YamlValue) -> Result<String, AppError> {
    let mut entry = serde_yaml::Mapping::new();
    entry.insert(
        YamlValue::String("id".to_string()),
        YamlValue::String(id.to_string()),
    );
    entry.insert(
        YamlValue::String("name".to_string()),
        YamlValue::String(format!("@deepseek-ai/dsh-{id}")),
    );
    entry.insert(YamlValue::String("config".to_string()), config.clone());

    let serialized = serde_yaml::to_string(&YamlValue::Mapping(entry)).map_err(|e| {
        AppError::Config(format!("Failed to serialize DSH patch entry '{id}': {e}"))
    })?;
    // serde 会把映射序列化为 `id: …` 各行；在数组里要给首行加 `- `、后续行缩进 2。
    let mut out = String::new();
    for (idx, line) in serialized.lines().enumerate() {
        if line.trim().is_empty() {
            out.push('\n');
            continue;
        }
        if idx == 0 {
            out.push_str("- ");
            out.push_str(line);
        } else {
            out.push_str("  ");
            out.push_str(line);
        }
        out.push('\n');
    }
    Ok(out)
}

/// 生成 YAML key 字面量（仅在需要时加引号）。
fn yaml_key_literal(id: &str) -> String {
    let needs_quotes = id.is_empty()
        || id.contains(':')
        || id.contains('#')
        || id.starts_with(' ')
        || id.ends_with(' ')
        || id.starts_with('\'')
        || id.starts_with('"')
        || id.starts_with('&')
        || id.starts_with('*')
        || id.starts_with('!')
        || id.starts_with('%')
        || id.starts_with('@')
        || id.starts_with('`')
        || id.starts_with('{')
        || id.starts_with('[');
    if needs_quotes {
        format!("\"{}\"", id.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        id.to_string()
    }
}

/// 行尾 key 后的内联值（去掉首尾空白）；`key:` 后为空那么返回空串。
fn inline_body<'a>(line: &'a str, key: &str) -> &'a str {
    line.trim_start_matches(' ')
        .trim_end_matches('\r')
        .trim_start_matches(key)
        .trim()
}

/// 插入点：块尾前最后一行非空非注释行之后（不把新块插在尾部注释后面）。
fn insert_point(lines: &[&str], from: usize, end: usize) -> usize {
    let mut at = end;
    while at > from && is_blank_or_comment(lines[at - 1]) {
        at -= 1;
    }
    at
}

/// 把 provider config 序列化为 patch 层 `llm-pi-ai` 条目的 `config` 映射。
fn patch_llm_config(id: &str, config: &JsonValue) -> Result<YamlValue, AppError> {
    let mut providers = serde_yaml::Mapping::new();
    providers.insert(
        YamlValue::String(id.to_string()),
        crate::hermes_config::json_to_yaml(config)?,
    );
    let mut section_config = serde_yaml::Mapping::new();
    section_config.insert(
        YamlValue::String("providers".to_string()),
        YamlValue::Mapping(providers),
    );
    Ok(YamlValue::Mapping(section_config))
}

/// 在文件末尾追加一个新节（含 providers 与该 provider）。
///
/// patch 层的空数组 `[]` 会先被去掉——`[]` 是合法但为空的 patch 层，留着它再追加条目
/// 会变成非法 YAML（实测 dsh 会直接报 "end of the stream or a document separator is expected"）。
fn append_new_section(
    raw: &str,
    layout: DshLayout,
    id: &str,
    config: &JsonValue,
) -> Result<String, AppError> {
    let new_section = match layout {
        DshLayout::SettingsFile => {
            let block = serialize_provider_block(id, config, layout.provider_indent())?;
            format!(
                "llm-pi-ai:\n{}providers:\n{}",
                " ".repeat(layout.providers_indent()),
                block
            )
        }
        DshLayout::ProfilePatch => {
            serialize_patch_entry("llm-pi-ai", &patch_llm_config(id, config)?)?
        }
    };

    let mut base = match layout {
        DshLayout::SettingsFile => raw.to_string(),
        DshLayout::ProfilePatch => raw
            .split('\n')
            .filter(|line| line.trim() != "[]")
            .collect::<Vec<_>>()
            .join("\n"),
    };
    if !base.trim().is_empty() && !base.ends_with('\n') {
        base.push('\n');
    }
    if !base.trim().is_empty() {
        base.push('\n');
    }
    base.push_str(&new_section);
    if !base.ends_with('\n') {
        base.push('\n');
    }
    Ok(base)
}

/// 对 raw 文本执行「upsert provider 块」的纯函数核心。
///
/// 策略（按结构复杂度递增）：
/// 1. 节 + 容器 + `providers:` + `<id>:` 均为块格式 → 块级替换/插入；
/// 2. `providers: {}` 内联空 → 展开为块格式；
/// 3. 关键行带内联值 → 退化为解析-修改-序列化（只重写该节/条目，注释会丢，但文件其余部分保留）；
/// 4. 没有 `llm-pi-ai` 节（或 patch 层条目里没有 `config:`）→ 追加/插入新块。
fn upsert_provider_raw(
    raw: &str,
    id: &str,
    config: &JsonValue,
    layout: DshLayout,
) -> Result<String, AppError> {
    let provider_indent = layout.provider_indent();
    let providers_indent = layout.providers_indent();
    let block = serialize_provider_block(id, config, provider_indent)?;
    let lines: Vec<&str> = raw.split('\n').collect();

    // 情形 4：没有 llm-pi-ai 节 → 追加。
    let Some(section_line) = layout.find_section_line(&lines, "llm-pi-ai") else {
        return append_new_section(raw, layout, id, config);
    };

    // 定位 providers 的容器：老版就是节本身（缩进 0）；patch 层是节里的 `  config:`（缩进 2）。
    let (container_line, container_indent) = match layout {
        DshLayout::SettingsFile => {
            let body = inline_body(lines[section_line], "llm-pi-ai:");
            if !body.is_empty() && !body.starts_with('#') {
                // 情形 3：整节重写。
                return rewrite_section_via_parse(raw, layout, "llm-pi-ai", |providers| {
                    providers.insert(
                        YamlValue::String(id.to_string()),
                        crate::hermes_config::json_to_yaml(config)?,
                    );
                    Ok(())
                });
            }
            (section_line, 0)
        }
        DshLayout::ProfilePatch => match find_patch_config_line(&lines, section_line) {
            Some(config_line) => {
                let body = inline_body(lines[config_line], "config:");
                if !body.is_empty() && !body.starts_with('#') {
                    // 情形 3：`config:` 带内联值 → 重写该条目。
                    return rewrite_section_via_parse(raw, layout, "llm-pi-ai", |providers| {
                        providers.insert(
                            YamlValue::String(id.to_string()),
                            crate::hermes_config::json_to_yaml(config)?,
                        );
                        Ok(())
                    });
                }
                (config_line, 2)
            }
            None => {
                // 条目存在但没有 `config:` → 在条目内插一个 config + providers + 块。
                let entry_end = block_end(&lines, section_line, 0);
                let insert_at = insert_point(&lines, section_line + 1, entry_end);
                let mut result_lines: Vec<String> =
                    lines[..insert_at].iter().map(|s| s.to_string()).collect();
                result_lines.push("  config:".to_string());
                result_lines.push(format!("{}providers:", " ".repeat(providers_indent)));
                for line in block.lines() {
                    result_lines.push(line.to_string());
                }
                result_lines.extend(lines[insert_at..].iter().map(|s| s.to_string()));
                return Ok(join_lines(&result_lines));
            }
        },
    };

    let container_end = block_end(&lines, container_line, container_indent);

    match find_key_line(
        &lines,
        container_line + 1,
        container_end,
        providers_indent,
        "providers",
    ) {
        Some(providers_line) => {
            let p_body = inline_body(lines[providers_line], "providers:");
            if p_body == "{}" {
                // 情形 2：内联空 providers → 展开为块格式并插入。
                let mut result_lines: Vec<String> = lines[..providers_line]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                result_lines.push(format!("{}providers:", " ".repeat(providers_indent)));
                for line in block.lines() {
                    result_lines.push(line.to_string());
                }
                result_lines.extend(lines[providers_line + 1..].iter().map(|s| s.to_string()));
                return Ok(join_lines(&result_lines));
            }
            if !p_body.is_empty() && !p_body.starts_with('#') {
                // providers 行带其他内联值 → 重写。
                return rewrite_section_via_parse(raw, layout, "llm-pi-ai", |providers| {
                    providers.insert(
                        YamlValue::String(id.to_string()),
                        crate::hermes_config::json_to_yaml(config)?,
                    );
                    Ok(())
                });
            }

            let providers_end = block_end(&lines, providers_line, providers_indent);

            match find_key_line(
                &lines,
                providers_line + 1,
                providers_end,
                provider_indent,
                id,
            ) {
                Some(existing_line) => {
                    // 情形 1a：块级替换。
                    let end = block_end(&lines, existing_line, provider_indent);
                    let mut result_lines: Vec<String> = lines[..existing_line]
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                    for line in block.lines() {
                        result_lines.push(line.to_string());
                    }
                    result_lines.extend(lines[end..].iter().map(|s| s.to_string()));
                    Ok(join_lines(&result_lines))
                }
                None => {
                    // 情形 1b：providers 节内插入（在其块尾之前）。
                    let insert_at = insert_point(&lines, providers_line + 1, providers_end);
                    let mut result_lines: Vec<String> =
                        lines[..insert_at].iter().map(|s| s.to_string()).collect();
                    for line in block.lines() {
                        result_lines.push(line.to_string());
                    }
                    result_lines.extend(lines[insert_at..].iter().map(|s| s.to_string()));
                    Ok(join_lines(&result_lines))
                }
            }
        }
        None => {
            // 容器里没有 providers 键 → 在容器尾插入 providers + 块。
            let insert_at = insert_point(&lines, container_line + 1, container_end);
            let mut result_lines: Vec<String> =
                lines[..insert_at].iter().map(|s| s.to_string()).collect();
            result_lines.push(format!("{}providers:", " ".repeat(providers_indent)));
            for line in block.lines() {
                result_lines.push(line.to_string());
            }
            result_lines.extend(lines[insert_at..].iter().map(|s| s.to_string()));
            Ok(join_lines(&result_lines))
        }
    }
}

/// 对 raw 文本执行「删除 provider 块」的纯函数核心。
fn remove_provider_raw(raw: &str, id: &str, layout: DshLayout) -> Result<String, AppError> {
    let provider_indent = layout.provider_indent();
    let providers_indent = layout.providers_indent();
    let lines: Vec<&str> = raw.split('\n').collect();
    let Some((providers_line, _providers_end)) = find_providers_section(&lines, layout) else {
        return Ok(raw.to_string());
    };
    let providers_end = block_end(&lines, providers_line, providers_indent);

    let Some(existing_line) = find_key_line(
        &lines,
        providers_line + 1,
        providers_end,
        provider_indent,
        id,
    ) else {
        return Ok(raw.to_string());
    };

    let end = block_end(&lines, existing_line, provider_indent);
    let mut result_lines: Vec<String> = lines[..existing_line]
        .iter()
        .map(|s| s.to_string())
        .collect();
    result_lines.extend(lines[end..].iter().map(|s| s.to_string()));

    // 若删除后 providers 块内已无同层 key，则折叠为 `providers: {}`。
    let has_remaining = result_lines[providers_line + 1..]
        .iter()
        .take_while(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#') && line_indent(l) >= providers_indent
        })
        .any(|l| line_indent(l) == provider_indent && !is_blank_or_comment(l));
    if !has_remaining {
        // providers 行号在删除后不变（删除发生在其之后）。
        let mut p_end = result_lines.len();
        for (idx, line) in result_lines.iter().enumerate().skip(providers_line + 1) {
            if is_blank_or_comment(line) {
                continue;
            }
            if line_indent(line) <= providers_indent {
                p_end = idx;
                break;
            }
        }
        result_lines.splice(providers_line + 1..p_end, std::iter::once(String::new()));
        result_lines[providers_line] = format!("{}providers: {{}}", " ".repeat(providers_indent));
    }

    Ok(join_lines(&result_lines))
}

/// 重写一个节的 `providers`（解析-修改-序列化）：
/// - 老版：只重写 `llm-pi-ai` 顶层小节，文件其余部分原样保留；
/// - patch 层：只重写那条 `- id: llm-pi-ai` 条目，其余条目与注释保留。
fn rewrite_section_via_parse<F>(
    raw: &str,
    layout: DshLayout,
    section: &str,
    mutator: F,
) -> Result<String, AppError>
where
    F: FnOnce(&mut serde_yaml::Mapping) -> Result<(), AppError>,
{
    let parsed: YamlValue = serde_yaml::from_str(raw)
        .map_err(|e| AppError::Config(format!("Failed to parse DSH settings: {e}")))?;

    let mut config = section_value(&parsed, section)
        .cloned()
        .unwrap_or(YamlValue::Mapping(serde_yaml::Mapping::new()));
    if !config.is_mapping() {
        config = YamlValue::Mapping(serde_yaml::Mapping::new());
    }
    let config_map = config.as_mapping_mut().expect("checked is_mapping");

    let mut providers = config_map
        .get(YamlValue::String("providers".to_string()))
        .cloned()
        .unwrap_or(YamlValue::Mapping(serde_yaml::Mapping::new()));
    if !providers.is_mapping() {
        providers = YamlValue::Mapping(serde_yaml::Mapping::new());
    }
    let providers_map = providers.as_mapping_mut().expect("checked is_mapping");

    mutator(providers_map)?;

    config_map.insert(
        YamlValue::String("providers".to_string()),
        YamlValue::Mapping(providers_map.clone()),
    );

    match layout {
        DshLayout::SettingsFile => {
            replace_yaml_section(raw, section, &YamlValue::Mapping(config_map.clone()))
        }
        DshLayout::ProfilePatch => {
            let lines: Vec<&str> = raw.split('\n').collect();
            let Some(entry_line) = find_array_entry_line(&lines, 0, lines.len(), "id", section)
            else {
                return Err(AppError::Config(format!(
                    "DSH patch entry '{section}' not found"
                )));
            };
            let entry_end = block_end(&lines, entry_line, 0);
            let replacement =
                serialize_patch_entry(section, &YamlValue::Mapping(config_map.clone()))?;
            let mut result = String::new();
            result.push_str(&lines[..entry_line].join("\n"));
            if entry_line > 0 {
                result.push('\n');
            }
            result.push_str(&replacement);
            result.push_str(&lines[entry_end..].join("\n"));
            if !result.ends_with('\n') {
                result.push('\n');
            }
            Ok(result)
        }
    }
}

/// 在 patch 层里 upsert 一个条目的 `config:` 块（目前用于 `agent-default-model`）。
fn upsert_patch_entry_config_raw(
    raw: &str,
    id: &str,
    config: &YamlValue,
) -> Result<String, AppError> {
    let lines: Vec<&str> = raw.split('\n').collect();
    let Some(entry_line) = find_array_entry_line(&lines, 0, lines.len(), "id", id) else {
        // 条目不存在：去掉空数组占位后追加一个完整条目。
        let base = raw
            .split('\n')
            .filter(|line| line.trim() != "[]")
            .collect::<Vec<_>>()
            .join("\n");
        let mut result = base;
        if !result.trim().is_empty() && !result.ends_with('\n') {
            result.push('\n');
        }
        if !result.trim().is_empty() {
            result.push('\n');
        }
        result.push_str(&serialize_patch_entry(id, config)?);
        if !result.ends_with('\n') {
            result.push('\n');
        }
        return Ok(result);
    };

    let entry_end = block_end(&lines, entry_line, 0);
    let serialized = serde_yaml::to_string(config)
        .map_err(|e| AppError::Config(format!("Failed to serialize DSH entry '{id}': {e}")))?;
    let mut config_block = String::new();
    for line in serialized.lines() {
        if line.trim().is_empty() {
            config_block.push('\n');
        } else {
            config_block.push_str("    ");
            config_block.push_str(line);
            config_block.push('\n');
        }
    }

    match find_patch_config_line(&lines, entry_line) {
        Some(config_line) => {
            let body = inline_body(lines[config_line], "config:");
            if !body.is_empty() && !body.starts_with('#') {
                return rewrite_section_via_parse(raw, DshLayout::ProfilePatch, id, |_providers| {
                    Err(AppError::Config(
                        "DSH patch entry uses an inline config; rewrite it by hand".to_string(),
                    ))
                });
            }
            let config_end = block_end(&lines, config_line, 2);
            let mut result_lines: Vec<String> =
                lines[..config_line].iter().map(|s| s.to_string()).collect();
            result_lines.push("  config:".to_string());
            for line in config_block.lines() {
                result_lines.push(line.to_string());
            }
            result_lines.extend(lines[config_end..].iter().map(|s| s.to_string()));
            Ok(join_lines(&result_lines))
        }
        None => {
            let insert_at = insert_point(&lines, entry_line + 1, entry_end);
            let mut result_lines: Vec<String> =
                lines[..insert_at].iter().map(|s| s.to_string()).collect();
            result_lines.push("  config:".to_string());
            for line in config_block.lines() {
                result_lines.push(line.to_string());
            }
            result_lines.extend(lines[insert_at..].iter().map(|s| s.to_string()));
            Ok(join_lines(&result_lines))
        }
    }
}

fn join_lines(lines: &[String]) -> String {
    let mut out = lines.join("\n");
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

// ============================================================================
// Backup & Cleanup
// ============================================================================

fn create_dsh_backup(source: &str) -> Result<PathBuf, AppError> {
    let backup_dir = get_app_config_dir().join("backups").join("dsh");
    fs::create_dir_all(&backup_dir).map_err(|e| AppError::io(&backup_dir, e))?;

    let base_id = format!("dsh_{}", Local::now().format("%Y%m%d_%H%M%S"));
    let mut filename = format!("{base_id}.yaml");
    let mut backup_path = backup_dir.join(&filename);
    let mut counter = 1;

    while backup_path.exists() {
        filename = format!("{base_id}_{counter}.yaml");
        backup_path = backup_dir.join(&filename);
        counter += 1;
    }

    atomic_write(&backup_path, source.as_bytes())?;
    cleanup_dsh_backups(&backup_dir)?;
    Ok(backup_path)
}

fn cleanup_dsh_backups(dir: &Path) -> Result<(), AppError> {
    let retain = effective_backup_retain_count();
    let mut entries = fs::read_dir(dir)
        .map_err(|e| AppError::io(dir, e))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .map(|ext| ext == "yaml" || ext == "yml")
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();

    if entries.len() <= retain {
        return Ok(());
    }

    entries.sort_by_key(|entry| entry.metadata().and_then(|m| m.modified()).ok());
    let remove_count = entries.len().saturating_sub(retain);
    for entry in entries.into_iter().take(remove_count) {
        if let Err(err) = fs::remove_file(entry.path()) {
            log::warn!(
                "Failed to remove old DSH settings backup {}: {err}",
                entry.path().display()
            );
        }
    }

    Ok(())
}

// ============================================================================
// High-level Write Operations
// ============================================================================

/// 写入（upsert）一个 DSH provider 到指定 settings 文件。
///
/// 区块级替换 + 备份 + 原子写；不触碰 `agent-default-model`。
pub fn set_provider_at(
    path: &Path,
    id: &str,
    config: &JsonValue,
) -> Result<DshWriteOutcome, AppError> {
    let _guard = dsh_write_lock()
        .lock()
        .map_err(|e| AppError::Config(format!("DSH write lock poisoned: {e}")))?;

    let raw = if path.exists() {
        fs::read_to_string(path).map_err(|e| AppError::io(path, e))?
    } else {
        String::new()
    };

    let new_raw = upsert_provider_raw(&raw, id, config, DshLayout::for_path(path, &raw))?;

    if new_raw == raw {
        return Ok(DshWriteOutcome::default());
    }

    // 写前校验：新内容必须能被解析，且 provider 可读回。
    let reparsed: YamlValue = serde_yaml::from_str(&new_raw).map_err(|e| {
        AppError::Config(format!(
            "Refusing to write invalid DSH settings for provider '{id}': {e}"
        ))
    })?;
    let readback = section_value(&reparsed, "llm-pi-ai")
        .and_then(|v| v.get("providers"))
        .and_then(|v| v.get(id));
    if readback.is_none() {
        return Err(AppError::Config(format!(
            "DSH provider '{id}' missing after upsert — refusing to write"
        )));
    }

    let backup_path = if !raw.is_empty() {
        Some(create_dsh_backup(&raw)?)
    } else {
        None
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| AppError::io(parent, e))?;
    }

    atomic_write(path, new_raw.as_bytes())?;

    log::debug!("DSH provider '{id}' written to {path:?}");
    Ok(DshWriteOutcome {
        backup_path: backup_path.map(|p| p.display().to_string()),
    })
}

/// 便捷封装：默认路径 upsert provider。
pub fn set_provider(id: &str, config: JsonValue) -> Result<DshWriteOutcome, AppError> {
    set_provider_at(&get_dsh_settings_path(), id, &config)
}

/// 从指定 settings 文件删除一个 DSH provider。
pub fn remove_provider_at(path: &Path, id: &str) -> Result<DshWriteOutcome, AppError> {
    let _guard = dsh_write_lock()
        .lock()
        .map_err(|e| AppError::Config(format!("DSH write lock poisoned: {e}")))?;

    if !path.exists() {
        return Ok(DshWriteOutcome::default());
    }
    let raw = fs::read_to_string(path).map_err(|e| AppError::io(path, e))?;
    let new_raw = remove_provider_raw(&raw, id, DshLayout::for_path(path, &raw))?;

    if new_raw == raw {
        return Ok(DshWriteOutcome::default());
    }

    // 写前校验：删除后的内容必须仍可解析。
    serde_yaml::from_str::<YamlValue>(&new_raw).map_err(|e| {
        AppError::Config(format!(
            "Refusing to write invalid DSH settings after removing '{id}': {e}"
        ))
    })?;

    let backup_path = if !raw.is_empty() {
        Some(create_dsh_backup(&raw)?)
    } else {
        None
    };

    atomic_write(path, new_raw.as_bytes())?;

    log::debug!("DSH provider '{id}' removed from {path:?}");
    Ok(DshWriteOutcome {
        backup_path: backup_path.map(|p| p.display().to_string()),
    })
}

/// 便捷封装：默认路径删除 provider。
pub fn remove_provider(id: &str) -> Result<DshWriteOutcome, AppError> {
    remove_provider_at(&get_dsh_settings_path(), id)
}

/// 覆写顶层 `agent-default-model` 小节（切换当前路由）。
///
/// 只重写该顶层小节，保留其余内容。
pub fn set_default_model_at(
    path: &Path,
    default_model: &DshDefaultModel,
) -> Result<DshWriteOutcome, AppError> {
    let _guard = dsh_write_lock()
        .lock()
        .map_err(|e| AppError::Config(format!("DSH write lock poisoned: {e}")))?;

    let raw = if path.exists() {
        fs::read_to_string(path).map_err(|e| AppError::io(path, e))?
    } else {
        String::new()
    };

    if default_model.provider.trim().is_empty() || default_model.model.trim().is_empty() {
        return Err(AppError::Config(
            "DSH default model requires non-empty provider and model".to_string(),
        ));
    }

    let mut section = serde_yaml::Mapping::new();
    section.insert(
        YamlValue::String("provider".to_string()),
        YamlValue::String(default_model.provider.clone()),
    );
    section.insert(
        YamlValue::String("model".to_string()),
        YamlValue::String(default_model.model.clone()),
    );
    if let Some(effort) = &default_model.reasoning_effort {
        section.insert(
            YamlValue::String("reasoningEffort".to_string()),
            YamlValue::String(effort.clone()),
        );
    }

    let new_raw = match DshLayout::for_path(path, &raw) {
        DshLayout::SettingsFile => {
            replace_yaml_section(&raw, "agent-default-model", &YamlValue::Mapping(section))?
        }
        DshLayout::ProfilePatch => upsert_patch_entry_config_raw(
            &raw,
            "agent-default-model",
            &YamlValue::Mapping(section),
        )?,
    };

    if new_raw == raw {
        return Ok(DshWriteOutcome::default());
    }

    let backup_path = if !raw.is_empty() {
        Some(create_dsh_backup(&raw)?)
    } else {
        None
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| AppError::io(parent, e))?;
    }

    atomic_write(path, new_raw.as_bytes())?;

    log::debug!(
        "DSH agent-default-model set to {}/{} at {:?}",
        default_model.provider,
        default_model.model,
        path
    );
    Ok(DshWriteOutcome {
        backup_path: backup_path.map(|p| p.display().to_string()),
    })
}

/// 便捷封装：默认路径设置 agent-default-model。
pub fn set_default_model(default_model: &DshDefaultModel) -> Result<DshWriteOutcome, AppError> {
    set_default_model_at(&get_dsh_settings_path(), default_model)
}

/// 检查 provider 是否存在于 DSH settings。
pub fn provider_exists(id: &str) -> Result<bool, AppError> {
    Ok(get_providers()?.contains_key(id))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SAMPLE: &str = r#"# DSH settings
ui-onboarding:
  welcomeNoticeVersion: 2026-08-13.1
agent-presets:
  default: standard-noweb
llm-pi-ai:
  providers:
    a6api-under-0.1-cny:
      displayName: A6API <= 0.1
      apiKeyEnv: A6API_UNDER_0_1_CNY_API_KEY
      api: openai-completions
      baseURL: https://api.a6api.com/v1
      models:
        - id: glm-5.3
          reasoningEfforts:
            off: null
            low: low
        - id: glm-5.3-flash
agent-default-model:
  provider: a6api-under-0.5-cny
  model: gpt-6-astra
  reasoningEffort: xhigh
"#;

    fn provider_config() -> JsonValue {
        json!({
            "displayName": "Test Relay",
            "apiKeyEnv": "TEST_RELAY_API_KEY",
            "api": "openai-completions",
            "baseURL": "https://relay.example.com/v1",
            "models": [
                {"id": "test-model", "reasoningEfforts": {"off": null, "low": "low"}}
            ]
        })
    }

    #[test]
    fn upsert_into_existing_providers_preserves_other_sections() {
        let raw = upsert_provider_raw(
            SAMPLE,
            "test-relay",
            &provider_config(),
            DshLayout::SettingsFile,
        )
        .unwrap();
        assert!(raw.contains("test-relay:"));
        assert!(raw.contains("https://relay.example.com/v1"));
        // 其他顶层小节原样保留
        assert!(raw.contains("agent-presets:"));
        assert!(raw.contains("standard-noweb"));
        assert!(raw.contains("# DSH settings"));
        // 既有 provider 不受影响
        assert!(raw.contains("a6api-under-0.1-cny:"));
        assert!(raw.contains("glm-5.3-flash"));
        // agent-default-model 未被触碰
        assert!(raw.contains("gpt-6-astra"));
        // 结果可解析且新 provider 可读回
        let parsed: YamlValue = serde_yaml::from_str(&raw).unwrap();
        assert!(parsed
            .get("llm-pi-ai")
            .and_then(|v| v.get("providers"))
            .and_then(|v| v.get("test-relay"))
            .is_some());
    }

    #[test]
    fn upsert_replaces_existing_provider_block() {
        let once = upsert_provider_raw(
            SAMPLE,
            "test-relay",
            &provider_config(),
            DshLayout::SettingsFile,
        )
        .unwrap();
        let updated_config = json!({
            "displayName": "Test Relay v2",
            "apiKeyEnv": "TEST_RELAY_API_KEY",
            "api": "openai-completions",
            "baseURL": "https://relay2.example.com/v1",
            "models": [
                {"id": "test-model-v2"}
            ]
        });
        let twice = upsert_provider_raw(
            &once,
            "test-relay",
            &updated_config,
            DshLayout::SettingsFile,
        )
        .unwrap();
        assert!(twice.contains("Test Relay v2"));
        assert!(twice.contains("relay2.example.com"));
        assert!(!twice.contains("relay.example.com"));
        // 只有一份 test-relay 块
        assert_eq!(twice.matches("test-relay:").count(), 1);
    }

    #[test]
    fn upsert_into_file_without_llm_section_appends() {
        let raw = "locale:\n  preference: en\n";
        let out = upsert_provider_raw(
            raw,
            "test-relay",
            &provider_config(),
            DshLayout::SettingsFile,
        )
        .unwrap();
        assert!(out.contains("locale:"));
        assert!(out.contains("llm-pi-ai:"));
        assert!(out.contains("  providers:"));
        let parsed: YamlValue = serde_yaml::from_str(&out).unwrap();
        assert!(parsed
            .get("llm-pi-ai")
            .and_then(|v| v.get("providers"))
            .and_then(|v| v.get("test-relay"))
            .is_some());
    }

    #[test]
    fn upsert_expands_inline_empty_providers() {
        let raw = "llm-pi-ai:\n  providers: {}\nagent-default-model:\n  provider: x\n";
        let out = upsert_provider_raw(
            raw,
            "test-relay",
            &provider_config(),
            DshLayout::SettingsFile,
        )
        .unwrap();
        assert!(out.contains("test-relay:"));
        let parsed: YamlValue = serde_yaml::from_str(&out).unwrap();
        assert!(parsed
            .get("llm-pi-ai")
            .and_then(|v| v.get("providers"))
            .and_then(|v| v.get("test-relay"))
            .is_some());
    }

    #[test]
    fn upsert_rejects_config_without_base_url() {
        let bad = json!({"displayName": "No URL"});
        assert!(upsert_provider_raw(SAMPLE, "bad-relay", &bad, DshLayout::SettingsFile).is_err());
    }

    #[test]
    fn remove_existing_provider_keeps_siblings() {
        let raw = upsert_provider_raw(
            SAMPLE,
            "test-relay",
            &provider_config(),
            DshLayout::SettingsFile,
        )
        .unwrap();
        let out = remove_provider_raw(&raw, "test-relay", DshLayout::SettingsFile).unwrap();
        assert!(!out.contains("test-relay:"));
        assert!(!out.contains("relay.example.com"));
        assert!(out.contains("a6api-under-0.1-cny:"));
        assert!(out.contains("gpt-6-astra"));
        let parsed: YamlValue = serde_yaml::from_str(&out).unwrap();
        assert!(parsed
            .get("llm-pi-ai")
            .and_then(|v| v.get("providers"))
            .and_then(|v| v.get("test-relay"))
            .is_none());
        assert!(parsed
            .get("llm-pi-ai")
            .and_then(|v| v.get("providers"))
            .and_then(|v| v.get("a6api-under-0.1-cny"))
            .is_some());
    }

    #[test]
    fn remove_last_provider_collapses_to_empty_map() {
        let raw = "llm-pi-ai:\n  providers:\n    only-one:\n      baseURL: https://x.example\n";
        let out = remove_provider_raw(raw, "only-one", DshLayout::SettingsFile).unwrap();
        assert!(out.contains("providers: {}"));
        let parsed: YamlValue = serde_yaml::from_str(&out).unwrap();
        let providers = parsed
            .get("llm-pi-ai")
            .and_then(|v| v.get("providers"))
            .unwrap();
        assert!(providers
            .as_mapping()
            .map(|m| m.is_empty())
            .unwrap_or(false));
    }

    #[test]
    fn remove_missing_provider_is_noop() {
        let out = remove_provider_raw(SAMPLE, "ghost-relay", DshLayout::SettingsFile).unwrap();
        assert_eq!(out, SAMPLE);
    }

    #[test]
    fn set_default_model_replaces_only_that_section() {
        let raw = SAMPLE.to_string();
        let out = replace_yaml_section(
            &raw,
            "agent-default-model",
            &serde_yaml::from_str::<YamlValue>(
                "provider: a6api-under-0.1-cny\nmodel: glm-5.3\nreasoningEffort: low",
            )
            .unwrap(),
        )
        .unwrap();
        assert!(out.contains("glm-5.3"));
        assert!(!out.contains("gpt-6-astra"));
        assert!(out.contains("standard-noweb"));
        assert!(out.contains("# DSH settings"));
    }

    #[test]
    fn end_to_end_file_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "ccswitch-dsh-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.yaml");
        fs::write(&path, SAMPLE).unwrap();

        // upsert
        let outcome = set_provider_at(&path, "test-relay", &provider_config()).unwrap();
        assert!(outcome.backup_path.is_some());
        let providers = get_providers_at(&path).unwrap();
        assert!(providers.contains_key("test-relay"));
        assert!(providers.contains_key("a6api-under-0.1-cny"));
        assert_eq!(
            providers.get("test-relay").unwrap().get("baseURL"),
            Some(&json!("https://relay.example.com/v1"))
        );

        // default model
        set_default_model_at(
            &path,
            &DshDefaultModel {
                provider: "test-relay".to_string(),
                model: "test-model".to_string(),
                reasoning_effort: Some("low".to_string()),
            },
        )
        .unwrap();
        let dm = get_default_model_at(&path).unwrap().unwrap();
        assert_eq!(dm.provider, "test-relay");
        assert_eq!(dm.model, "test-model");
        assert_eq!(dm.reasoning_effort.as_deref(), Some("low"));

        // remove
        remove_provider_at(&path, "test-relay").unwrap();
        let providers = get_providers_at(&path).unwrap();
        assert!(!providers.contains_key("test-relay"));

        fs::remove_dir_all(&dir).ok();
    }

    // ========================================================================
    // profile patch 层（当前 dsh 形态）
    // ========================================================================

    /// 形状照抄本机 `~/.dsh/profiles/desktop/cordis.patch.yml`：顶层是 loader patch
    /// 条目数组，前面有注释和 Orca 托管块，`llm-pi-ai` 的 providers 里还有用户注释。
    const PATCH_SAMPLE: &str = r#"# Your patch layer for this dsh profile, applied after every bundle layer:
# a top-level YAML array of loader patch entries (id-targeted config
# overrides, disables, and insert lists; `!!js` expressions allowed).

# >>> orca-managed-dsh-hooks (managed by Orca; do not edit) >>>
- insert:
    - id: orca-agent-hooks
      name: '@deepseek-ai/dsh-hooks-claude-code'
      config:
        configPath: '/tmp/orca-hooks.json'
# <<< orca-managed-dsh-hooks <<<

- id: ui-chat
  name: "@deepseek-ai/dsh-client-ui-chat"
  config:
    transcriptView: verbose

- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      # 用户写在 providers 里的注释，块级替换不能吃掉它
      a6api-under-0.1-cny:
        apiKeyEnv: A6API_UNDER_0_1_CNY_API_KEY
        api: openai-completions
        baseURL: https://api.a6api.com/v1
        models:
          - id: glm-5.3

- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: a6api-under-0.1-cny
    model: glm-5.3
"#;

    fn write_patch(dir: &tempfile::TempDir) -> PathBuf {
        let path = dir.path().join(DSH_PATCH_FILE_NAME);
        fs::write(&path, PATCH_SAMPLE).unwrap();
        path
    }

    #[test]
    fn patch_layout_reads_providers_and_default_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_patch(&dir);

        let providers = get_providers_at(&path).unwrap();
        assert_eq!(providers.len(), 1);
        let provider = providers.get("a6api-under-0.1-cny").unwrap();
        assert_eq!(
            provider.get("baseURL").and_then(|v| v.as_str()),
            Some("https://api.a6api.com/v1")
        );

        let model = get_default_model_at(&path).unwrap().expect("default model");
        assert_eq!(model.provider, "a6api-under-0.1-cny");
        assert_eq!(model.model, "glm-5.3");

        // 老版顶层小节的读法在 patch 层上不能也「碰巧」成功。
        assert!(section_value(
            &serde_yaml::from_str::<YamlValue>(PATCH_SAMPLE).unwrap(),
            "llm-pi-ai"
        )
        .is_some());
    }

    #[test]
    fn patch_layout_upsert_preserves_siblings_comments_and_other_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_patch(&dir);

        set_provider_at(&path, "test-relay", &provider_config()).unwrap();
        let out = fs::read_to_string(&path).unwrap();

        // 注释、Orca 托管块、别的条目、原 provider 的字段都还在。
        for needle in [
            "用户写在 providers 里的注释",
            "orca-managed-dsh-hooks",
            "transcriptView: verbose",
            "A6API_UNDER_0_1_CNY_API_KEY",
            "@deepseek-ai/dsh-agent-default-model",
        ] {
            assert!(out.contains(needle), "lost {needle}:\n{out}");
        }

        let providers = get_providers_at(&path).unwrap();
        assert_eq!(providers.len(), 2);
        assert!(providers.contains_key("test-relay"));
        assert!(providers["a6api-under-0.1-cny"]["models"].is_array());

        // 默认模型不被 provider 写入碰到。
        let model = get_default_model_at(&path).unwrap().unwrap();
        assert_eq!(model.model, "glm-5.3");

        // 再写一次同 id：替换而不是追加。
        set_provider_at(&path, "test-relay", &provider_config()).unwrap();
        assert_eq!(get_providers_at(&path).unwrap().len(), 2);
    }

    #[test]
    fn patch_layout_remove_keeps_siblings_and_collapses_when_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_patch(&dir);

        remove_provider_at(&path, "a6api-under-0.1-cny").unwrap();
        let out = fs::read_to_string(&path).unwrap();
        assert!(out.contains("providers: {}"), "{out}");
        assert!(out.contains("transcriptView: verbose"), "{out}");
        assert!(get_providers_at(&path).unwrap().is_empty());

        // 空 providers 上再写一条：展开回块格式。
        set_provider_at(&path, "test-relay", &provider_config()).unwrap();
        let providers = get_providers_at(&path).unwrap();
        assert_eq!(providers.len(), 1);
        assert!(providers.contains_key("test-relay"));
    }

    #[test]
    fn patch_layout_upsert_into_empty_array_file() {
        // 本机 headless profile 就是「注释 + []」：留着 `[]` 再追加条目会变成非法 YAML。
        let raw = "# Your patch layer for this dsh profile:\n[]\n";
        let out = upsert_provider_raw(
            raw,
            "test-relay",
            &provider_config(),
            DshLayout::ProfilePatch,
        )
        .unwrap();
        assert!(!out.contains("\n[]\n"), "{out}");

        let parsed: YamlValue = serde_yaml::from_str(&out).expect("result must stay valid YAML");
        let providers = section_value(&parsed, "llm-pi-ai")
            .and_then(|v| v.get("providers"))
            .and_then(|v| v.as_mapping())
            .expect("providers block");
        assert_eq!(providers.len(), 1);
        assert!(out.contains("@deepseek-ai/dsh-llm-pi-ai"), "{out}");
    }

    #[test]
    fn patch_layout_default_model_upsert_only_touches_its_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_patch(&dir);

        set_default_model_at(
            &path,
            &DshDefaultModel {
                provider: "test-relay".to_string(),
                model: "test-model".to_string(),
                reasoning_effort: Some("low".to_string()),
            },
        )
        .unwrap();

        let out = fs::read_to_string(&path).unwrap();
        let model = get_default_model_at(&path).unwrap().unwrap();
        assert_eq!(model.provider, "test-relay");
        assert_eq!(model.model, "test-model");
        assert_eq!(model.reasoning_effort.as_deref(), Some("low"));

        // provider 与 Orca 条目没被动。
        assert_eq!(get_providers_at(&path).unwrap().len(), 1);
        assert!(out.contains("orca-managed-dsh-hooks"));

        // 条目不存在时：追加带 name 的完整条目。
        let raw = "# patch\n[]\n";
        let appended = upsert_patch_entry_config_raw(
            raw,
            "agent-default-model",
            &YamlValue::Mapping({
                let mut m = serde_yaml::Mapping::new();
                m.insert(
                    YamlValue::String("provider".to_string()),
                    YamlValue::String("p".to_string()),
                );
                m.insert(
                    YamlValue::String("model".to_string()),
                    YamlValue::String("m".to_string()),
                );
                m
            }),
        )
        .unwrap();
        assert!(appended.contains("- id: agent-default-model"), "{appended}");
        assert!(
            appended.contains("@deepseek-ai/dsh-agent-default-model"),
            "{appended}"
        );
        serde_yaml::from_str::<YamlValue>(&appended).expect("appended file must parse");
    }

    /// 当前生效路径的选择：patch 层优先，其次老版 settings.yaml，都没有时按目录形态决定。
    #[test]
    #[serial_test::serial]
    fn live_settings_path_prefers_profile_patch_then_legacy() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let profile_dir = root.join("profiles").join("desktop");
        fs::create_dir_all(&profile_dir).unwrap();
        let patch = profile_dir.join(DSH_PATCH_FILE_NAME);

        unsafe {
            std::env::set_var("DSH_HOME", root);
            std::env::set_var("DSH_PROFILE", "desktop");
        }

        // 两者都没有：profiles/desktop 已存在 → 按新形态指向 patch。
        assert_eq!(get_dsh_settings_path(), patch);

        // 老版 settings.yaml 存在 → 用它（旧版 dsh 还在用）。
        let legacy = root.join("settings.yaml");
        fs::write(&legacy, "llm-pi-ai:\n  providers: {}\n").unwrap();
        assert_eq!(get_dsh_settings_path(), legacy);

        // patch 层也在 → patch 优先。
        fs::write(&patch, "[]\n").unwrap();
        assert_eq!(get_dsh_settings_path(), patch);

        unsafe {
            std::env::remove_var("DSH_HOME");
            std::env::remove_var("DSH_PROFILE");
        }
        fs::remove_dir_all(root).ok();
    }
}
