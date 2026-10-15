//! 读取 CLI 自己登录后落盘的 OAuth 凭据（只读，不登录、不刷新）
//!
//! Kimi Code 与 MCode 都有官方登录（设备码 / 浏览器），登录结果由 CLI 自己写在
//! 各自的目录里。cc-switch 沿用 `subscription.rs` 的策略：**只读凭据**，用来查
//! 套餐额度；token 过期时照打一次请求，401 由调用方按「凭据过期」呈现——用户下次
//! 跑 CLI 时它会自己刷新，cc-switch 不碰刷新流程（刷新会轮换 refresh token，
//! 万一写回失败会把 CLI 踢下线）。
//!
//! 凭据位置（都是实测到的当前形态）：
//! - Kimi Code：`$KIMI_CODE_HOME/credentials/kimi-code.json` → `access_token`
//! - MCode：`~/.minimax/auth/<buildEnv>/<region>/<clientId>/auth.json` →
//!   `records.*.accessToken`，同一台机器可能有 cn/en、prod/staging 多条，取
//!   `expiresAtMs` 最大的那条。

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

/// Kimi Code 的 OAuth access token；文件缺失、内容非法或 token 为空时返回 `None`。
pub(crate) fn kimi_access_token() -> Option<String> {
    kimi_access_token_in(&crate::kimi_config::get_kimi_dir())
}

pub(crate) fn kimi_access_token_in(kimi_dir: &Path) -> Option<String> {
    let path = kimi_dir.join("credentials").join("kimi-code.json");
    let raw = fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&raw).ok()?;
    non_empty_str(value.get("access_token")?)
}

/// MCode 的 OAuth access token；没有记录时返回 `None`。
pub(crate) fn minimax_access_token() -> Option<String> {
    minimax_access_token_in(&mcode_auth_root())
}

/// `~/.minimax/auth`（MCode 的凭据库根目录）。
///
/// 不复用 `mcode_config` 的目录：那个是可被设置覆盖的 MCode 家目录（放配置与
/// sessions），凭据固定在家目录下的 `auth/`，没有独立的覆盖项。
fn mcode_auth_root() -> PathBuf {
    crate::config::get_home_dir().join(".minimax").join("auth")
}

pub(crate) fn minimax_access_token_in(auth_root: &Path) -> Option<String> {
    let mut best: Option<(i64, String)> = None;

    for build_env in read_dirs(auth_root) {
        for region in read_dirs(&build_env) {
            for client in read_dirs(&region) {
                let Some(token) = read_mcode_auth_file(&client.join("auth.json")) else {
                    continue;
                };
                // 过期时间只用来挑最新的一条，不在这里判过期：过期也让调用方试一次，
                // 和 subscription.rs 对 Claude/Codex 的处理一致。
                if best
                    .as_ref()
                    .is_none_or(|(expires_at, _)| token.0 >= *expires_at)
                {
                    best = Some(token);
                }
            }
        }
    }

    best.map(|(_, token)| token)
}

/// 读一个 MCode `auth.json`，返回 (expiresAtMs, accessToken)。
fn read_mcode_auth_file(path: &Path) -> Option<(i64, String)> {
    let raw = fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&raw).ok()?;
    let records = value.get("records")?.as_object()?;

    records
        .values()
        .filter_map(|record| {
            let token = non_empty_str(record.get("accessToken")?)?;
            let expires_at = record
                .get("expiresAtMs")
                .and_then(Value::as_i64)
                .unwrap_or_default();
            Some((expires_at, token))
        })
        .max_by_key(|(expires_at, _)| *expires_at)
}

/// 列出目录下的子目录（读不到就空），并排序，保证挑选结果稳定。
fn read_dirs(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    dirs
}

fn non_empty_str(value: &Value) -> Option<String> {
    let text = value.as_str()?.trim();
    (!text.is_empty()).then(|| text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kimi_reads_access_token() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("credentials");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("kimi-code.json"),
            r#"{"access_token":"tok","refresh_token":"r","expires_at":1}"#,
        )
        .unwrap();

        assert_eq!(kimi_access_token_in(tmp.path()).as_deref(), Some("tok"));
    }

    #[test]
    fn kimi_missing_or_broken_file_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(kimi_access_token_in(tmp.path()), None);

        let dir = tmp.path().join("credentials");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("kimi-code.json"), "not json").unwrap();
        assert_eq!(kimi_access_token_in(tmp.path()), None);

        // 空 token（登录被清掉后的残留）不算凭据
        fs::write(dir.join("kimi-code.json"), r#"{"access_token":"  "}"#).unwrap();
        assert_eq!(kimi_access_token_in(tmp.path()), None);
    }

    #[test]
    fn minimax_picks_the_freshest_record_across_namespaces() {
        let tmp = tempfile::tempdir().unwrap();
        let write = |path: &[&str], expires_at: i64, token: &str| {
            let dir = path
                .iter()
                .fold(tmp.path().to_path_buf(), |acc, part| acc.join(part));
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("auth.json"),
                format!(
                    r#"{{"schemaVersion":1,"records":{{"k {token}":{{"accessToken":"{token}","expiresAtMs":{expires_at}}}}}}}"#
                ),
            )
            .unwrap();
        };
        // cn/prod 旧、en/prod 新、cn/staging 更新
        write(&["prod", "cn", "mcode-public"], 100, "old-cn");
        write(&["prod", "en", "mcode-public"], 200, "new-en");
        write(&["staging", "cn", "mcode-public"], 300, "newest-staging");

        assert_eq!(
            minimax_access_token_in(tmp.path()).as_deref(),
            Some("newest-staging")
        );
    }

    #[test]
    fn minimax_ignores_records_without_token() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("prod").join("cn").join("mcode-public");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("auth.json"),
            r#"{"records":{"k":{"accessToken":"","expiresAtMs":999}}}"#,
        )
        .unwrap();

        assert_eq!(minimax_access_token_in(tmp.path()), None);
    }

    #[test]
    fn minimax_missing_root_is_none() {
        assert_eq!(
            minimax_access_token_in(Path::new("/nonexistent/minimax/auth")),
            None
        );
    }
}
