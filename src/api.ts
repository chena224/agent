/** API 客户端:统一 Bearer 鉴权、错误规整、SSE 流式聊天。
 *  令牌存 localStorage(kb_token),与后端 /api/auth/* 配套。 */

export const TOKEN_KEY = "kb_token";

export interface UserInfo {
  id: number;
  username: string;
  display_name: string;
  department: string;
  role: "admin" | "workspace_admin" | "user";
  role_name?: string;
  status: number;
}

export interface IndexStatus { indexed: boolean; count: number; }

export interface KBItem {
  id: number;
  name: string;
  description: string;
  doc_name: string;
  visibility: "public" | "private";
  owner_id: number;
  owner_name?: string;
  strategy: string;
  chunk_count: number;
  built_at: number;
  retrieval: RetrievalConfig;
  can_manage?: boolean;
  index?: IndexStatus;
  has_document?: boolean;
  created_at: number;
}

export interface RetrievalConfig {
  mode: "semantic" | "fulltext" | "hybrid";
  top_k: number;
  score_threshold: number;
  bm25_weight: number;
  query_rewrite: boolean;
  rerank: boolean;
  route_threshold: number;
  oos_words: string[];
  image_caption: boolean;
}

export interface SourceHit {
  score: number;
  rrf_score?: number;
  from?: string;                    // vector | bm25 | vector+bm25
  text: string;
  kind: string;
  meta: {
    page_start?: number; page_end?: number;
    chapter?: string; article?: string;
    image_ids?: string[]; parent_id?: number | null; parent_text?: string;
  };
}

export interface ToolCallInfo {
  name: string;
  args: Record<string, unknown>;
  status: "running" | "ok" | "error";
  preview?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  toolCalls?: ToolCallInfo[];
  sources?: SourceHit[];
  intent?: string;
  intent_label?: string;
  intent_score?: number;
  intent_source?: string;
  route_name?: string;
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
  latency_ms?: number;
}

export interface SessionItem {
  session_id: string;
  title: string;
  turns: number;
  updated_at: number;
}

export interface IntentItem {
  id: number;
  kb_id: number;
  name: string;
  label: string;
  utterances: string[];
  boost_keywords: string[];
}

export interface SkillItem {
  name: string;
  description: string;
  enabled: boolean;
  size?: number;
}

export interface Settings {
  deepseek_api_key_masked: string;
  deepseek_api_key_set: boolean;
  deepseek_backup_api_key_masked: string;
  deepseek_backup_api_key_set: boolean;
  deepseek_base_url: string;
  deepseek_model: string;
  vision_model: string;
  embedding_model: string;
  memory_turns: number;
  retrieve_top_k: number;
  tools_enabled: boolean;
  tools_model: string;
  mcp_servers: string;
  skills_enabled: boolean;
}

export interface ConnectorParam {
  name: string; label: string; type: string;
  description: string; required: boolean; default?: unknown;
}

export interface Connector {
  id: number;
  name: string;
  label: string;
  description: string;
  type: "rest" | "database";
  config: {
    method?: string; url?: string; headers?: Record<string, string>;
    body?: unknown; timeout?: number;
    db_url?: string; query?: string; max_rows?: number;
    params?: ConnectorParam[];
  };
  enabled: boolean;
  created_by: string;
  created_at: number;
}

export interface PreviewResult {
  ok: boolean;
  duration_ms?: number;
  columns?: string[];
  rows?: string[][];
  truncated?: boolean;
  error?: string;
}

export interface ChunkItem {
  id: number;
  text: string;
  kind: string;
  page_start: number;
  page_end: number;
  chapter: string;
  article: string;
  length: number;
  parent_id: number | null;
  parent_text: string;
  image_ids: string[];
}

export interface BuildProgress { stage: string; percent: number; message: string; }

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}

async function req<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = localStorage.getItem(TOKEN_KEY) || "";
  const headers: Record<string, string> = { ...(options.headers as Record<string, string> || {}) };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (options.body && typeof options.body === "string") headers["Content-Type"] = "application/json";
  const resp = await fetch(path, { ...options, headers });
  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      msg = data.error || data.detail || msg;
    } catch { /* 非 JSON 错误体 */ }
    throw new ApiError(resp.status, msg);
  }
  return resp.json() as Promise<T>;
}

const get = <T>(path: string) => req<T>(path);
const post = <T>(path: string, body?: unknown) =>
  req<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
const patch = <T>(path: string, body: unknown) =>
  req<T>(path, { method: "PATCH", body: JSON.stringify(body) });
const del = <T>(path: string) => req<T>(path, { method: "DELETE" });

/** 知识库图片走 blob + 鉴权头(<img> 标签无法带 Authorization)。 */
export async function fetchKbImageURL(kbId: number, imgId: string): Promise<string> {
  const token = localStorage.getItem(TOKEN_KEY) || "";
  const resp = await fetch(`/api/kbs/${kbId}/images/${imgId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!resp.ok) throw new ApiError(resp.status, "图片加载失败");
  return URL.createObjectURL(await resp.blob());
}

export const api = {
  // 认证
  login: (username: string, password: string) =>
    post<{ token: string; user: UserInfo }>("/api/auth/login", { username, password }),
  logout: () => post<{ ok: boolean }>("/api/auth/logout"),
  me: () => get<{ user: UserInfo }>("/api/auth/me"),

  // 组织机构与权限(1.7)
  adminUsers: () => get<{ users: UserInfo[]; roles: Record<string, string>; departments: string[] }>("/api/admin/users"),
  createUser: (body: { username: string; password: string; display_name: string; department: string; role: string }) =>
    post<{ user: UserInfo }>("/api/admin/users", body),
  updateUser: (id: number, body: Partial<{ display_name: string; department: string; role: string; status: number; password: string }>) =>
    patch<{ user: UserInfo }>(`/api/admin/users/${id}`, body),
  departments: () => get<{ departments: string[] }>("/api/admin/departments"),

  // 知识库与文档解析(1.3)
  kbs: () => get<{ kbs: KBItem[] }>("/api/kbs"),
  createKb: (body: { name: string; description: string; visibility: string }) =>
    post<{ kb: KBItem }>("/api/kbs", body),
  kbDetail: (id: number) => get<KBItem>(`/api/kbs/${id}`),
  updateKb: (id: number, body: Record<string, unknown>) => patch<{ kb: KBItem }>(`/api/kbs/${id}`, body),
  deleteKb: (id: number) => del<{ ok: boolean }>(`/api/kbs/${id}`),
  uploadKbDoc: async (id: number, file: File) => {
    const token = localStorage.getItem(TOKEN_KEY) || "";
    const fd = new FormData();
    fd.append("file", file);
    const resp = await fetch(`/api/kbs/${id}/upload`, {
      method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd,
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new ApiError(resp.status, data.error || `HTTP ${resp.status}`);
    }
    return resp.json() as Promise<{ ok: boolean; name: string; page_count: number; table_count: number; image_count: number; doc_id: number }>;
  },
  previewChunks: (id: number, body: { strategy: string; params: Record<string, number>; offset?: number; limit?: number }) =>
    post<{ stats: Record<string, unknown>; total: number; chunks: ChunkItem[] }>(`/api/kbs/${id}/chunks/preview`, body),
  buildIndex: (id: number, body: { strategy: string; params: Record<string, number> }) =>
    post<{ ok: boolean; mode: string }>(`/api/kbs/${id}/index/build`, body),
  buildProgress: (id: number) => get<BuildProgress>(`/api/kbs/${id}/index/progress`),
  indexStatus: (id: number) => get<IndexStatus & { strategy?: string; built_at?: number }>(`/api/kbs/${id}/index/status`),
  clearIndex: (id: number) => del<{ ok: boolean }>(`/api/kbs/${id}/index`),
  retrieve: (id: number, query: string) =>
    post<{ hits: SourceHit[]; debug: Record<string, unknown> }>(`/api/kbs/${id}/retrieve`, { query }),

  // 授权
  grants: (kbId: number) => get<{ grants: { id: number; target_type: string; target_value: string; created_at: number }[] }>(`/api/kbs/${kbId}/grants`),
  addGrant: (kbId: number, body: { target_type: string; target_value: string }) =>
    post<{ grant: { id: number } }>(`/api/kbs/${kbId}/grants`, body),
  removeGrant: (kbId: number, gid: number) => del<{ ok: boolean }>(`/api/kbs/${kbId}/grants/${gid}`),

  // 智能体编排(1.4):意图识别路由
  intents: (kbId: number) => get<{ intents: IntentItem[] }>(`/api/kbs/${kbId}/intents`),
  createIntent: (kbId: number, body: { name: string; label: string; utterances: string[]; boost_keywords: string[] }) =>
    post<{ intent: IntentItem }>(`/api/kbs/${kbId}/intents`, body),
  updateIntent: (kbId: number, intentId: number, body: Partial<{ label: string; utterances: string[]; boost_keywords: string[] }>) =>
    patch<{ ok: boolean }>(`/api/kbs/${kbId}/intents/${intentId}`, body),
  deleteIntent: (kbId: number, intentId: number) => del<{ ok: boolean }>(`/api/kbs/${kbId}/intents/${intentId}`),

  // 会话
  sessions: (kbId: number) => get<{ sessions: SessionItem[] }>(`/api/chat/sessions?kb_id=${kbId}`),
  history: (kbId: number, sid: string) => get<{ messages: ChatMessage[]; turns?: number }>(`/api/chat/history/${kbId}/${sid}`),
  deleteSession: (kbId: number, sid: string) => del<{ ok: boolean }>(`/api/chat/sessions/${kbId}/${sid}`),

  // 统计分析(1.10)
  statsSummary: () => get<{ total_questions: number; total_tokens: number; avg_latency_ms: number; today_questions: number; today_tokens: number; sessions: number; builds: number }>("/api/stats/summary"),
  statsTimeseries: () => get<{ days: { date: string; questions: number; tokens: number }[] }>("/api/stats/timeseries"),
  statsRecent: () => get<{ items: { time: string; session_id: string; question: string; answer_preview: string; tokens: number; latency_ms: number; sources_n: number; username: string; kb_name: string; intent: string }[] }>("/api/stats/recent"),
  statsIntents: () => get<{ items: { intent: string; label?: string; count: number }[] }>("/api/stats/intents"),
  statsKbs: () => get<{ items: { kb_name: string; questions: number; tokens: number }[] }>("/api/stats/kbs"),

  // 系统管理(1.11)
  settings: () => get<Settings>("/api/settings"),
  saveSettings: (body: Partial<Settings> & { deepseek_api_key?: string; deepseek_backup_api_key?: string }) =>
    post<{ ok: boolean; deepseek_api_key_set: boolean }>("/api/settings", body),
  testSettings: () => post<{ ok: boolean; error?: string; latency_ms?: number }>("/api/settings/test"),

  // 技能(1.4 提示词技能)
  skills: () => get<{ items: SkillItem[] }>("/api/skills"),
  setSkillEnabled: (name: string, enabled: boolean) => patch<{ ok: boolean }>(`/api/skills/${name}`, { enabled }),
  deleteSkill: (name: string) => del<{ ok: boolean }>(`/api/skills/${name}`),
  uploadSkill: async (file: File) => {
    const token = localStorage.getItem(TOKEN_KEY) || "";
    const fd = new FormData();
    fd.append("file", file);
    const resp = await fetch("/api/skills", {
      method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd,
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new ApiError(resp.status, data.error || `HTTP ${resp.status}`);
    }
    return resp.json();
  },

  // 工具管理(1.5):业务系统数据对接
  connectors: () => get<{ items: Connector[] }>("/api/connectors"),
  createConnector: (body: { name: string; label: string; description: string; type: string; config: Record<string, unknown>; enabled: boolean }) =>
    post<Connector>("/api/connectors", body),
  updateConnector: (id: number, body: Partial<{ name: string; label: string; description: string; config: Record<string, unknown>; enabled: boolean }>) =>
    patch<Connector>(`/api/connectors/${id}`, body),
  deleteConnector: (id: number) => del<{ ok: boolean }>(`/api/connectors/${id}`),
  testConnector: (id: number, args: Record<string, unknown>) =>
    post<PreviewResult>(`/api/connectors/${id}/test`, { args }),
  previewConnector: (type: string, config: Record<string, unknown>, args: Record<string, unknown>) =>
    post<PreviewResult>("/api/connectors/preview", { type, config, args }),

  health: () => get<{ ok: boolean; llm_configured: boolean; llm_model: string; embedding_ready: boolean; kbs: number }>("/api/health"),
};

/* ---------------------------------------------------------------- SSE 聊天 */

export interface StreamHandlers {
  onToken?: (text: string) => void;
  onThinking?: (text: string) => void;
  onToolCall?: (info: ToolCallInfo) => void;
  onDone?: (msg: Partial<ChatMessage> & { session_id: string; memory_turns?: number }) => void;
  onError?: (message: string) => void;
}

/** SSE 流式聊天:fetch + ReadableStream(EventSource 无法携带鉴权头)。 */
export async function streamChat(
  message: string, kbId: number, sessionId: string | null, handlers: StreamHandlers,
): Promise<void> {
  const token = localStorage.getItem(TOKEN_KEY) || "";
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ message, kb_id: kbId, session_id: sessionId || undefined }),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    handlers.onError?.(data.error || `HTTP ${resp.status}`);
    return;
  }
  const reader = resp.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const rawEvent = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of rawEvent.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (ev.type === "token") handlers.onToken?.(ev.content ?? "");
          else if (ev.type === "thinking") handlers.onThinking?.(ev.content ?? "");
          else if (ev.type === "tool_call") handlers.onToolCall?.(ev as ToolCallInfo & { type: string });
          else if (ev.type === "done") handlers.onDone?.(ev);
          else if (ev.type === "error") handlers.onError?.(ev.error || "生成失败");
        } catch { /* 半包忽略,下轮续接 */ }
      }
    }
  }
}
