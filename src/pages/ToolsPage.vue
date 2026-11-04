<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import { useMessage } from "naive-ui";
import { api, type Connector, type ConnectorParam, type PreviewResult } from "@/api";
import { useAuthStore } from "@/stores/auth";

const message = useMessage();
const auth = useAuthStore();

/* ------------------------------------------------------------ 内置工具(只读清单) */

const BUILTIN_TOOLS = [
  { name: "current_time", label: "当前时间", desc: "获取当前日期与时间" },
  { name: "calculator", label: "计算器", desc: "四则运算与表达式求值" },
  { name: "date_calc", label: "日期推算", desc: "日期差/推算/星期计算" },
  { name: "timestamp_convert", label: "时间戳转换", desc: "Unix 时间戳与日期互转" },
  { name: "unit_convert", label: "单位换算", desc: "长度/温度/重量等常用单位" },
  { name: "tax_calc", label: "个税计算", desc: "工资薪金个人所得税测算" },
  { name: "web_fetch", label: "网页抓取", desc: "抓取网页正文(内网/公网 http/https)" },
  { name: "kb_stats", label: "知识库统计", desc: "当前库分块/索引规模(按权限过滤)" },
  { name: "kb_list", label: "可见知识库列表", desc: "列出调用者可访问的知识库" },
  { name: "doc_outline", label: "文档大纲", desc: "当前库文档的章节结构" },
  { name: "use_skill", label: "加载技能", desc: "按需拉取技能指令包(智能体编排页维护)" },
];

/* ------------------------------------------------------------ 连接器列表 */

const connectors = ref<Connector[]>([]);
const loading = ref(false);

async function load() {
  loading.value = true;
  try { connectors.value = (await api.connectors()).items; }
  catch (e) { message.error((e as Error).message); }
  finally { loading.value = false; }
}
onMounted(load);

async function toggleEnabled(c: Connector) {
  try {
    await api.updateConnector(c.id, { enabled: !c.enabled });
    await load();
    message.success(c.enabled ? "已停用,工具即时下线" : "已启用,自动注册为智能体工具");
  } catch (e) { message.error((e as Error).message); }
}

async function removeConnector(c: Connector) {
  if (!confirm(`确认删除连接器「${c.label || c.name}」?`)) return;
  try {
    await api.deleteConnector(c.id);
    await load();
    message.success("已删除");
  } catch (e) { message.error((e as Error).message); }
}

/* ------------------------------------------------------------ 编辑表单 */

interface FormState {
  id: number | null;
  name: string; label: string; description: string;
  type: "rest" | "database";
  method: string; url: string; headersText: string; bodyText: string; timeout: number;
  db_url: string; query: string; max_rows: number;
  params: ConnectorParam[];
  enabled: boolean;
}

const blank: FormState = {
  id: null, name: "", label: "", description: "", type: "rest",
  method: "GET", url: "", headersText: "", bodyText: "", timeout: 10,
  db_url: "", query: "", max_rows: 50, params: [], enabled: true,
};
const form = reactive<FormState>({ ...blank });
const showEditor = ref(false);
const saving = ref(false);

function headersToText(h?: Record<string, string>): string {
  return Object.entries(h || {}).map(([k, v]) => `${k}: ${v}`).join("\n");
}
function textToHeaders(t: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of t.split("\n")) {
    const i = line.indexOf(":");
    if (i > 0) out[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return out;
}

function openEditor(c?: Connector) {
  if (c) {
    const cfg = c.config || {};
    Object.assign(form, {
      id: c.id, name: c.name, label: c.label, description: c.description, type: c.type,
      method: cfg.method || "GET", url: cfg.url || "", headersText: headersToText(cfg.headers),
      bodyText: cfg.body ? JSON.stringify(cfg.body, null, 2) : "", timeout: cfg.timeout || 10,
      db_url: cfg.db_url || "", query: cfg.query || "", max_rows: cfg.max_rows || 50,
      params: (cfg.params || []).map((p) => ({ ...p })), enabled: c.enabled,
    });
  } else {
    Object.assign(form, { ...blank, params: [] });
  }
  testResult.value = null;
  showEditor.value = true;
}

function buildConfig(): Record<string, unknown> {
  if (form.type === "rest") {
    let body: unknown = undefined;
    if (form.method === "POST" && form.bodyText.trim()) {
      try { body = JSON.parse(form.bodyText); }
      catch { throw new Error("POST 请求体不是合法 JSON"); }
    }
    return {
      method: form.method, url: form.url.trim(), headers: textToHeaders(form.headersText),
      ...(body !== undefined ? { body } : {}),
      timeout: form.timeout, params: form.params,
    };
  }
  return { db_url: form.db_url.trim(), query: form.query.trim(), max_rows: form.max_rows, params: form.params };
}

async function saveConnector() {
  if (!/^[a-z][a-z0-9_]{0,31}$/.test(form.name.trim())) {
    message.warning("连接器标识需为小写字母开头的英文(工具名 biz_<标识>)");
    return;
  }
  saving.value = true;
  try {
    const config = buildConfig();
    if (form.id) {
      await api.updateConnector(form.id, {
        label: form.label.trim(), description: form.description.trim(),
        config, enabled: form.enabled,
      });
    } else {
      await api.createConnector({
        name: form.name.trim(), label: form.label.trim(), description: form.description.trim(),
        type: form.type, config, enabled: form.enabled,
      });
    }
    showEditor.value = false;
    await load();
    message.success("连接器已保存");
  } catch (e) { message.error((e as Error).message); }
  finally { saving.value = false; }
}

/* ------------------------------------------------------------ 参数编辑器 */

function addParam() {
  form.params.push({ name: "", label: "", type: "string", description: "", required: true, default: "" });
}
function removeParam(i: number) { form.params.splice(i, 1); }

/* ------------------------------------------------------------ 测试连接(预览数据) */

const testArgs = reactive<Record<string, string>>({});
const testing = ref(false);
const testResult = ref<PreviewResult | null>(null);

async function runTest() {
  testing.value = true;
  testResult.value = null;
  try {
    const config = buildConfig();
    if (form.id) {
      // 已保存的连接器:先落库再测,保证测的就是生效配置
      await api.updateConnector(form.id, {
        label: form.label.trim(), description: form.description.trim(),
        config, enabled: form.enabled,
      });
      testResult.value = await api.testConnector(form.id, { ...testArgs });
    } else {
      testResult.value = await api.previewConnector(form.type, config, { ...testArgs });
    }
    if (!testResult.value.ok) message.warning(testResult.value.error || "连接失败");
  } catch (e) { message.error((e as Error).message); }
  finally { testing.value = false; }
}

/** 已保存连接器的就地测试(列表行内) */
const rowTesting = ref<number | null>(null);
const rowResult = reactive<Record<number, PreviewResult>>({});
async function runRowTest(c: Connector) {
  rowTesting.value = c.id;
  try {
    const args: Record<string, string> = {};
    for (const p of c.config?.params || []) {
      if (!p.required) continue;
      const v = prompt(`参数 ${p.name}(${p.label || p.name})的值:`, String(p.default ?? ""));
      if (v === null) return;
      args[p.name] = v;
    }
    rowResult[c.id] = await api.testConnector(c.id, args);
    if (!rowResult[c.id].ok) message.warning(rowResult[c.id].error || "连接失败");
  } catch (e) { message.error((e as Error).message); }
  finally { rowTesting.value = null; }
}

/* ------------------------------------------------------------ MCP(仅系统管理员) */

const mcpServers = ref("");
const toolsEnabled = ref(true);
const toolsModel = ref("deepseek-chat");
const savingMcp = ref(false);
const mcpLoaded = ref(false);

async function loadMcp() {
  if (!auth.isAdmin) return;
  try {
    const s = await api.settings();
    mcpServers.value = s.mcp_servers || "";
    toolsEnabled.value = s.tools_enabled;
    toolsModel.value = s.tools_model || "deepseek-chat";
    mcpLoaded.value = true;
  } catch { /* 忽略 */ }
}
onMounted(loadMcp);

async function saveMcp() {
  if (mcpServers.value.trim()) {
    try { JSON.parse(mcpServers.value); }
    catch { message.warning("MCP servers 需为合法 JSON"); return; }
  }
  savingMcp.value = true;
  try {
    await api.saveSettings({ mcp_servers: mcpServers.value, tools_enabled: toolsEnabled.value, tools_model: toolsModel.value });
    message.success("工具运行配置已保存");
  } catch (e) { message.error((e as Error).message); }
  finally { savingMcp.value = false; }
}

const typeLabel = computed(() => form.type === "rest" ? "REST API" : "数据库(只读)");
</script>

<template>
  <div class="mx-auto max-w-[1180px] px-6 py-6">
    <div class="mb-5">
      <h1 class="text-[17px] font-semibold">工具管理</h1>
      <p class="mt-1 text-[12px] text-muted-foreground">对应规格书 §1.5:内置工具、业务系统数据对接(REST/数据库)、MCP 扩展,统一启停与调试;工具调用全程审计留痕</p>
    </div>

    <!-- 业务系统数据对接 -->
    <section class="mb-4 panel-shadow rounded-[12px] border border-border bg-card p-4">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-[13.5px] font-medium">业务系统数据对接</h2>
          <p class="mt-0.5 text-[11.5px] text-muted-foreground">
            配置 REST 接口或只读数据库查询;启用后自动注册为智能体工具(biz_*),参与意图识别与工具调用;数据库仅允许 SELECT。
          </p>
        </div>
        <button class="cursor-pointer rounded-[7px] bg-foreground px-3.5 py-1.5 text-[12px] text-primary-foreground transition-colors hover:bg-foreground/85"
                @click="openEditor()">+ 新增连接器</button>
      </div>
      <div class="mt-3 space-y-2">
        <div v-if="!connectors.length && !loading"
             class="rounded-[8px] border border-dashed border-border px-4 py-6 text-center text-[12px] text-muted-foreground">
          暂无连接器。示例:对接 MES 设备台账(SQLite/MySQL/PG 只读 SELECT)或 ERP 的 REST 接口。
        </div>
        <div v-for="c in connectors" :key="c.id" class="rounded-[9px] border border-border p-3">
          <div class="flex flex-wrap items-center gap-2">
            <span class="text-[13px] font-medium">{{ c.label || c.name }}</span>
            <span class="rounded bg-secondary px-1.5 py-0.5 font-mono-num text-[10.5px] text-muted-foreground">biz_{{ c.name }}</span>
            <span class="rounded px-1.5 py-0.5 text-[10.5px]"
                  :class="c.type === 'database' ? 'bg-[#4a6d8c]/10 text-[#41608c]' : 'bg-[#3f6f5f]/10 text-[#3f6f5f]'">
              {{ c.type === 'database' ? '数据库' : 'REST API' }}
            </span>
            <span class="text-[10.5px] text-muted-foreground">{{ c.config?.params?.length || 0 }} 个参数</span>
            <span class="ml-auto flex items-center gap-1.5">
              <button class="cursor-pointer rounded border border-border px-2 py-0.5 text-[11px] transition-colors hover:bg-secondary/70"
                      :disabled="rowTesting === c.id" @click="runRowTest(c)">
                {{ rowTesting === c.id ? '测试中…' : '测试连接' }}
              </button>
              <button class="cursor-pointer rounded border border-border px-2 py-0.5 text-[11px] transition-colors hover:bg-secondary/70"
                      @click="openEditor(c)">编辑</button>
              <button class="cursor-pointer rounded border border-border px-2 py-0.5 text-[11px] transition-colors hover:bg-destructive/10 hover:text-destructive"
                      @click="removeConnector(c)">删除</button>
              <label class="ml-1 flex cursor-pointer items-center gap-1 text-[11px] text-muted-foreground">
                <input type="checkbox" :checked="c.enabled" class="h-3.5 w-3.5 accent-[#3f6f5f]" @change="toggleEnabled(c)" />启用
              </label>
            </span>
          </div>
          <div class="mt-1 line-clamp-1 text-[11.5px] text-muted-foreground">{{ c.description || '—' }}</div>
          <!-- 行内测试结果 -->
          <div v-if="rowResult[c.id]" class="mt-2">
            <div v-if="rowResult[c.id].ok" class="overflow-x-auto rounded-[7px] border border-border">
              <table class="data-table">
                <thead><tr><th v-for="col in rowResult[c.id].columns" :key="col">{{ col }}</th></tr></thead>
                <tbody>
                  <tr v-for="(row, ri) in rowResult[c.id].rows" :key="ri">
                    <td v-for="(cell, ci) in row" :key="ci" class="font-mono-num text-[12px]">{{ cell }}</td>
                  </tr>
                </tbody>
              </table>
              <div v-if="rowResult[c.id].truncated" class="bg-secondary/50 px-3 py-1 text-[10.5px] text-muted-foreground">仅显示前 {{ rowResult[c.id].rows?.length }} 行</div>
            </div>
            <div v-else class="rounded-[7px] border border-destructive/40 bg-destructive/10 px-3 py-2 text-[12px] text-destructive">{{ rowResult[c.id].error }}</div>
          </div>
        </div>
      </div>
    </section>

    <div class="grid gap-4 lg:grid-cols-2">
      <!-- 内置工具 -->
      <section class="panel-shadow rounded-[12px] border border-border bg-card p-4">
        <h2 class="text-[13.5px] font-medium">内置工具</h2>
        <p class="mt-0.5 text-[11.5px] text-muted-foreground">平台预置,默认可用;知识库类工具按调用者权限过滤数据。</p>
        <div class="mt-3 grid grid-cols-1 gap-1.5 md:grid-cols-2">
          <div v-for="t in BUILTIN_TOOLS" :key="t.name" class="rounded-[8px] border border-border px-3 py-2">
            <div class="flex items-center justify-between">
              <span class="text-[12.5px] font-medium">{{ t.label }}</span>
              <span class="font-mono-num text-[10px] text-muted-foreground">{{ t.name }}</span>
            </div>
            <div class="mt-0.5 text-[11px] leading-4 text-muted-foreground">{{ t.desc }}</div>
          </div>
        </div>
      </section>

      <!-- MCP 扩展(仅系统管理员可改) -->
      <section v-if="auth.isAdmin" class="panel-shadow rounded-[12px] border border-border bg-card p-4">
        <h2 class="text-[13.5px] font-medium">MCP 扩展与工具开关</h2>
        <p class="mt-0.5 text-[11.5px] text-muted-foreground">接入 MCP server 暴露的外部工具;未配置时不启用。工具总开关关闭后,工具类意图整体停用。</p>
        <div class="mt-3 space-y-3 text-[12.5px]">
          <label class="flex cursor-pointer items-center justify-between">
            <span>工具调用总开关</span>
            <input type="checkbox" v-model="toolsEnabled" class="h-4 w-4 accent-[#3f6f5f]" />
          </label>
          <label class="block">
            <span class="mb-1 block text-[11.5px] text-muted-foreground">工具调用模型(工具选择/总结)</span>
            <input v-model="toolsModel"
                   class="h-8 w-full rounded-[7px] border border-input bg-background px-2.5 font-mono-num text-[12.5px] outline-none focus:border-foreground/50" />
          </label>
          <label class="block">
            <span class="mb-1 block text-[11.5px] text-muted-foreground">MCP servers(JSON:{'{'}名称: {'{'}url 或 command{'}'}{'}'})</span>
            <textarea v-model="mcpServers" rows="4" spellcheck="false"
                      placeholder='{"example": {"url": "http://192.168.1.10:8000/sse"}}'
                      class="w-full resize-none rounded-[7px] border border-input bg-background px-2.5 py-2 font-mono-num text-[11.5px] leading-5 outline-none focus:border-foreground/50" />
          </label>
          <button :disabled="savingMcp || !mcpLoaded"
                  class="w-full cursor-pointer rounded-[8px] bg-foreground py-2 text-[12.5px] text-primary-foreground transition-colors hover:bg-foreground/85 disabled:opacity-50"
                  @click="saveMcp">{{ savingMcp ? '保存中…' : '保存工具配置' }}</button>
        </div>
      </section>
    </div>

    <!-- 连接器编辑弹窗 -->
    <div v-if="showEditor" class="fixed inset-0 z-40 flex items-center justify-center bg-black/40 p-4"
         @click.self="showEditor = false">
      <div class="thin-scroll max-h-[90vh] w-full max-w-[720px] overflow-y-auto rounded-[12px] border border-border bg-card p-5 shadow-lg">
        <div class="flex items-center justify-between">
          <h3 class="text-[14.5px] font-medium">{{ form.id ? `编辑连接器「${form.label || form.name}」` : '新增连接器' }}</h3>
          <button class="cursor-pointer text-muted-foreground hover:text-foreground" @click="showEditor = false">×</button>
        </div>

        <div class="mt-4 space-y-3.5 text-[12.5px]">
          <div class="grid grid-cols-2 gap-3">
            <label class="block">
              <span class="mb-1 block text-[11.5px] text-muted-foreground">标识(英文,工具名 biz_&lt;标识&gt;)</span>
              <input v-model="form.name" :disabled="!!form.id" placeholder="如 mes_equipment"
                     class="h-8 w-full rounded-[7px] border border-input bg-background px-2.5 font-mono-num outline-none focus:border-foreground/50 disabled:opacity-50" />
            </label>
            <label class="block">
              <span class="mb-1 block text-[11.5px] text-muted-foreground">展示名</span>
              <input v-model="form.label" placeholder="如 设备台账查询(MES)"
                     class="h-8 w-full rounded-[7px] border border-input bg-background px-2.5 outline-none focus:border-foreground/50" />
            </label>
          </div>
          <label class="block">
            <span class="mb-1 block text-[11.5px] text-muted-foreground">用途描述(供模型判断何时调用,建议写清数据范围)</span>
            <textarea v-model="form.description" rows="2" placeholder="查询生产设备的台账信息,回答设备运行状态类问题"
                      class="w-full resize-none rounded-[7px] border border-input bg-background px-2.5 py-2 text-[12px] leading-5 outline-none focus:border-foreground/50" />
          </label>

          <div>
            <span class="mb-1.5 block text-[11.5px] text-muted-foreground">对接类型</span>
            <div class="grid grid-cols-2 gap-1.5">
              <button class="cursor-pointer rounded-[7px] border px-3 py-2 text-left transition-colors"
                      :class="form.type === 'rest' ? 'border-foreground/50 bg-secondary/70' : 'border-border'"
                      @click="form.type = 'rest'">
                <div class="text-[12.5px] font-medium">REST API</div>
                <div class="mt-0.5 text-[10.5px] text-muted-foreground">GET/POST,URL/请求体支持 {'{参数}'} 占位</div>
              </button>
              <button class="cursor-pointer rounded-[7px] border px-3 py-2 text-left transition-colors"
                      :class="form.type === 'database' ? 'border-foreground/50 bg-secondary/70' : 'border-border'"
                      @click="form.type = 'database'">
                <div class="text-[12.5px] font-medium">数据库(只读)</div>
                <div class="mt-0.5 text-[10.5px] text-muted-foreground">SQLite/MySQL/PG,仅 SELECT,强制行数上限</div>
              </button>
            </div>
          </div>

          <!-- REST 配置 -->
          <template v-if="form.type === 'rest'">
            <div class="flex gap-2">
              <select v-model="form.method" class="h-8 rounded-[7px] border border-input bg-background px-2 text-[12px] outline-none">
                <option>GET</option><option>POST</option>
              </select>
              <input v-model="form.url" placeholder="http://mes.internal/api/equipment?ws={ws}"
                     class="h-8 flex-1 rounded-[7px] border border-input bg-background px-2.5 font-mono-num text-[12px] outline-none focus:border-foreground/50" />
            </div>
            <label class="block">
              <span class="mb-1 block text-[11.5px] text-muted-foreground">请求头(每行一个 Key: Value,值支持 {'{参数}'} 占位)</span>
              <textarea v-model="form.headersText" rows="2" spellcheck="false" placeholder="Authorization: Bearer xxx"
                        class="w-full resize-none rounded-[7px] border border-input bg-background px-2.5 py-2 font-mono-num text-[11.5px] leading-5 outline-none focus:border-foreground/50" />
            </label>
            <label v-if="form.method === 'POST'" class="block">
              <span class="mb-1 block text-[11.5px] text-muted-foreground">POST 请求体(JSON,字符串值支持 {'{参数}'} 占位)</span>
              <textarea v-model="form.bodyText" rows="3" spellcheck="false" placeholder='{"workshop": "{ws}"}'
                        class="w-full resize-none rounded-[7px] border border-input bg-background px-2.5 py-2 font-mono-num text-[11.5px] leading-5 outline-none focus:border-foreground/50" />
            </label>
          </template>

          <!-- 数据库配置 -->
          <template v-else>
            <label class="block">
              <span class="mb-1 block text-[11.5px] text-muted-foreground">连接串(sqlite:/// / mysql+pymysql:// / postgresql://,建议使用只读账号)</span>
              <input v-model="form.db_url" placeholder="mysql+pymysql://reader:***@192.168.1.10:3306/mes"
                     class="h-8 w-full rounded-[7px] border border-input bg-background px-2.5 font-mono-num text-[12px] outline-none focus:border-foreground/50" />
            </label>
            <label class="block">
              <span class="mb-1 block text-[11.5px] text-muted-foreground">查询 SQL(仅 SELECT;命名参数 :参数名)</span>
              <textarea v-model="form.query" rows="3" spellcheck="false"
                        placeholder="SELECT asset_no,name,workshop,status FROM equipment_ledger WHERE (:ws = '' OR workshop = :ws)"
                        class="w-full resize-none rounded-[7px] border border-input bg-background px-2.5 py-2 font-mono-num text-[11.5px] leading-5 outline-none focus:border-foreground/50" />
            </label>
            <label class="block w-40">
              <span class="mb-1 block text-[11.5px] text-muted-foreground">行数上限(≤200)</span>
              <input v-model.number="form.max_rows" type="number" min="1" max="200"
                     class="h-8 w-full rounded-[7px] border border-input bg-background px-2.5 font-mono-num outline-none focus:border-foreground/50" />
            </label>
          </template>

          <!-- 参数定义 -->
          <div>
            <div class="mb-1.5 flex items-center justify-between">
              <span class="text-[11.5px] text-muted-foreground">参数定义(模型按此传参;URL/SQL 中用 {'{参数}'} / :参数名 引用)</span>
              <button class="cursor-pointer rounded border border-border px-2 py-0.5 text-[11px] transition-colors hover:bg-secondary/70" @click="addParam">+ 参数</button>
            </div>
            <div v-for="(p, i) in form.params" :key="i" class="mb-1.5 flex flex-wrap items-center gap-1.5">
              <input v-model="p.name" placeholder="参数名" class="h-7.5 h-8 w-28 rounded-[6px] border border-input bg-background px-2 font-mono-num text-[11.5px] outline-none" />
              <input v-model="p.label" placeholder="展示名" class="h-8 w-24 rounded-[6px] border border-input bg-background px-2 text-[11.5px] outline-none" />
              <select v-model="p.type" class="h-8 rounded-[6px] border border-input bg-background px-1.5 text-[11.5px] outline-none">
                <option value="string">文本</option><option value="integer">整数</option>
                <option value="number">数值</option><option value="boolean">布尔</option>
              </select>
              <label class="flex items-center gap-1 text-[11px] text-muted-foreground">
                <input type="checkbox" v-model="p.required" class="h-3.5 w-3.5 accent-[#3f6f5f]" />必填
              </label>
              <input v-model="p.description" placeholder="参数说明(供模型理解)" class="h-8 min-w-40 flex-1 rounded-[6px] border border-input bg-background px-2 text-[11.5px] outline-none" />
              <button class="cursor-pointer rounded border border-border px-1.5 text-[11px] text-muted-foreground hover:bg-destructive/10 hover:text-destructive" @click="removeParam(i)">×</button>
            </div>
          </div>

          <!-- 测试连接 -->
          <div class="rounded-[9px] border border-border bg-secondary/30 p-3">
            <div class="flex items-center justify-between">
              <span class="text-[12px] font-medium">测试连接({{ typeLabel }})</span>
              <button :disabled="testing"
                      class="cursor-pointer rounded-[7px] bg-[#3f6f5f] px-3.5 py-1.5 text-[12px] text-white transition-colors hover:bg-[#3f6f5f]/85 disabled:opacity-50"
                      @click="runTest">{{ testing ? '测试中…' : '▶ 测试连接' }}</button>
            </div>
            <div v-if="form.params.length" class="mt-2 flex flex-wrap gap-2">
              <label v-for="p in form.params" :key="p.name" class="flex items-center gap-1.5 text-[11.5px] text-muted-foreground">
                {{ p.label || p.name }}
                <input v-model="testArgs[p.name]" :placeholder="String(p.default ?? '')"
                       class="h-7.5 h-8 w-32 rounded-[6px] border border-input bg-background px-2 font-mono-num text-[11.5px] outline-none" />
              </label>
            </div>
            <div v-if="testResult" class="mt-2.5">
              <div v-if="testResult.ok">
                <div class="mb-1 text-[11px] text-muted-foreground">连接成功 · {{ testResult.duration_ms }}ms · {{ testResult.rows?.length }} 行<span v-if="testResult.truncated">(已截断)</span></div>
                <div class="overflow-x-auto rounded-[7px] border border-border bg-card">
                  <table class="data-table">
                    <thead><tr><th v-for="col in testResult.columns" :key="col">{{ col }}</th></tr></thead>
                    <tbody>
                      <tr v-for="(row, ri) in testResult.rows" :key="ri">
                        <td v-for="(cell, ci) in row" :key="ci" class="font-mono-num text-[12px]">{{ cell }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
              <div v-else class="rounded-[7px] border border-destructive/40 bg-destructive/10 px-3 py-2 text-[12px] text-destructive">
                {{ testResult.error }}
              </div>
            </div>
          </div>

          <div class="flex items-center justify-between border-t border-border pt-3">
            <label class="flex cursor-pointer items-center gap-1.5 text-[12px] text-muted-foreground">
              <input type="checkbox" v-model="form.enabled" class="h-4 w-4 accent-[#3f6f5f]" />启用(停用后工具即时下线)
            </label>
            <button :disabled="saving"
                    class="cursor-pointer rounded-[8px] bg-foreground px-5 py-2 text-[13px] text-primary-foreground transition-colors hover:bg-foreground/85 disabled:opacity-50"
                    @click="saveConnector">{{ saving ? '保存中…' : '保存' }}</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
