<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref, watch } from "vue";
import { useMessage } from "naive-ui";
import MarkdownIt from "markdown-it";
import {
  api, fetchKbImageURL, streamChat,
  type ChatMessage, type KBItem, type SessionItem, type SourceHit, type ToolCallInfo,
} from "@/api";

const message = useMessage();
const md = new MarkdownIt({ linkify: true, breaks: false });

/* ------------------------------------------------------------ 静态文案 */

/** 示例问题(语料对齐内置文档) */
const SAMPLES: Record<number, string[]> = {
  1: ["这个知识库主要包含哪些内容？", "违规处理的申诉流程是什么？", "查一下二车间的注塑机设备台账", "现在几点了？"],
  2: ["这份文档的核心要点是什么？", "文档里包含哪些操作流程？", "文档的章节结构是怎样的？"],
};
const FALLBACK_SAMPLES = ["你好，介绍一下你能做什么", "这个知识库主要包含哪些内容？", "现在几点了？"];

const INTENT_STYLE: Record<string, { label: string; cls: string }> = {
  kb_query: { label: "知识库问答", cls: "bg-[#3f6f5f]/10 text-[#3f6f5f]" },
  chitchat: { label: "闲聊", cls: "bg-foreground/8 text-muted-foreground" },
  out_of_scope: { label: "超出范围", cls: "bg-[#c9a227]/12 text-[#7a621c]" },
  tool: { label: "工具调用", cls: "bg-[#4a6d8c]/10 text-[#41608c]" },
};
const INTENT_SOURCE_TIP: Record<string, string> = {
  semantic: "语义路由命中(示例句向量匹配,未调用LLM)",
  rule: "规则层命中(问候/敏感词等关键词规则,零开销)",
  llm: "LLM 分类器判定(JSON 结构化输出)",
};
const TOOL_LABELS: Record<string, string> = {
  current_time: "当前时间", calculator: "计算器", web_fetch: "网页抓取",
  date_calc: "日期推算", timestamp_convert: "时间戳转换", unit_convert: "单位换算",
  tax_calc: "个税计算", kb_stats: "知识库统计", kb_list: "可见知识库列表",
  doc_outline: "文档大纲", use_skill: "加载技能",
};
function toolLabel(name: string): string {
  if (TOOL_LABELS[name]) return TOOL_LABELS[name];
  if (name.startsWith("biz_")) return `业务数据查询 · ${name.slice(4)}`;
  return name;
}

/* ------------------------------------------------------------ 状态 */

const kbs = ref<KBItem[]>([]);
const kbId = ref<number>(0);
const sessions = ref<SessionItem[]>([]);
const sessionId = ref<string | null>(null);
const messages = ref<ChatMessage[]>([]);
const memoryTurns = ref(0);
const input = ref("");
const streaming = ref(false);
const activeSources = ref<SourceHit[] | null>(null);
const activeSourceIdx = ref(0);
const bigImage = ref<string | null>(null);
const msgListEl = ref<HTMLElement | null>(null);

const currentKb = computed(() => kbs.value.find((k) => k.id === kbId.value));
const kbReady = computed(() => !!currentKb.value?.index?.indexed);
const samples = computed(() => SAMPLES[kbId.value] || FALLBACK_SAMPLES);
const imgCache = reactive(new Map<string, string>());

async function scrollBottom() {
  await nextTick();
  msgListEl.value?.scrollTo({ top: msgListEl.value.scrollHeight });
}

/* ------------------------------------------------------------ 数据加载 */

async function loadKbs() {
  try {
    const r = await api.kbs();
    kbs.value = r.kbs;
    if (r.kbs.length && !r.kbs.find((k) => k.id === kbId.value)) {
      kbId.value = r.kbs[0].id;
    }
  } catch (e) {
    message.error((e as Error).message);
  }
}

async function loadSessions() {
  if (!kbId.value) return;
  try {
    sessions.value = (await api.sessions(kbId.value)).sessions;
  } catch { sessions.value = []; }
}

async function openSession(sid: string) {
  try {
    const r = await api.history(kbId.value, sid);
    sessionId.value = sid;
    messages.value = r.messages;
    memoryTurns.value = r.turns ?? Math.ceil(r.messages.length / 2);
    activeSources.value = null;
    scrollBottom();
  } catch (e) {
    message.error((e as Error).message);
  }
}

function newSession() {
  sessionId.value = null;
  messages.value = [];
  memoryTurns.value = 0;
  activeSources.value = null;
}

async function removeSession(sid: string) {
  try {
    await api.deleteSession(kbId.value, sid);
    if (sessionId.value === sid) newSession();
    loadSessions();
  } catch (e) {
    message.error((e as Error).message);
  }
}

watch(kbId, () => { newSession(); loadSessions(); });
onMounted(loadKbs);

/* ------------------------------------------------------------ 发送 */

async function send(text?: string) {
  const q = (text ?? input.value).trim();
  if (!q || streaming.value || !kbReady.value) return;
  input.value = "";
  streaming.value = true;
  const historyIdx = messages.value.length;
  messages.value.push({ role: "user", content: q });
  const assistant = reactive<ChatMessage>({
    role: "assistant", content: "", thinking: "", toolCalls: [],
  });
  messages.value.push(assistant);
  scrollBottom();

  await streamChat(q, kbId.value, sessionId.value, {
    onToken: (t) => { assistant.content += t; scrollBottom(); },
    onThinking: (t) => { assistant.thinking = (assistant.thinking || "") + t; },
    onToolCall: (info) => {
      // 同名 running 记录就地更新为终态(与后端 writer 的二次发出配套)
      const list = assistant.toolCalls!;
      const running = [...list].reverse().find((c) => c.name === info.name && c.status === "running");
      if (running && info.status !== "running") {
        running.status = info.status;
        running.preview = info.preview;
      } else {
        list.push({ name: info.name, args: info.args, status: info.status, preview: info.preview });
      }
      scrollBottom();
    },
    onDone: (ev) => {
      assistant.content = ev.answer || assistant.content;
      assistant.sources = ev.sources;
      assistant.intent = ev.intent;
      assistant.intent_label = ev.intent_label;
      assistant.intent_score = ev.intent_score;
      assistant.intent_source = ev.intent_source;
      assistant.route_name = ev.route_name;
      assistant.usage = ev.usage;
      assistant.latency_ms = ev.latency_ms;
      sessionId.value = ev.session_id;
      memoryTurns.value = ev.memory_turns ?? memoryTurns.value;
      streaming.value = false;
      loadSessions();
      scrollBottom();
    },
    onError: (err) => {
      streaming.value = false;
      messages.value.splice(historyIdx, 2);
      message.error(err);
    },
  });
  streaming.value = false;
}

/* ------------------------------------------------------------ 引用与图片 */

/** [n] 引用渲染为可点徽标,点击定位来源面板;越界编号原样输出。 */
function renderAnswer(m: ChatMessage): string {
  const html = md.render(m.content || "");
  const n = m.sources?.length ?? 0;
  return html.replace(/\[(\d{1,2})\]/g, (raw, num) => {
    const i = Number(num);
    if (i < 1 || i > n) return raw;
    return `<button type="button" class="cite-badge" data-cite="${i - 1}">[${i}]</button>`;
  });
}

function onAnswerClick(e: MouseEvent, m: ChatMessage) {
  const el = (e.target as HTMLElement).closest("[data-cite]") as HTMLElement | null;
  if (!el || !m.sources) return;
  activeSources.value = m.sources;
  activeSourceIdx.value = Number(el.dataset.cite || 0);
}

function openSources(m: ChatMessage, idx: number) {
  activeSources.value = m.sources || null;
  activeSourceIdx.value = idx;
}

async function imgUrl(imgId: string): Promise<string> {
  if (imgCache.has(imgId)) return imgCache.get(imgId)!;
  const url = await fetchKbImageURL(kbId.value, imgId);
  imgCache.set(imgId, url);
  return url;
}

const imgLoading = reactive(new Set<string>());
function ensureImage(imgId: string) {
  if (imgCache.has(imgId) || imgLoading.has(imgId)) return;
  imgLoading.add(imgId);
  imgUrl(imgId).catch(() => message.error("图片加载失败"))
    .finally(() => imgLoading.delete(imgId));
}

async function openBig(imgId: string) {
  try { bigImage.value = await imgUrl(imgId); } catch { message.error("图片加载失败"); }
}

function fmtTime(ts: number): string {
  const d = new Date(ts * 1000);
  return `${d.getMonth() + 1}-${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}
</script>

<template>
  <div class="flex h-full min-h-0">
    <!-- 会话侧栏 -->
    <div class="flex w-[220px] shrink-0 flex-col border-r border-border bg-card/60">
      <div class="border-b border-border p-3">
        <label class="mb-1.5 block text-[11.5px] text-muted-foreground">当前知识库</label>
        <select v-model.number="kbId"
                class="h-9 w-full rounded-[7px] border border-input bg-background px-2 text-[13px] outline-none focus:border-foreground/50">
          <option v-for="k in kbs" :key="k.id" :value="k.id">{{ k.name }}</option>
        </select>
        <button class="mt-2.5 h-8 w-full cursor-pointer rounded-[7px] bg-foreground text-[12.5px] text-primary-foreground transition-colors hover:bg-foreground/85"
                @click="newSession">+ 新会话</button>
      </div>
      <div class="thin-scroll flex-1 overflow-y-auto p-2">
        <div v-if="!sessions.length" class="px-2 py-6 text-center text-[11.5px] text-muted-foreground">暂无历史会话</div>
        <div v-for="s in sessions" :key="s.session_id"
             class="group mb-1 flex cursor-pointer items-center gap-1.5 rounded-[7px] px-2.5 py-2 transition-colors hover:bg-secondary/70"
             :class="{ 'bg-secondary': s.session_id === sessionId }"
             @click="openSession(s.session_id)">
          <div class="min-w-0 flex-1">
            <div class="truncate text-[12.5px]">{{ s.title || "未命名会话" }}</div>
            <div class="mt-0.5 text-[10.5px] text-muted-foreground">{{ fmtTime(s.updated_at) }} · {{ s.turns }} 轮</div>
          </div>
          <button class="hidden h-5 w-5 shrink-0 cursor-pointer items-center justify-center rounded text-muted-foreground hover:bg-destructive/10 hover:text-destructive group-hover:flex"
                  title="删除会话" @click.stop="removeSession(s.session_id)">×</button>
        </div>
      </div>
    </div>

    <!-- 聊天主区 -->
    <div class="flex min-w-0 flex-1 flex-col">
      <div class="flex items-center justify-between border-b border-border px-5 py-3">
        <div>
          <div class="text-[14.5px] font-medium">智能问答</div>
          <div class="mt-0.5 text-[11.5px] text-muted-foreground">
            {{ currentKb?.name || "—" }} · 三级意图识别 · 来源可追溯
          </div>
        </div>
        <div class="flex items-center gap-2 text-[11px] text-muted-foreground">
          <span v-if="currentKb && !kbReady" class="rounded-full bg-[#c9a227]/12 px-2 py-0.5 text-[#7a621c]">索引未构建</span>
          <span class="rounded-full bg-secondary px-2 py-0.5 font-mono-num">短时记忆 {{ memoryTurns }}/50 轮</span>
        </div>
      </div>

      <div ref="msgListEl" class="thin-scroll flex-1 overflow-y-auto px-5 py-4">
        <!-- 空态:示例问题 -->
        <div v-if="!messages.length" class="mx-auto mt-16 max-w-[640px]">
          <div class="mx-auto flex h-12 w-12 items-center justify-center rounded-[14px] brand-mark">
            <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" class="h-5.5 w-5.5" style="width:22px;height:22px">
              <path d="M8 10h8M8 14h5M21 12a9 9 0 1 1-4.4-7.7" />
            </svg>
          </div>
          <div class="mt-4 text-center text-[17px] font-semibold tracking-wide">向「{{ currentKb?.name || "知识库" }}」提问</div>
          <div class="mt-1.5 text-center text-[12.5px] text-muted-foreground">
            支持知识问答、文档溯源、工具调用与业务系统数据查询
          </div>
          <div class="mt-7 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
            <button v-for="s in samples" :key="s"
                    class="card-hover group flex cursor-pointer items-center justify-between gap-2 rounded-[12px] border border-border bg-card px-4 py-3 text-left text-[13px] leading-5"
                    @click="send(s)">
              <span>{{ s }}</span>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"
                   class="h-3.5 w-3.5 shrink-0 text-muted-foreground/50 transition-all group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-[#3f6f5f]">
                <path d="M7 17 17 7M8 7h9v9" />
              </svg>
            </button>
          </div>
        </div>

        <!-- 消息列表 -->
        <div v-for="(m, mi) in messages" :key="mi" class="mx-auto mb-4 max-w-[860px]">
          <div v-if="m.role === 'user'" class="flex justify-end">
            <div class="max-w-[75%] whitespace-pre-wrap rounded-[12px] rounded-br-[4px] bg-foreground px-3.5 py-2.5 text-[14px] leading-6 text-primary-foreground shadow-[0_4px_12px_-4px_rgb(0_0_0/0.35)]">{{ m.content }}</div>
          </div>
          <div v-else class="rise-in panel-shadow rounded-[12px] rounded-bl-[4px] border border-border bg-card px-4 py-3">
            <!-- 意图徽标 -->
            <div class="mb-2 flex flex-wrap items-center gap-1.5">
              <span v-if="m.intent"
                    class="rounded-full px-2 py-0.5 text-[10.5px] font-medium"
                    :class="(INTENT_STYLE[m.intent] || INTENT_STYLE.kb_query).cls"
                    :title="(INTENT_SOURCE_TIP[m.intent_source || ''] || '意图识别') + (m.route_name && m.route_name !== m.intent ? `,路由名: ${m.route_name}` : '')">
                意图 · {{ m.intent_label || (INTENT_STYLE[m.intent] || {}).label || m.intent }}{{ m.intent_source === 'semantic' && m.intent_score ? ` ${(m.intent_score * 100).toFixed(0)}%` : '' }}{{ m.intent_source === 'llm' ? ' · LLM' : '' }}{{ m.intent_source === 'rule' ? ' · 规则' : '' }}
              </span>
              <span v-if="m.latency_ms" class="rounded-full bg-secondary px-2 py-0.5 font-mono-num text-[10.5px] text-muted-foreground">{{ (m.latency_ms / 1000).toFixed(1) }}s</span>
              <span v-if="m.usage?.total_tokens" class="rounded-full bg-secondary px-2 py-0.5 font-mono-num text-[10.5px] text-muted-foreground">{{ m.usage.total_tokens }} tokens</span>
            </div>
            <!-- 思考过程 -->
            <details v-if="m.thinking" class="mb-2 rounded-[8px] border border-border bg-secondary/40 px-3 py-2">
              <summary class="cursor-pointer text-[11.5px] text-muted-foreground">推理过程(点击展开)</summary>
              <div class="mt-2 whitespace-pre-wrap text-[12px] leading-5 text-muted-foreground">{{ m.thinking }}</div>
            </details>
            <!-- 工具调用 -->
            <div v-for="(tc, ti) in m.toolCalls" :key="ti"
                 class="mb-2 rounded-[8px] border border-border bg-secondary/40 px-3 py-2">
              <div class="flex items-center gap-2 text-[12px]">
                <span class="inline-block h-1.5 w-1.5 rounded-full"
                      :class="tc.status === 'running' ? 'animate-pulse bg-[#c9a227]' : tc.status === 'error' ? 'bg-[#b4413c]' : 'bg-[#3f6f5f]'" />
                <span class="font-medium">{{ toolLabel(tc.name) }}</span>
                <span class="font-mono-num text-[10.5px] text-muted-foreground">{{ tc.status === 'running' ? '调用中…' : tc.status === 'error' ? '失败' : '完成' }}</span>
              </div>
              <div class="mt-1 font-mono-num text-[10.5px] text-muted-foreground">{{ JSON.stringify(tc.args) }}</div>
              <details v-if="tc.preview && tc.status !== 'running'" class="mt-1.5">
                <summary class="cursor-pointer text-[11px] text-muted-foreground">工具返回(点击展开)</summary>
                <pre class="md-dark mt-1.5 overflow-x-auto rounded-[6px] bg-[#161616] p-2.5 text-[11px] leading-5 text-white/85">{{ tc.preview }}</pre>
              </details>
            </div>
            <!-- 正文 -->
            <div class="md-body" @click="onAnswerClick($event, m)" v-html="renderAnswer(m)" />
            <span v-if="streaming && mi === messages.length - 1 && !m.content" class="text-[12px] text-muted-foreground">正在生成…</span>
            <!-- 来源 -->
            <div v-if="m.sources?.length" class="mt-3 border-t border-border pt-2.5">
              <div class="mb-1.5 text-[11px] text-muted-foreground">知识来源(点击可查看原文段落)</div>
              <div class="flex flex-wrap gap-1.5">
                <button v-for="(s, si) in m.sources" :key="si"
                        class="cursor-pointer rounded-full border border-border px-2 py-0.5 text-[11px] text-muted-foreground transition-colors hover:bg-secondary"
                        @click="openSources(m, si)">
                  [{{ si + 1 }}] {{ s.meta.chapter || s.meta.article || '正文' }} · 第{{ s.meta.page_start ?? '?' }}页
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 输入区 -->
      <div class="border-t border-border px-5 py-3">
        <div class="mx-auto flex max-w-[860px] items-end gap-2">
          <textarea v-model="input" rows="2" :disabled="!kbReady || streaming"
                    :placeholder="kbReady ? `向「${currentKb?.name || '知识库'}」提问,Enter 发送,Shift+Enter 换行` : '知识库未就绪,暂不可提问'"
                    class="flex-1 resize-none rounded-[10px] border border-input bg-card px-3 py-2.5 text-[14px] leading-6 outline-none transition-colors placeholder:text-muted-foreground/70 focus:border-foreground/50 disabled:opacity-50"
                    @keydown.enter.exact.prevent="send()" />
          <button :disabled="!kbReady || streaming || !input.trim()" aria-label="发送"
                  class="flex h-11 w-11 cursor-pointer items-center justify-center rounded-[10px] bg-foreground text-primary-foreground transition-colors hover:bg-foreground/85 disabled:cursor-not-allowed disabled:opacity-40"
                  @click="send()">
            <svg v-if="!streaming" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="h-4 w-4"><path d="M22 2 11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
            <span v-else class="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
          </button>
        </div>
      </div>
    </div>

    <!-- 来源面板 -->
    <div v-if="activeSources" class="flex w-[340px] shrink-0 flex-col border-l border-border bg-card">
      <div class="flex items-center justify-between border-b border-border px-4 py-3">
        <div class="text-[13px] font-medium">知识来源展示与原文追溯</div>
        <button class="cursor-pointer text-muted-foreground hover:text-foreground" @click="activeSources = null">×</button>
      </div>
      <div class="thin-scroll flex-1 space-y-3 overflow-y-auto p-3">
        <div v-for="(s, i) in activeSources" :key="i"
             class="rounded-[9px] border bg-background p-3 transition-colors"
             :class="i === activeSourceIdx ? 'border-foreground/40' : 'border-border'"
             @mouseenter="activeSourceIdx = i">
          <div class="flex items-center justify-between">
            <span class="rounded bg-foreground px-1.5 py-0.5 font-mono-num text-[10px] text-primary-foreground">[{{ i + 1 }}]</span>
            <span class="font-mono-num text-[10.5px] text-muted-foreground">score {{ (s.rrf_score ?? s.score).toFixed(3) }}</span>
          </div>
          <div class="mt-1.5 text-[11px] text-muted-foreground">
            {{ s.meta.chapter || '正文' }}<template v-if="s.meta.article"> · {{ s.meta.article }}</template> · 第{{ s.meta.page_start ?? '?' }}页
            <span v-if="s.meta.parent_id !== null && s.meta.parent_id !== undefined" class="ml-1 rounded bg-secondary px-1 py-px">父块</span>
            <span v-if="s.from" class="ml-1 rounded bg-secondary px-1 py-px font-mono-num">{{ s.from }}</span>
          </div>
          <div class="mt-2 line-clamp-6 whitespace-pre-wrap text-[12px] leading-5">{{ s.meta.parent_text || s.text }}</div>
          <!-- 原文图片 -->
          <div v-for="img in s.meta.image_ids || []" :key="img" class="mt-2">
            <template v-if="ensureImage(img), true">
              <img v-if="imgCache.has(img)" :src="imgCache.get(img)"
                   class="max-h-40 cursor-zoom-in rounded-[6px] border border-border" @click="openBig(img)" />
              <div v-else class="rounded-[6px] border border-dashed border-border px-3 py-2 text-[11px] text-muted-foreground">图片加载中…</div>
            </template>
          </div>
        </div>
      </div>
    </div>

    <!-- 大图预览 -->
    <div v-if="bigImage" class="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-8" @click="bigImage = null">
      <img :src="bigImage" class="max-h-full max-w-full rounded-[8px]" />
      <button class="absolute right-5 top-5 cursor-pointer text-[24px] text-white/80 hover:text-white" @click="bigImage = null">×</button>
    </div>
  </div>
</template>

<style scoped>
:deep(.cite-badge) {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 18px; height: 16px; padding: 0 3px; margin: 0 1px;
  border-radius: 4px; background: hsl(150 30% 34% / 0.12); color: hsl(150 30% 30%);
  font-size: 10.5px; font-weight: 600; cursor: pointer; vertical-align: 2px;
}
:deep(.cite-badge:hover) { background: hsl(150 30% 34% / 0.22); }
</style>
