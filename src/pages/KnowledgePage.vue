<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useMessage } from "naive-ui";
import MarkdownIt from "markdown-it";
import { api, type ChunkItem, type KBItem, type RetrievalConfig } from "@/api";

const message = useMessage();
const md = new MarkdownIt();

/* ------------------------------------------------------------ 常量 */

const STRATEGY_KEYS = ["structural", "recursive", "fixed", "token", "parent_child", "semantic"] as const;
const STRATEGY_LABELS: Record<string, { name: string; desc: string }> = {
  structural: { name: "结构优先", desc: "按章节/条文边界切分,保留层级锚点(推荐制度类文档)" },
  recursive: { name: "递归切分", desc: "LangChain RecursiveCharacterTextSplitter,多分隔符回退" },
  fixed: { name: "固定窗口", desc: "定长滑窗,重叠区可调,最简基线" },
  token: { name: "Token 切分", desc: "按 tiktoken token 数定长,贴合模型上下文" },
  parent_child: { name: "父子块", desc: "子块检索、父块供上下文,兼顾精度与完整性" },
  semantic: { name: "语义切分", desc: "按嵌入相似度找语义断点(需向量模型就绪)" },
};
const DEFAULT_PARAMS: Record<string, Record<string, number>> = {
  structural: { max_len: 900 },
  recursive: { chunk_size: 500, overlap: 50 },
  fixed: { chunk_size: 500, overlap: 50 },
  token: { chunk_size: 400 },
  semantic: { percentile: 90, max_len: 600, min_len: 60 },
  parent_child: { parent_size: 1200, parent_overlap: 100, child_size: 300, child_overlap: 40 },
};
const PARAM_LABELS: Record<string, string> = {
  chunk_size: "块长度", overlap: "重叠区", max_len: "最大块长",
  percentile: "语义阈值(%)", min_len: "最小块长",
  parent_size: "父块长度", parent_overlap: "父块重叠", child_size: "子块长度", child_overlap: "子块重叠",
};
const PARAM_RANGE: Record<string, [number, number, number]> = {
  chunk_size: [100, 2000, 50], overlap: [0, 400, 10], max_len: [200, 2000, 50],
  percentile: [50, 99, 1], min_len: [20, 400, 10],
  parent_size: [400, 3000, 100], parent_overlap: [0, 400, 20],
  child_size: [100, 800, 20], child_overlap: [0, 200, 10],
};

/* ------------------------------------------------------------ 状态 */

const kbs = ref<KBItem[]>([]);
const kbId = ref(0);
const current = computed(() => kbs.value.find((k) => k.id === kbId.value));

const strategy = ref<string>("structural");
const params = reactive<Record<string, number>>({ ...DEFAULT_PARAMS.structural });
const preview = ref<{ stats: Record<string, unknown>; total: number; chunks: ChunkItem[] } | null>(null);
const previewLoading = ref(false);
const page = ref(0);
const PAGE_SIZE = 12;

const build = ref<{ stage: string; percent: number; message: string } | null>(null);
let buildTimer: ReturnType<typeof setInterval> | null = null;
const uploading = ref(false);

const rc = reactive<RetrievalConfig>({
  mode: "hybrid", top_k: 6, score_threshold: 0.2, bm25_weight: 0.5,
  query_rewrite: true, rerank: false, route_threshold: 0.35, oos_words: [], image_caption: true,
});
const oosText = ref("");
const savingRc = ref(false);

const showCreate = ref(false);
const createForm = reactive({ name: "", description: "", visibility: "private" });
const fileInput = ref<HTMLInputElement | null>(null);

/* ------------------------------------------------------------ 加载 */

async function load() {
  try {
    const r = await api.kbs();
    kbs.value = r.kbs.filter((k) => k.can_manage);
    if (kbs.value.length && !kbs.value.find((k) => k.id === kbId.value)) kbId.value = kbs.value[0].id;
  } catch (e) { message.error((e as Error).message); }
}

function syncRetrieval() {
  const r = current.value?.retrieval;
  if (!r) return;
  Object.assign(rc, {
    mode: r.mode ?? "hybrid", top_k: r.top_k ?? 6, score_threshold: r.score_threshold ?? 0.2,
    bm25_weight: r.bm25_weight ?? 0.5, query_rewrite: r.query_rewrite !== false,
    rerank: !!r.rerank, route_threshold: r.route_threshold ?? 0.35,
    oos_words: r.oos_words ?? [], image_caption: r.image_caption !== false,
  });
  oosText.value = (r.oos_words || []).join("\n");
}

watch(kbId, () => { preview.value = null; page.value = 0; syncRetrieval(); });
watch(strategy, (s) => Object.assign(params, {}, DEFAULT_PARAMS[s]));
onMounted(async () => { await load(); syncRetrieval(); });

/* ------------------------------------------------------------ 操作 */

async function createKb() {
  if (!createForm.name.trim()) { message.warning("请输入知识库名称"); return; }
  try {
    const r = await api.createKb({ ...createForm, name: createForm.name.trim() });
    showCreate.value = false;
    Object.assign(createForm, { name: "", description: "", visibility: "private" });
    await load();
    kbId.value = r.kb.id;
    message.success("知识库已创建");
  } catch (e) { message.error((e as Error).message); }
}

async function removeKb(kb: KBItem) {
  if (!confirm(`确认删除知识库「${kb.name}」?其文档、索引与授权将一并清除。`)) return;
  try {
    await api.deleteKb(kb.id);
    if (kbId.value === kb.id) kbId.value = 0;
    await load();
    message.success("已删除");
  } catch (e) { message.error((e as Error).message); }
}

async function onUpload(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file || !kbId.value) return;
  uploading.value = true;
  try {
    const r = await api.uploadKbDoc(kbId.value, file);
    message.success(`解析完成:${r.page_count} 页 · ${r.table_count} 表格 · ${r.image_count} 图片`);
    preview.value = null;
    await load();
  } catch (err) { message.error((err as Error).message); }
  finally { uploading.value = false; if (fileInput.value) fileInput.value.value = ""; }
}

async function runPreview() {
  previewLoading.value = true;
  try {
    preview.value = await api.previewChunks(kbId.value, {
      strategy: strategy.value, params: { ...params }, offset: 0, limit: 500,
    });
    page.value = 0;
  } catch (e) { message.error((e as Error).message); }
  finally { previewLoading.value = false; }
}

const pageChunks = computed(() =>
  preview.value ? preview.value.chunks.slice(page.value * PAGE_SIZE, (page.value + 1) * PAGE_SIZE) : []);
const pageCount = computed(() =>
  preview.value ? Math.ceil(preview.value.chunks.length / PAGE_SIZE) : 0);

async function startBuild() {
  try {
    await api.buildIndex(kbId.value, { strategy: strategy.value, params: { ...params } });
    build.value = { stage: "running", percent: 0, message: "已提交构建任务" };
    if (buildTimer) clearInterval(buildTimer);
    buildTimer = setInterval(async () => {
      try {
        const p = await api.buildProgress(kbId.value);
        build.value = p;
        if (p.stage === "done" || p.stage === "error" || p.stage === "idle") {
          if (buildTimer) clearInterval(buildTimer);
          buildTimer = null;
          if (p.stage === "done") { message.success("索引构建完成"); await load(); }
          if (p.stage === "error") message.error(p.message || "构建失败");
        }
      } catch { /* 轮询失败忽略,下拍重试 */ }
    }, 1000);
  } catch (e) { message.error((e as Error).message); }
}

async function clearIndex() {
  if (!confirm("确认清空该知识库的向量索引?清空后不可问答,需重新构建。")) return;
  try {
    await api.clearIndex(kbId.value);
    await load();
    message.success("索引已清空");
  } catch (e) { message.error((e as Error).message); }
}

async function saveRetrieval() {
  savingRc.value = true;
  try {
    await api.updateKb(kbId.value, {
      retrieval: {
        ...rc,
        oos_words: oosText.value.split("\n").map((s) => s.trim()).filter(Boolean).slice(0, 200),
      },
    });
    await load();
    message.success("检索与合规配置已保存");
  } catch (e) { message.error((e as Error).message); }
  finally { savingRc.value = false; }
}

function chunkHtml(c: ChunkItem): string {
  return md.render(c.text);
}

function fmtBuiltAt(ts?: number): string {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
</script>

<template>
  <div class="mx-auto max-w-[1180px] px-6 py-6">
    <div class="mb-5">
      <h1 class="text-[17px] font-semibold">知识库与文档解析</h1>
      <p class="mt-1 text-[12px] text-muted-foreground">对应规格书 §1.3:文档解析、多策略切分、混合检索与来源追溯;知识库多库物理隔离、可启停维护</p>
    </div>

    <!-- 库选择 -->
    <div class="mb-4 flex flex-wrap items-center gap-2">
      <button v-for="k in kbs" :key="k.id"
              class="cursor-pointer rounded-full border px-3.5 py-1.5 text-[12.5px] transition-colors"
              :class="k.id === kbId ? 'border-foreground bg-foreground text-primary-foreground' : 'border-border bg-card hover:bg-secondary/70'"
              @click="kbId = k.id">
        {{ k.name }}
        <span class="ml-1 font-mono-num text-[10px] opacity-70">{{ k.index?.indexed ? `${k.index.count}块` : '未建索引' }}</span>
      </button>
      <button class="cursor-pointer rounded-full border border-dashed border-border px-3.5 py-1.5 text-[12.5px] text-muted-foreground transition-colors hover:bg-secondary/70"
              @click="showCreate = true">+ 新建知识库</button>
      <span v-if="!kbs.length" class="text-[12px] text-muted-foreground">暂无可管理的知识库</span>
    </div>

    <template v-if="current">
      <!-- 库信息条 -->
      <div class="mb-4 flex flex-wrap items-center gap-x-5 gap-y-2 panel-shadow rounded-[12px] border border-border bg-card px-4 py-3">
        <div>
          <div class="text-[14px] font-medium">{{ current.name }}</div>
          <div class="mt-0.5 text-[11.5px] text-muted-foreground">{{ current.description || '—' }}</div>
        </div>
        <div class="ml-auto flex items-center gap-3 text-[11.5px] text-muted-foreground">
          <span>{{ current.visibility === 'public' ? '公开' : '私有(按授权)' }}</span>
          <span class="font-mono-num">构建于 {{ fmtBuiltAt(current.built_at) }}</span>
          <button class="cursor-pointer rounded-[6px] border border-border px-2 py-1 text-[11.5px] transition-colors hover:bg-destructive/10 hover:text-destructive"
                  @click="removeKb(current)">删除库</button>
        </div>
      </div>

      <div class="grid gap-4 lg:grid-cols-[1fr_360px]">
        <div class="space-y-4">
          <!-- 文档解析 -->
          <section class="panel-shadow rounded-[12px] border border-border bg-card p-4">
            <div class="flex items-center justify-between">
              <h2 class="text-[13.5px] font-medium">文档解析</h2>
              <span class="text-[11px] text-muted-foreground">支持 PDF / DOCX / XLSX,≤50MB</span>
            </div>
            <div class="mt-3 flex items-center gap-3">
              <input ref="fileInput" type="file" accept=".pdf,.docx,.xlsx" class="hidden" @change="onUpload" />
              <button :disabled="uploading"
                      class="cursor-pointer rounded-[8px] bg-foreground px-4 py-2 text-[12.5px] text-primary-foreground transition-colors hover:bg-foreground/85 disabled:opacity-50"
                      @click="fileInput?.click()">
                {{ uploading ? '解析中…' : current.has_document ? '重新上传文档' : '上传文档' }}
              </button>
              <span class="text-[11.5px] text-muted-foreground">
                {{ current.has_document ? '已加载文档,可直接预览切分或重建索引' : '尚未加载文档' }}
              </span>
            </div>
          </section>

          <!-- 切分策略与预览 -->
          <section class="panel-shadow rounded-[12px] border border-border bg-card p-4">
            <div class="flex items-center justify-between">
              <h2 class="text-[13.5px] font-medium">切分策略</h2>
              <button class="cursor-pointer rounded-[7px] border border-border px-2.5 py-1 text-[11.5px] transition-colors hover:bg-secondary/70"
                      :disabled="!current.has_document || previewLoading"
                      @click="runPreview">{{ previewLoading ? '切分中…' : '预览切分' }}</button>
            </div>
            <div class="mt-3 grid grid-cols-2 gap-1.5 md:grid-cols-3">
              <button v-for="key in STRATEGY_KEYS" :key="key"
                      class="cursor-pointer rounded-[8px] border px-3 py-2 text-left transition-colors"
                      :class="strategy === key ? 'border-foreground/50 bg-secondary/70' : 'border-border hover:bg-secondary/40'"
                      @click="strategy = key">
                <div class="text-[12.5px] font-medium">{{ STRATEGY_LABELS[key].name }}</div>
                <div class="mt-0.5 line-clamp-2 text-[10.5px] leading-4 text-muted-foreground">{{ STRATEGY_LABELS[key].desc }}</div>
              </button>
            </div>
            <div class="mt-3 flex flex-wrap gap-x-5 gap-y-2">
              <label v-for="(_, pk) in params" :key="pk" class="flex items-center gap-2 text-[12px] text-muted-foreground">
                <span class="whitespace-nowrap">{{ PARAM_LABELS[pk] || pk }}</span>
                <input type="range" :min="PARAM_RANGE[pk]?.[0] ?? 0" :max="PARAM_RANGE[pk]?.[1] ?? 2000"
                       :step="PARAM_RANGE[pk]?.[2] ?? 10" v-model.number="params[pk]" class="w-28 accent-[#3f6f5f]" />
                <span class="w-12 font-mono-num text-foreground">{{ params[pk] }}</span>
              </label>
            </div>

            <!-- 预览结果 -->
            <div v-if="preview" class="mt-4">
              <div class="mb-2 flex flex-wrap items-center gap-3 text-[11.5px] text-muted-foreground">
                <span>共 <b class="font-mono-num text-foreground">{{ preview.total }}</b> 块</span>
                <span v-if="preview.stats.elapsed_s" class="font-mono-num">耗时 {{ preview.stats.elapsed_s }}s</span>
                <span v-if="preview.stats.note" class="text-[#7a621c]">{{ preview.stats.note }}</span>
                <span v-if="pageCount > 1" class="ml-auto flex items-center gap-1.5">
                  <button class="cursor-pointer rounded border border-border px-1.5 disabled:opacity-40" :disabled="page === 0" @click="page--">‹</button>
                  <span class="font-mono-num">{{ page + 1 }}/{{ pageCount }}</span>
                  <button class="cursor-pointer rounded border border-border px-1.5 disabled:opacity-40" :disabled="page >= pageCount - 1" @click="page++">›</button>
                </span>
              </div>
              <div class="grid gap-2 md:grid-cols-2">
                <div v-for="c in pageChunks" :key="c.id"
                     class="rounded-[9px] bg-[#161616] p-3 text-[#ececec]">
                  <div class="mb-1.5 flex flex-wrap items-center gap-1.5 text-[10px] text-white/45">
                    <span class="font-mono-num">#{{ c.id }}</span>
                    <span>{{ c.kind }}</span>
                    <span v-if="c.chapter">{{ c.chapter }}</span>
                    <span class="font-mono-num">{{ c.length }} 字</span>
                    <span v-if="c.image_ids?.length" class="rounded bg-[#d8b36a]/20 px-1 text-[#d8b36a]">含图 ×{{ c.image_ids.length }}</span>
                    <span v-if="c.parent_id !== null && c.parent_id !== undefined" class="rounded bg-white/10 px-1">子块→父#{{ c.parent_id }}</span>
                  </div>
                  <div class="md-dark line-clamp-6 text-[12px] leading-5" v-html="chunkHtml(c)" />
                  <details v-if="c.parent_text" class="mt-1.5">
                    <summary class="cursor-pointer text-[10.5px] text-white/40">父块上下文</summary>
                    <div class="mt-1 line-clamp-5 whitespace-pre-wrap text-[11px] leading-4 text-white/60">{{ c.parent_text }}</div>
                  </details>
                </div>
              </div>
            </div>
          </section>

          <!-- 索引构建 -->
          <section class="panel-shadow rounded-[12px] border border-border bg-card p-4">
            <div class="flex items-center justify-between">
              <h2 class="text-[13.5px] font-medium">向量索引</h2>
              <div class="flex gap-2">
                <button v-if="current.index?.indexed"
                        class="cursor-pointer rounded-[7px] border border-border px-2.5 py-1 text-[11.5px] transition-colors hover:bg-destructive/10 hover:text-destructive"
                        @click="clearIndex">清空索引</button>
                <button :disabled="!current.has_document || (build?.stage === 'running')"
                        class="cursor-pointer rounded-[7px] bg-foreground px-3 py-1 text-[11.5px] text-primary-foreground transition-colors hover:bg-foreground/85 disabled:opacity-50"
                        @click="startBuild">{{ current.index?.indexed ? '重建索引' : '构建索引' }}</button>
              </div>
            </div>
            <div class="mt-2 text-[11.5px] text-muted-foreground">
              当前状态:{{ current.index?.indexed ? `已构建(${current.index.count} 个向量块,策略 ${current.strategy || '-'})` : '未构建' }}
            </div>
            <div v-if="build && build.stage !== 'idle'" class="mt-3">
              <div class="h-1.5 overflow-hidden rounded-full bg-secondary">
                <div class="h-full rounded-full bg-[#3f6f5f] transition-all" :style="{ width: `${build.percent}%` }" />
              </div>
              <div class="mt-1.5 flex justify-between text-[11px] text-muted-foreground">
                <span>{{ build.message || build.stage }}</span>
                <span class="font-mono-num">{{ build.percent }}%</span>
              </div>
            </div>
          </section>
        </div>

        <!-- 检索与合规配置 -->
        <section class="h-fit panel-shadow rounded-[12px] border border-border bg-card p-4 lg:sticky lg:top-6">
          <h2 class="text-[13.5px] font-medium">检索与内容合规</h2>
          <div class="mt-3 space-y-3.5 text-[12.5px]">
            <div>
              <div class="mb-1.5 text-[11.5px] text-muted-foreground">检索模式</div>
              <div class="grid grid-cols-3 gap-1">
                <button v-for="m in (['semantic', 'fulltext', 'hybrid'] as const)" :key="m"
                        class="cursor-pointer rounded-[6px] border px-2 py-1.5 text-[11.5px] transition-colors"
                        :class="rc.mode === m ? 'border-foreground/50 bg-secondary font-medium' : 'border-border hover:bg-secondary/50'"
                        @click="rc.mode = m">
                  {{ m === 'semantic' ? '语义' : m === 'fulltext' ? '全文' : '混合' }}
                </button>
              </div>
            </div>
            <label class="block">
              <span class="mb-1 flex justify-between text-[11.5px] text-muted-foreground">TopK <b class="font-mono-num text-foreground">{{ rc.top_k }}</b></span>
              <input type="range" min="1" max="20" v-model.number="rc.top_k" class="w-full accent-[#3f6f5f]" />
            </label>
            <label class="block">
              <span class="mb-1 flex justify-between text-[11.5px] text-muted-foreground">相关度阈值 <b class="font-mono-num text-foreground">{{ rc.score_threshold.toFixed(2) }}</b></span>
              <input type="range" min="0" max="1" step="0.05" v-model.number="rc.score_threshold" class="w-full accent-[#3f6f5f]" />
            </label>
            <label class="block">
              <span class="mb-1 flex justify-between text-[11.5px] text-muted-foreground">BM25 权重 <b class="font-mono-num text-foreground">{{ rc.bm25_weight.toFixed(2) }}</b></span>
              <input type="range" min="0" max="1" step="0.05" v-model.number="rc.bm25_weight" class="w-full accent-[#3f6f5f]" />
            </label>
            <label class="block">
              <span class="mb-1 flex justify-between text-[11.5px] text-muted-foreground">意图路由阈值 <b class="font-mono-num text-foreground">{{ rc.route_threshold.toFixed(2) }}</b></span>
              <input type="range" min="0" max="1" step="0.05" v-model.number="rc.route_threshold" class="w-full accent-[#3f6f5f]" />
            </label>
            <div class="space-y-2 border-t border-border pt-3">
              <label class="flex cursor-pointer items-center justify-between">
                <span>查询改写(多轮指代消解)</span>
                <input type="checkbox" v-model="rc.query_rewrite" class="h-4 w-4 accent-[#3f6f5f]" />
              </label>
              <label class="flex cursor-pointer items-center justify-between">
                <span>BGE 精排(候选重排序)</span>
                <input type="checkbox" v-model="rc.rerank" class="h-4 w-4 accent-[#3f6f5f]" />
              </label>
              <label class="flex cursor-pointer items-center justify-between">
                <span>图片描述(视觉模型,图片入索引)</span>
                <input type="checkbox" v-model="rc.image_caption" class="h-4 w-4 accent-[#3f6f5f]" />
              </label>
            </div>
            <div class="border-t border-border pt-3">
              <div class="mb-1 text-[11.5px] text-muted-foreground">内容合规 · 超范围词表(每行一条,命中走拒答话术;全局敏感词由平台统一管控)</div>
              <textarea v-model="oosText" rows="3" placeholder="如:薪资密级词汇"
                        class="w-full resize-none rounded-[7px] border border-input bg-background px-2.5 py-2 text-[12px] leading-5 outline-none focus:border-foreground/50" />
            </div>
            <button :disabled="savingRc"
                    class="w-full cursor-pointer rounded-[8px] bg-foreground py-2 text-[12.5px] text-primary-foreground transition-colors hover:bg-foreground/85 disabled:opacity-50"
                    @click="saveRetrieval">{{ savingRc ? '保存中…' : '保存配置' }}</button>
          </div>
        </section>
      </div>
    </template>

    <!-- 新建知识库弹窗 -->
    <div v-if="showCreate" class="fixed inset-0 z-40 flex items-center justify-center bg-black/40 p-4"
         @click.self="showCreate = false">
      <div class="w-full max-w-[420px] rounded-[12px] border border-border bg-card p-5 shadow-lg">
        <div class="flex items-center justify-between">
          <h3 class="text-[14.5px] font-medium">新建知识库</h3>
          <button class="cursor-pointer text-muted-foreground hover:text-foreground" @click="showCreate = false">×</button>
        </div>
        <div class="mt-4 space-y-3">
          <label class="block">
            <span class="mb-1 block text-[12px] text-muted-foreground">名称</span>
            <input v-model="createForm.name" placeholder="如:设备工艺知识库"
                   class="h-9 w-full rounded-[7px] border border-input bg-background px-2.5 text-[13px] outline-none focus:border-foreground/50" />
          </label>
          <label class="block">
            <span class="mb-1 block text-[12px] text-muted-foreground">用途说明</span>
            <textarea v-model="createForm.description" rows="2" placeholder="该库面向的业务场景与文档范围"
                      class="w-full resize-none rounded-[7px] border border-input bg-background px-2.5 py-2 text-[12.5px] leading-5 outline-none focus:border-foreground/50" />
          </label>
          <div>
            <span class="mb-1.5 block text-[12px] text-muted-foreground">可见性</span>
            <div class="grid grid-cols-2 gap-1.5">
              <button class="cursor-pointer rounded-[7px] border px-3 py-2 text-left transition-colors"
                      :class="createForm.visibility === 'private' ? 'border-foreground/50 bg-secondary/70' : 'border-border'"
                      @click="createForm.visibility = 'private'">
                <div class="text-[12.5px] font-medium">私有</div>
                <div class="mt-0.5 text-[10.5px] text-muted-foreground">仅授权用户/部门可见</div>
              </button>
              <button class="cursor-pointer rounded-[7px] border px-3 py-2 text-left transition-colors"
                      :class="createForm.visibility === 'public' ? 'border-foreground/50 bg-secondary/70' : 'border-border'"
                      @click="createForm.visibility = 'public'">
                <div class="text-[12.5px] font-medium">公开</div>
                <div class="mt-0.5 text-[10.5px] text-muted-foreground">全员可问答</div>
              </button>
            </div>
          </div>
          <button class="w-full cursor-pointer rounded-[8px] bg-foreground py-2 text-[13px] text-primary-foreground transition-colors hover:bg/85 hover:bg-foreground/85"
                  @click="createKb">创建</button>
        </div>
      </div>
    </div>
  </div>
</template>
