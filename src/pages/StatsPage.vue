<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { api } from "@/api";

interface Summary { total_questions: number; total_tokens: number; avg_latency_ms: number; today_questions: number; today_tokens: number; sessions: number; builds: number; }
interface TsPoint { date: string; questions: number; tokens: number; }
interface RecentItem { time: string; session_id: string; question: string; answer_preview: string; tokens: number; latency_ms: number; sources_n: number; username: string; kb_name: string; intent: string; intent_label?: string; }
interface IntentCount { intent: string; label?: string; count: number; }
interface KbCount { kb_name: string; questions: number; tokens: number; }

const summary = ref<Summary | null>(null);
const series = ref<TsPoint[]>([]);
const intents = ref<IntentCount[]>([]);
const kbStats = ref<KbCount[]>([]);
const recent = ref<RecentItem[]>([]);

async function load() {
  try {
    const [s, t, i, k, r] = await Promise.all([
      api.statsSummary(), api.statsTimeseries(), api.statsIntents(), api.statsKbs(), api.statsRecent(),
    ]);
    summary.value = s;
    series.value = t.days ?? [];
    intents.value = i.items ?? [];
    kbStats.value = k.items ?? [];
    recent.value = r.items ?? [];
  } catch { /* 静默,下拍重试 */ }
}
let timer: ReturnType<typeof setInterval> | null = null;
onMounted(() => { load(); timer = setInterval(load, 10_000); });
onBeforeUnmount(() => { if (timer) clearInterval(timer); });

/* ---------------- 面积图(双序列:提问量 + token 千分位缩放) ---------------- */
const CHART_W = 720, CHART_H = 200, PAD = 8;

const chart = computed(() => {
  const pts = series.value;
  if (!pts.length) return null;
  const maxQ = Math.max(1, ...pts.map((p) => p.questions));
  const maxT = Math.max(1, ...pts.map((p) => p.tokens));
  const x = (i: number) => PAD + (i * (CHART_W - PAD * 2)) / Math.max(1, pts.length - 1);
  const yQ = (v: number) => CHART_H - PAD - (v / maxQ) * (CHART_H - PAD * 2);
  const yT = (v: number) => CHART_H - PAD - (v / maxT) * (CHART_H - PAD * 2);
  const line = (getY: (v: number) => number) =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${getY(p.questions).toFixed(1)}`).join(" ");
  const qLine = pts.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${yQ(p.questions).toFixed(1)}`).join(" ");
  const tLine = pts.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${yT(p.tokens).toFixed(1)}`).join(" ");
  const qArea = `${qLine} L${x(pts.length - 1).toFixed(1)},${CHART_H - PAD} L${x(0).toFixed(1)},${CHART_H - PAD} Z`;
  const tArea = `${tLine} L${x(pts.length - 1).toFixed(1)},${CHART_H - PAD} L${x(0).toFixed(1)},${CHART_H - PAD} Z`;
  return { x, yQ, qLine, tLine, qArea, tArea };
});

const cards = computed(() => [
  { label: "累计提问", value: summary.value ? fmtNum(summary.value.total_questions) : "—", sub: `今日 ${summary.value?.today_questions ?? 0}` },
  { label: "累计 Token", value: summary.value ? fmtNum(summary.value.total_tokens) : "—", sub: `今日 ${summary.value?.today_tokens ?? 0}` },
  { label: "会话数", value: summary.value ? fmtNum(summary.value.sessions) : "—", sub: `索引构建 ${summary.value?.builds ?? 0} 次` },
  { label: "平均耗时", value: summary.value ? (summary.value.avg_latency_ms / 1000).toFixed(1) + "s" : "—", sub: "端到端问答时延" },
]);

const intentTotal = computed(() => intents.value.reduce((s, i) => s + i.count, 0));
const INTENT_LABELS: Record<string, string> = {
  kb_query: "知识库问答", chitchat: "闲聊", out_of_scope: "超出范围", tool: "工具调用",
};
const INTENT_COLORS: Record<string, string> = {
  kb_query: "#3f6f5f", chitchat: "#999", out_of_scope: "#c9a227", tool: "#4a6d8c",
};
const kbMax = computed(() => Math.max(1, ...kbStats.value.map((k) => k.questions)));

function fmtNum(n?: number): string {
  if (n === undefined || n === null) return "—";
  return n >= 10000 ? `${(n / 10000).toFixed(1)}w` : String(n);
}
</script>

<template>
  <div class="mx-auto max-w-[1180px] px-6 py-6">
    <div class="mb-5">
      <h1 class="text-[17px] font-semibold">统计分析</h1>
      <p class="mt-1 text-[12px] text-muted-foreground">对应规格书 §1.10:用户数、提问次数、Token 用量与意图分布,每 10 秒自动刷新</p>
    </div>

    <!-- 指标卡 -->
    <div class="grid grid-cols-2 gap-3 md:grid-cols-4">
      <div v-for="c in cards" :key="c.label"
           class="panel-shadow rounded-[12px] border border-border bg-card p-4">
        <div class="text-[11.5px] text-muted-foreground">{{ c.label }}</div>
        <div class="mt-1.5 font-mono-num text-[24px] font-semibold leading-7">{{ c.value }}</div>
        <div class="mt-1 text-[10.5px] text-muted-foreground">{{ c.sub }}</div>
      </div>
    </div>

    <!-- 趋势图 -->
    <section class="mt-4 panel-shadow rounded-[12px] border border-border bg-card p-4">
      <div class="flex items-center justify-between">
        <h2 class="text-[13.5px] font-medium">近 14 天趋势</h2>
        <div class="flex items-center gap-4 text-[11px] text-muted-foreground">
          <span class="flex items-center gap-1.5"><i class="inline-block h-2 w-2 rounded-full bg-[#161616]" />提问量</span>
          <span class="flex items-center gap-1.5"><i class="inline-block h-2 w-2 rounded-full bg-[#3f6f5f]" />Token</span>
        </div>
      </div>
      <svg v-if="chart" :viewBox="`0 0 ${CHART_W} ${CHART_H}`" class="mt-3 h-[200px] w-full">
        <path :d="chart.qArea" fill="#161616" opacity="0.06" />
        <path :d="chart.tArea" fill="#3f6f5f" opacity="0.10" />
        <path :d="chart.qLine" fill="none" stroke="#161616" stroke-width="1.5" />
        <path :d="chart.tLine" fill="none" stroke="#3f6f5f" stroke-width="1.5" />
        <g v-for="(p, i) in series" :key="p.date">
          <circle :cx="chart.x(i)" :cy="chart.yQ(p.questions)" r="2" fill="#161616">
            <title>{{ p.date }} · 提问 {{ p.questions }} · Token {{ p.tokens }}</title>
          </circle>
        </g>
        <text v-for="(p, i) in series" :key="'t' + p.date" v-show="i % 3 === 0 || i === series.length - 1"
              :x="chart.x(i)" :y="CHART_H - 2" text-anchor="middle" font-size="9" fill="#999">{{ p.date }}</text>
      </svg>
      <div v-else class="py-10 text-center text-[12px] text-muted-foreground">暂无统计数据</div>
    </section>

    <div class="mt-4 grid gap-4 lg:grid-cols-2">
      <!-- 意图分布 -->
      <section class="panel-shadow rounded-[12px] border border-border bg-card p-4">
        <h2 class="text-[13.5px] font-medium">意图分布</h2>
        <div class="mt-3 space-y-2.5">
          <div v-if="!intents.length" class="py-6 text-center text-[12px] text-muted-foreground">暂无数据</div>
          <div v-for="it in intents" :key="it.intent">
            <div class="mb-1 flex justify-between text-[11.5px]">
              <span>{{ INTENT_LABELS[it.intent] || it.label || it.intent }}</span>
              <span class="font-mono-num text-muted-foreground">{{ it.count }} · {{ intentTotal ? ((it.count / intentTotal) * 100).toFixed(1) : 0 }}%</span>
            </div>
            <div class="h-1.5 overflow-hidden rounded-full bg-secondary">
              <div class="h-full rounded-full transition-all"
                   :style="{ width: `${intentTotal ? (it.count / intentTotal) * 100 : 0}%`, background: INTENT_COLORS[it.intent] || '#888' }" />
            </div>
          </div>
        </div>
      </section>

      <!-- 分库统计 -->
      <section class="panel-shadow rounded-[12px] border border-border bg-card p-4">
        <h2 class="text-[13.5px] font-medium">分知识库用量</h2>
        <div class="mt-3 space-y-2.5">
          <div v-if="!kbStats.length" class="py-6 text-center text-[12px] text-muted-foreground">暂无数据</div>
          <div v-for="k in kbStats" :key="k.kb_name">
            <div class="mb-1 flex justify-between text-[11.5px]">
              <span class="truncate">{{ k.kb_name }}</span>
              <span class="shrink-0 font-mono-num text-muted-foreground">{{ k.questions }} 次 · {{ fmtNum(k.tokens) }} tokens</span>
            </div>
            <div class="h-1.5 overflow-hidden rounded-full bg-secondary">
              <div class="h-full rounded-full bg-[#3f6f5f] transition-all" :style="{ width: `${(k.questions / kbMax) * 100}%` }" />
            </div>
          </div>
        </div>
      </section>
    </div>

    <!-- 最近问答 -->
    <section class="mt-4 panel-shadow rounded-[12px] border border-border bg-card p-4">
      <h2 class="text-[13.5px] font-medium">最近问答记录</h2>
      <div class="mt-3 overflow-x-auto">
        <table class="data-table">
          <thead>
            <tr><th>时间</th><th>用户</th><th>知识库</th><th>问题</th><th>意图</th><th class="text-right">Token</th><th class="text-right">耗时</th><th class="text-right">来源</th></tr>
          </thead>
          <tbody>
            <tr v-if="!recent.length"><td colspan="8" class="py-6 text-center text-muted-foreground">暂无记录</td></tr>
            <tr v-for="(r, i) in recent" :key="i">
              <td class="whitespace-nowrap font-mono-num">{{ r.time }}</td>
              <td>{{ r.username }}</td>
              <td class="max-w-32 truncate">{{ r.kb_name }}</td>
              <td class="max-w-72 truncate" :title="r.question">{{ r.question }}</td>
              <td><span class="rounded bg-secondary px-1.5 py-0.5 text-[11px]">{{ INTENT_LABELS[r.intent] || r.intent_label || r.intent || '—' }}</span></td>
              <td class="text-right font-mono-num">{{ r.tokens }}</td>
              <td class="text-right font-mono-num">{{ (r.latency_ms / 1000).toFixed(1) }}s</td>
              <td class="text-right font-mono-num">{{ r.sources_n }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>
