<script setup lang="ts">
import { onMounted, reactive, ref } from "vue";
import { useMessage } from "naive-ui";
import { api, type Settings } from "@/api";

const message = useMessage();
const loading = ref(true);
const saving = ref(false);
const testing = ref(false);

const form = reactive({
  deepseek_api_key: "",
  deepseek_backup_api_key: "",
  deepseek_base_url: "",
  deepseek_model: "",
  vision_model: "",
  retrieve_top_k: 6,
});
const meta = ref<Settings | null>(null);

async function load() {
  loading.value = true;
  try {
    const s = await api.settings();
    meta.value = s;
    Object.assign(form, {
      deepseek_api_key: "", deepseek_backup_api_key: "",
      deepseek_base_url: s.deepseek_base_url, deepseek_model: s.deepseek_model,
      vision_model: s.vision_model, retrieve_top_k: s.retrieve_top_k,
    });
  } catch (e) { message.error((e as Error).message); }
  finally { loading.value = false; }
}
onMounted(load);

async function save() {
  saving.value = true;
  try {
    await api.saveSettings({
      deepseek_api_key: form.deepseek_api_key || undefined,
      deepseek_backup_api_key: form.deepseek_backup_api_key || undefined,
      deepseek_base_url: form.deepseek_base_url.trim(),
      deepseek_model: form.deepseek_model.trim(),
      vision_model: form.vision_model.trim(),
      retrieve_top_k: form.retrieve_top_k,
    });
    form.deepseek_api_key = "";
    form.deepseek_backup_api_key = "";
    await load();
    message.success("设置已保存,即时生效");
  } catch (e) { message.error((e as Error).message); }
  finally { saving.value = false; }
}

async function testConn() {
  testing.value = true;
  try {
    if (form.deepseek_api_key) await save();   // 先落库再测,测的就是新密钥
    const r = await api.testSettings();
    if (r.ok) message.success(`连接正常(${r.latency_ms}ms)`);
    else message.error(r.error || "连接失败");
  } catch (e) { message.error((e as Error).message); }
  finally { testing.value = false; }
}
</script>

<template>
  <div class="mx-auto max-w-[860px] px-6 py-6">
    <div class="mb-5">
      <h1 class="text-[17px] font-semibold">系统管理</h1>
      <p class="mt-1 text-[12px] text-muted-foreground">对应规格书 §1.1/§1.2/§1.11:大模型接入、主备密钥故障自动切换与运行参数;全平台 RESTful API 见部署文档</p>
    </div>

    <div v-if="loading" class="py-16 text-center text-[12.5px] text-muted-foreground">加载中…</div>
    <template v-else>
      <!-- 模型服务 -->
      <section class="panel-shadow rounded-[12px] border border-border bg-card p-5">
        <h2 class="text-[13.5px] font-medium">大模型调用与推理服务</h2>
        <p class="mt-0.5 text-[11.5px] text-muted-foreground">OpenAI 兼容协议接入;密钥仅保存在服务端数据目录,界面不明文回显。</p>
        <div class="mt-4 grid gap-4 md:grid-cols-2">
          <label class="block">
            <span class="mb-1 flex items-center gap-2 text-[11.5px] text-muted-foreground">
              API Key(主)
              <span v-if="meta?.deepseek_api_key_set" class="rounded bg-[#3f6f5f]/10 px-1.5 py-px font-mono-num text-[10px] text-[#3f6f5f]">{{ meta.deepseek_api_key_masked }}</span>
            </span>
            <input v-model="form.deepseek_api_key" type="password" autocomplete="new-password"
                   :placeholder="meta?.deepseek_api_key_set ? '已配置,留空则不修改' : 'sk-...'"
                   class="h-9 w-full rounded-[7px] border border-input bg-background px-2.5 font-mono-num text-[12.5px] outline-none focus:border-foreground/50" />
          </label>
          <label class="block">
            <span class="mb-1 flex items-center gap-2 text-[11.5px] text-muted-foreground">
              API Key(备,故障自动切换)
              <span v-if="meta?.deepseek_backup_api_key_set" class="rounded bg-[#3f6f5f]/10 px-1.5 py-px font-mono-num text-[10px] text-[#3f6f5f]">{{ meta.deepseek_backup_api_key_masked }}</span>
            </span>
            <input v-model="form.deepseek_backup_api_key" type="password" autocomplete="new-password"
                   :placeholder="meta?.deepseek_backup_api_key_set ? '已配置,留空则不修改' : '主密钥故障时自动切换'"
                   class="h-9 w-full rounded-[7px] border border-input bg-background px-2.5 font-mono-num text-[12.5px] outline-none focus:border-foreground/50" />
          </label>
          <label class="block">
            <span class="mb-1 block text-[11.5px] text-muted-foreground">Base URL</span>
            <input v-model="form.deepseek_base_url"
                   class="h-9 w-full rounded-[7px] border border-input bg-background px-2.5 font-mono-num text-[12.5px] outline-none focus:border-foreground/50" />
          </label>
          <div class="grid grid-cols-2 gap-3">
            <label class="block">
              <span class="mb-1 block text-[11.5px] text-muted-foreground">对话模型</span>
              <input v-model="form.deepseek_model"
                     class="h-9 w-full rounded-[7px] border border-input bg-background px-2.5 font-mono-num text-[12.5px] outline-none focus:border-foreground/50" />
            </label>
            <label class="block">
              <span class="mb-1 block text-[11.5px] text-muted-foreground">视觉模型(图片描述)</span>
              <input v-model="form.vision_model"
                     class="h-9 w-full rounded-[7px] border border-input bg-background px-2.5 font-mono-num text-[12.5px] outline-none focus:border-foreground/50" />
            </label>
          </div>
        </div>
        <label class="mt-4 block max-w-xs">
          <span class="mb-1 flex justify-between text-[11.5px] text-muted-foreground">默认检索 TopK <b class="font-mono-num text-foreground">{{ form.retrieve_top_k }}</b></span>
          <input type="range" min="1" max="12" v-model.number="form.retrieve_top_k" class="w-full accent-[#3f6f5f]" />
        </label>
        <div class="mt-5 flex items-center gap-2.5">
          <button :disabled="saving"
                  class="cursor-pointer rounded-[8px] bg-foreground px-5 py-2 text-[13px] text-primary-foreground transition-colors hover:bg-foreground/85 disabled:opacity-50"
                  @click="save">{{ saving ? '保存中…' : '保存设置' }}</button>
          <button :disabled="testing"
                  class="cursor-pointer rounded-[8px] border border-border px-4 py-2 text-[13px] transition-colors hover:bg-secondary/70 disabled:opacity-50"
                  @click="testConn">{{ testing ? '测试中…' : '测试连接' }}</button>
        </div>
      </section>

      <!-- 运行参数 -->
      <section class="mt-4 panel-shadow rounded-[12px] border border-border bg-card p-5">
        <h2 class="text-[13.5px] font-medium">运行参数</h2>
        <dl class="mt-3 space-y-2 text-[13px]">
          <div v-for="[k, v] in [
              ['向量模型', meta?.embedding_model ?? '—'],
              ['向量库', 'Milvus Lite(嵌入式)'],
              ['短时记忆', `${meta?.memory_turns ?? 50} 轮对话(LangGraph Checkpointer)`],
              ['编排框架', 'LangChain + LangGraph'],
              ['部署形态', '单机一体化 / 集群(信创 x86 · ARM)'],
            ]" :key="k" class="flex justify-between border-b border-border/60 pb-2 last:border-0">
            <dt class="text-muted-foreground">{{ k }}</dt>
            <dd class="font-mono-num text-[12px]">{{ v }}</dd>
          </div>
        </dl>
      </section>
    </template>
  </div>
</template>
