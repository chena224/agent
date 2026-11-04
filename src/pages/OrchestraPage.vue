<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useMessage } from "naive-ui";
import { api, type IntentItem, type KBItem, type SkillItem } from "@/api";
import { useAuthStore } from "@/stores/auth";

const message = useMessage();
const auth = useAuthStore();

const kbs = ref<KBItem[]>([]);
const kbId = ref(0);
const intents = ref<IntentItem[]>([]);
const skills = ref<SkillItem[]>([]);
const skillsEnabled = ref(true);
const savingSkillsSwitch = ref(false);
const skillInput = ref<HTMLInputElement | null>(null);

const editing = ref<IntentItem | null>(null);
const form = reactive({ name: "", label: "", utterances: "", boost: "" });

/* ------------------------------------------------------------ 意图路由 */

async function loadKbs() {
  try {
    const r = await api.kbs();
    kbs.value = r.kbs.filter((k) => k.can_manage);
    if (kbs.value.length && !kbs.value.find((k) => k.id === kbId.value)) kbId.value = kbs.value[0].id;
  } catch (e) { message.error((e as Error).message); }
}

async function loadIntents() {
  if (!kbId.value) return;
  try { intents.value = (await api.intents(kbId.value)).intents; }
  catch (e) { message.error((e as Error).message); }
}

watch(kbId, loadIntents);

function startEdit(it?: IntentItem) {
  editing.value = it || null;
  if (it) {
    Object.assign(form, {
      name: it.name, label: it.label,
      utterances: it.utterances.join("\n"), boost: it.boost_keywords.join(" "),
    });
  } else {
    Object.assign(form, { name: "", label: "", utterances: "", boost: "" });
  }
}

async function saveIntent() {
  const payload = {
    label: form.label.trim(),
    utterances: form.utterances.split("\n").map((s) => s.trim()).filter(Boolean),
    boost_keywords: form.boost.split(/[\s,]+/).filter(Boolean),
  };
  if (!payload.label || !payload.utterances.length) {
    message.warning("请填写意图显示名与至少一条示例句");
    return;
  }
  try {
    if (editing.value) {
      await api.updateIntent(kbId.value, editing.value.id, payload);
    } else {
      const name = form.name.trim().toLowerCase();
      if (!/^[a-z][a-z0-9_]{0,63}$/.test(name)) {
        message.warning("路由名需为小写字母开头的英文标识");
        return;
      }
      await api.createIntent(kbId.value, { name, ...payload });
    }
    startEdit();
    await loadIntents();
    message.success("意图路由已保存,语义路由即时生效");
  } catch (e) { message.error((e as Error).message); }
}

async function removeIntent(it: IntentItem) {
  if (!confirm(`确认删除意图「${it.label}」?`)) return;
  try {
    await api.deleteIntent(kbId.value, it.id);
    await loadIntents();
    message.success("已删除");
  } catch (e) { message.error((e as Error).message); }
}

/* ------------------------------------------------------------ 技能 */

async function loadSkills() {
  if (!auth.isAdmin) return;
  try {
    skills.value = (await api.skills()).items;
    const s = await api.settings();
    skillsEnabled.value = s.skills_enabled;
  } catch { /* 非管理员跳过 */ }
}

async function toggleSkill(sk: SkillItem) {
  try {
    await api.setSkillEnabled(sk.name, !sk.enabled);
    await loadSkills();
  } catch (e) { message.error((e as Error).message); }
}

async function removeSkill(sk: SkillItem) {
  if (!confirm(`确认删除技能「${sk.name}」?`)) return;
  try {
    await api.deleteSkill(sk.name);
    await loadSkills();
    message.success("已删除");
  } catch (e) { message.error((e as Error).message); }
}

async function onUploadSkill(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    await api.uploadSkill(file);
    await loadSkills();
    message.success("技能已上传");
  } catch (err) { message.error((err as Error).message); }
  finally { if (skillInput.value) skillInput.value.value = ""; }
}

async function toggleSkillsEnabled() {
  savingSkillsSwitch.value = true;
  try {
    skillsEnabled.value = !skillsEnabled.value;
    await api.saveSettings({ skills_enabled: skillsEnabled.value });
    message.success(skillsEnabled.value ? "技能总开关已开启" : "技能总开关已关闭");
  } catch (e) { message.error((e as Error).message); skillsEnabled.value = !skillsEnabled.value; }
  finally { savingSkillsSwitch.value = false; }
}

onMounted(async () => { await loadKbs(); await loadIntents(); await loadSkills(); });
</script>

<template>
  <div class="mx-auto max-w-[1180px] px-6 py-6">
    <div class="mb-5">
      <h1 class="text-[17px] font-semibold">智能体编排</h1>
      <p class="mt-1 text-[12px] text-muted-foreground">对应规格书 §1.4:三级意图识别(规则 → 语义路由 → LLM 兜底),自定义意图路由与提示词技能管理</p>
    </div>

    <!-- 意图识别链路说明 -->
    <section class="mb-4 panel-shadow rounded-[12px] border border-border bg-card p-4">
      <h2 class="text-[13.5px] font-medium">意图识别链路</h2>
      <div class="mt-3 flex flex-wrap items-center gap-2 text-[12px]">
        <span class="rounded-[7px] bg-secondary px-3 py-1.5">① 规则层<span class="ml-1 text-[10.5px] text-muted-foreground">敏感词/问候,零开销</span></span>
        <span class="text-muted-foreground">→</span>
        <span class="rounded-[7px] bg-secondary px-3 py-1.5">② 语义路由<span class="ml-1 text-[10.5px] text-muted-foreground">示例句向量匹配,不调用 LLM</span></span>
        <span class="text-muted-foreground">→</span>
        <span class="rounded-[7px] bg-secondary px-3 py-1.5">③ LLM 兜底<span class="ml-1 text-[10.5px] text-muted-foreground">JSON 结构化分类</span></span>
      </div>
      <p class="mt-2.5 text-[11.5px] leading-5 text-muted-foreground">
        内置路由:知识库问答 / 闲聊 / 超出范围 / 工具调用。下方可为每个知识库追加自定义意图(如「违规处理」「报销流程」),命中后归入知识库问答分支,并可用增强关键词提升检索召回。
      </p>
    </section>

    <div class="grid gap-4 lg:grid-cols-[1fr_400px]">
      <!-- 意图路由管理 -->
      <section class="panel-shadow rounded-[12px] border border-border bg-card p-4">
        <div class="flex flex-wrap items-center gap-2">
          <h2 class="text-[13.5px] font-medium">意图路由</h2>
          <select v-model.number="kbId"
                  class="ml-auto h-8 rounded-[7px] border border-input bg-background px-2 text-[12px] outline-none focus:border-foreground/50">
            <option v-for="k in kbs" :key="k.id" :value="k.id">{{ k.name }}</option>
          </select>
          <button class="cursor-pointer rounded-[7px] bg-foreground px-3 py-1.5 text-[11.5px] text-primary-foreground transition-colors hover:bg-foreground/85"
                  @click="startEdit()">+ 新增意图</button>
        </div>
        <div class="mt-3 space-y-2">
          <div v-if="!intents.length" class="rounded-[8px] border border-dashed border-border px-4 py-6 text-center text-[12px] text-muted-foreground">
            该库暂无自定义意图,命中内置路由兜底
          </div>
          <div v-for="it in intents" :key="it.id" class="rounded-[9px] border border-border p-3">
            <div class="flex items-center gap-2">
              <span class="text-[13px] font-medium">{{ it.label }}</span>
              <span class="font-mono-num rounded bg-secondary px-1.5 py-0.5 text-[10.5px] text-muted-foreground">{{ it.name }}</span>
              <span class="text-[10.5px] text-muted-foreground">{{ it.utterances.length }} 条示例句</span>
              <span class="ml-auto flex gap-1.5">
                <button class="cursor-pointer rounded border border-border px-2 py-0.5 text-[11px] transition-colors hover:bg-secondary/70" @click="startEdit(it)">编辑</button>
                <button class="cursor-pointer rounded border border-border px-2 py-0.5 text-[11px] transition-colors hover:bg-destructive/10 hover:text-destructive" @click="removeIntent(it)">删除</button>
              </span>
            </div>
            <div class="mt-1.5 line-clamp-2 text-[11.5px] leading-5 text-muted-foreground">{{ it.utterances.join(" / ") }}</div>
            <div v-if="it.boost_keywords.length" class="mt-1.5 flex flex-wrap gap-1">
              <span v-for="kw in it.boost_keywords" :key="kw" class="rounded bg-secondary px-1.5 py-0.5 text-[10.5px] text-muted-foreground">{{ kw }}</span>
            </div>
          </div>
        </div>

        <!-- 编辑/新增表单 -->
        <div class="mt-4 rounded-[9px] border border-border bg-secondary/30 p-3.5">
          <div class="mb-2.5 text-[12.5px] font-medium">{{ editing ? `编辑意图「${editing.label}」` : '新增意图' }}</div>
          <div class="flex flex-wrap gap-2">
            <input v-model="form.name" :disabled="!!editing" placeholder="路由名(小写英文,如 violation)"
                   class="h-8 w-56 rounded-[6px] border border-input bg-background px-2 text-[12.5px] outline-none focus:border-foreground/50 disabled:opacity-50" />
            <input v-model="form.label" placeholder="显示名(如 违规处理)"
                   class="h-8 w-40 rounded-[6px] border border-input bg-background px-2 text-[12.5px] outline-none focus:border-foreground/50" />
            <input v-model="form.boost" placeholder="检索增强关键词,空格分隔"
                   class="h-8 min-w-52 flex-1 rounded-[6px] border border-input bg-background px-2 text-[12.5px] outline-none focus:border-foreground/50" />
          </div>
          <textarea v-model="form.utterances" rows="3" placeholder="示例句,每行一条(语义匹配的锚点,建议 5~10 条覆盖常见问法)"
                    class="mt-2 w-full resize-none rounded-[6px] border border-input bg-background px-2 py-1.5 text-[12.5px] leading-5 outline-none focus:border-foreground/50" />
          <div class="mt-2 flex items-center gap-2">
            <button class="cursor-pointer rounded-[7px] bg-foreground px-3.5 py-1.5 text-[12px] text-primary-foreground transition-colors hover:bg-foreground/85"
                    @click="saveIntent">{{ editing ? '保存修改' : '创建意图' }}</button>
            <button v-if="editing" class="cursor-pointer rounded-[7px] border border-border px-3 py-1.5 text-[12px] transition-colors hover:bg-secondary/70"
                    @click="startEdit()">取消编辑</button>
          </div>
        </div>
      </section>

      <!-- 技能管理(仅系统管理员) -->
      <section v-if="auth.isAdmin" class="h-fit panel-shadow rounded-[12px] border border-border bg-card p-4">
        <div class="flex items-center justify-between">
          <h2 class="text-[13.5px] font-medium">提示词技能</h2>
          <label class="flex cursor-pointer items-center gap-1.5 text-[11.5px] text-muted-foreground">
            总开关
            <input type="checkbox" :checked="skillsEnabled" :disabled="savingSkillsSwitch"
                   class="h-4 w-4 accent-[#3f6f5f]" @change="toggleSkillsEnabled" />
          </label>
        </div>
        <p class="mt-1.5 text-[11.5px] leading-5 text-muted-foreground">
          技能 = 按需加载的指令包(.md/.zip):系统提示只放名称与简介,模型判中后经 use_skill 工具拉取全文,支撑提示词复用与流程编排。
        </p>
        <div class="mt-3 space-y-2">
          <div v-if="!skills.length" class="rounded-[8px] border border-dashed border-border px-4 py-5 text-center text-[12px] text-muted-foreground">暂无技能</div>
          <div v-for="sk in skills" :key="sk.name" class="flex items-center gap-2.5 rounded-[8px] border border-border px-3 py-2.5">
            <div class="min-w-0 flex-1">
              <div class="text-[12.5px] font-medium">{{ sk.name }}</div>
              <div class="mt-0.5 line-clamp-2 text-[11px] leading-4 text-muted-foreground">{{ sk.description }}</div>
            </div>
            <label class="flex cursor-pointer items-center gap-1 text-[11px] text-muted-foreground">
              <input type="checkbox" :checked="sk.enabled" class="h-3.5 w-3.5 accent-[#3f6f5f]" @change="toggleSkill(sk)" />启用
            </label>
            <button class="cursor-pointer rounded border border-border px-2 py-0.5 text-[11px] transition-colors hover:bg-destructive/10 hover:text-destructive"
                    @click="removeSkill(sk)">删除</button>
          </div>
        </div>
        <input ref="skillInput" type="file" accept=".md,.zip" class="hidden" @change="onUploadSkill" />
        <button class="mt-3 w-full cursor-pointer rounded-[8px] border border-dashed border-border py-2 text-[12px] text-muted-foreground transition-colors hover:bg-secondary/60"
                @click="skillInput?.click()">上传技能包(.md / .zip)</button>
      </section>
    </div>
  </div>
</template>
