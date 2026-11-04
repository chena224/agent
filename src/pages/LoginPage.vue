<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";
import { useMessage } from "naive-ui";
import { useAuthStore } from "@/stores/auth";

const router = useRouter();
const message = useMessage();
const auth = useAuthStore();

const username = ref("");
const password = ref("");
const loading = ref(false);

/** 内置账号与后端 SEED_DEMO_DATA 种子保持一致(db.py init_db) */
const DEMO_ACCOUNTS = [
  { username: "admin", password: "admin123", name: "系统管理员", dept: "信息化办公室", role: "admin" },
  { username: "manager", password: "manager123", name: "张主管", dept: "综合管理部", role: "workspace_admin" },
  { username: "chen", password: "chen123", name: "陈工", dept: "生产制造部", role: "user" },
  { username: "li", password: "li123", name: "李工", dept: "工艺技术部", role: "user" },
  { username: "wang", password: "wang123", name: "王工", dept: "设备动力部", role: "user" },
];

const ROLE_LABEL: Record<string, string> = {
  admin: "系统管理员",
  workspace_admin: "工作空间管理员",
  user: "普通用户",
};

const FEATURES = [
  "大模型调用与推理服务:主备密钥故障自动切换,推理全程流式可见",
  "知识库与文档解析:多策略切分、混合检索、来源展示与原文追溯",
  "智能体编排:三级意图识别、工具调用、业务系统数据对接",
  "组织机构与权限:RBAC 三角色,部门/用户两级资源授权",
];

function fill(acc: (typeof DEMO_ACCOUNTS)[number]) {
  username.value = acc.username;
  password.value = acc.password;
}

async function submit() {
  if (!username.value.trim() || !password.value) {
    message.warning("请输入用户名与口令");
    return;
  }
  loading.value = true;
  try {
    await auth.login(username.value.trim(), password.value);
    router.push("/");
  } catch (e) {
    message.error((e as Error).message || "登录失败");
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <div class="login-ambient relative flex min-h-screen items-center justify-center px-4 py-10">
    <div class="login-grid pointer-events-none absolute inset-0" />
    <div class="panel-shadow-lg relative grid w-full max-w-[920px] overflow-hidden rounded-[18px] border border-white/10 bg-card md:grid-cols-[1.12fr_1fr]">
      <!-- 左侧:品牌与口径 -->
      <div class="relative hidden flex-col justify-between overflow-hidden bg-[#151815] p-9 text-[#ececec] md:flex">
        <div class="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-[#6fa287]/12 blur-3xl" />
        <div class="pointer-events-none absolute -bottom-24 -left-16 h-56 w-56 rounded-full bg-[#6fa287]/8 blur-3xl" />
        <div class="relative">
          <div class="flex items-center gap-3">
            <div class="brand-mark flex h-10 w-10 items-center justify-center rounded-[11px]">
              <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5">
                <path d="M12 2 3 7v10l9 5 9-5V7l-9-5z" /><path d="M3 7l9 5 9-5" /><path d="M12 12v10" />
              </svg>
            </div>
            <div>
              <div class="text-[17px] font-semibold tracking-wide">知识库智能体平台</div>
              <div class="mt-0.5 text-[11px] text-white/45">工厂AI智能体本地化部署及应用项目</div>
            </div>
          </div>
        </div>
        <ul class="relative space-y-4 text-[12.5px] leading-6 text-white/65">
          <li v-for="f in FEATURES" :key="f" class="flex gap-2.5">
            <svg viewBox="0 0 24 24" fill="none" stroke="#8fb8a5" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" class="mt-[5px] h-3.5 w-3.5 shrink-0">
              <path d="M20 6 9 17l-5-5" />
            </svg>
            <span>{{ f }}</span>
          </li>
        </ul>
        <div class="relative flex items-center gap-2 text-[11px] text-white/35">
          <span class="inline-block h-1 w-1 rounded-full bg-[#6fa287]" />
          支持信创环境(x86 / ARM)本地化部署
        </div>
      </div>
      <!-- 右侧:登录表单 -->
      <div class="p-9">
        <div class="text-[17px] font-semibold tracking-wide">登录</div>
        <div class="mt-1 text-[12px] text-muted-foreground">使用平台分配的账号登录,内置账号见下方快捷填充</div>
        <form class="mt-6 space-y-4" @submit.prevent="submit">
          <label class="block">
            <span class="mb-1.5 block text-[12.5px] text-muted-foreground">用户名</span>
            <input v-model="username" type="text" autocomplete="username"
                   class="h-10 w-full rounded-[8px] border border-input bg-background px-3 text-[14px] outline-none transition-colors focus:border-foreground/50" />
          </label>
          <label class="block">
            <span class="mb-1.5 block text-[12.5px] text-muted-foreground">口令</span>
            <input v-model="password" type="password" autocomplete="current-password"
                   class="h-10 w-full rounded-[8px] border border-input bg-background px-3 text-[14px] outline-none transition-colors focus:border-foreground/50" />
          </label>
          <button type="submit" :disabled="loading"
                  class="h-10 w-full cursor-pointer rounded-[9px] bg-foreground text-[14px] font-medium tracking-[0.2em] text-primary-foreground shadow-[0_6px_16px_-6px_rgb(0_0_0/0.4)] transition-all hover:bg-foreground/85 hover:shadow-[0_8px_20px_-6px_rgb(0_0_0/0.45)] active:translate-y-px disabled:cursor-not-allowed disabled:opacity-50">
            {{ loading ? "登录中…" : "登 录" }}
          </button>
        </form>
        <div class="mt-7 border-t border-border pt-5">
          <div class="mb-2.5 text-[11.5px] text-muted-foreground">内置账号(点击填充,对应三种角色权限)</div>
          <div class="grid grid-cols-1 gap-1.5">
            <button v-for="acc in DEMO_ACCOUNTS" :key="acc.username" type="button"
                    class="card-hover flex cursor-pointer items-center justify-between rounded-[9px] border border-border bg-card px-3 py-2 text-left"
                    @click="fill(acc)">
              <span class="flex items-center gap-2">
                <span class="text-[13px] font-medium">{{ acc.name }}</span>
                <span class="text-[11px] text-muted-foreground">{{ acc.dept }}</span>
              </span>
              <span class="font-mono-num text-[11px] text-muted-foreground">{{ acc.username }} · {{ ROLE_LABEL[acc.role] }}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
