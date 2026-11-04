<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { NConfigProvider, NDialogProvider, NMessageProvider, zhCN, dateZhCN, darkTheme } from "naive-ui";
import { useAuthStore } from "@/stores/auth";
import { api } from "@/api";

const auth = useAuthStore();
const route = useRoute();
const router = useRouter();

const isLogin = computed(() => route.name === "login");

interface NavItem { key: string; label: string; icon: string; roles: string[] }
/** 导航术语对齐《软件技术规格书》章节口径 */
const NAV: NavItem[] = [
  { key: "/", label: "智能问答", icon: "M8 10h8M8 14h5M21 12a9 9 0 1 1-4.4-7.7", roles: ["admin", "workspace_admin", "user"] },
  { key: "/knowledge", label: "知识库与文档解析", icon: "M4 19.5A2.5 2.5 0 0 1 6.5 17H20M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15z", roles: ["admin", "workspace_admin"] },
  { key: "/orchestration", label: "智能体编排", icon: "M12 2v4m0 12v4M2 12h4m12 0h4M7 7l2.5 2.5M17 17l-2.5-2.5M7 17l2.5-2.5M17 7l-2.5 2.5", roles: ["admin", "workspace_admin"] },
  { key: "/tools", label: "工具管理", icon: "M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18v3h3l6.3-6.3a4 4 0 0 0 5.4-5.4l-2.4 2.4-2.3-.6-.6-2.3 2.3-2.5z", roles: ["admin", "workspace_admin"] },
  { key: "/stats", label: "统计分析", icon: "M3 3v18h18M8 16v-5m4 5V8m4 8v-3", roles: ["admin", "workspace_admin"] },
  { key: "/org", label: "组织机构与权限", icon: "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm13 10v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75", roles: ["admin"] },
  { key: "/system", label: "系统管理", icon: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm7.4-3a7.4 7.4 0 0 0-.1-1.2l2.1-1.6-2-3.4-2.4 1a7.5 7.5 0 0 0-2-1.2L14.6 3h-4l-.4 2.6a7.5 7.5 0 0 0-2 1.2l-2.4-1-2 3.4L5.9 10a7.4 7.4 0 0 0 0 2.4l-2.1 1.6 2 3.4 2.4-1a7.5 7.5 0 0 0 2 1.2l.4 2.6h4l.4-2.6a7.5 7.5 0 0 0 2-1.2l2.4 1 2-3.4-2.1-1.6c.06-.4.1-.8.1-1.2z", roles: ["admin"] },
];

const navItems = computed(() =>
  NAV.filter((n) => auth.user && n.roles.includes(auth.user.role)));

const health = ref<{ llm_configured: boolean; llm_model: string } | null>(null);
let timer: ReturnType<typeof setInterval> | null = null;
async function poll() {
  try { health.value = await api.health(); } catch { health.value = null; }
}
onMounted(() => { poll(); timer = setInterval(poll, 30_000); });
onBeforeUnmount(() => { if (timer) clearInterval(timer); });

const roleBadge: Record<string, string> = {
  admin: "bg-[#b4413c]/20 text-[#e8a29e]",
  workspace_admin: "bg-[#3f6f5f]/25 text-[#a7c4b8]",
  user: "bg-white/10 text-white/70",
};

async function onLogout() {
  await auth.logout();
  router.push("/login");
}
</script>

<template>
  <n-config-provider :locale="zhCN" :date-locale="dateZhCN">
    <n-message-provider>
      <n-dialog-provider>
        <router-view v-if="isLogin" />
        <div v-else-if="auth.loggedIn" class="flex h-screen min-h-0">
          <!-- 深色侧边栏 -->
          <aside class="flex w-[228px] shrink-0 flex-col bg-[#161616] text-[#ececec]">
            <div class="flex items-center gap-2.5 px-4 pb-5 pt-5">
              <div class="brand-mark flex h-8 w-8 shrink-0 items-center justify-center rounded-[9px]">
                <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4">
                  <path d="M12 2 3 7v10l9 5 9-5V7l-9-5z" /><path d="M3 7l9 5 9-5" /><path d="M12 12v10" />
                </svg>
              </div>
              <div class="min-w-0">
                <div class="truncate text-[14px] font-semibold tracking-wide">知识库智能体平台</div>
                <div class="mt-0.5 truncate text-[10.5px] text-white/40">工厂AI智能体 · 本地化部署</div>
              </div>
            </div>
            <nav class="flex-1 space-y-1 overflow-y-auto px-2.5">
              <router-link
                v-for="item in navItems" :key="item.key" :to="item.key"
                class="group relative flex items-center gap-2.5 rounded-[8px] px-3 py-2 text-[13px] text-white/60 transition-all hover:bg-white/6 hover:text-white"
                :class="{ 'bg-white/10 !text-white font-medium shadow-[inset_2.5px_0_0_0_#6fa287]': route.path === item.key }"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
                     stroke-linecap="round" stroke-linejoin="round"
                     class="h-4 w-4 shrink-0 opacity-70 transition-opacity group-hover:opacity-100">
                  <path :d="item.icon" />
                </svg>
                <span class="truncate">{{ item.label }}</span>
              </router-link>
            </nav>
            <div class="px-3 pb-4 pt-2">
              <div class="rounded-[12px] border border-white/8 bg-white/4 p-3">
                <div class="flex items-center gap-2.5">
                  <div class="brand-mark flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-[12px] font-medium text-white">
                    {{ (auth.user?.display_name || auth.user?.username || "?").slice(0, 1) }}
                  </div>
                  <div class="min-w-0 flex-1">
                    <div class="truncate text-[12.5px]">{{ auth.user?.display_name || auth.user?.username }}</div>
                    <div class="truncate text-[10.5px] text-white/40">{{ auth.user?.department }}</div>
                  </div>
                  <span class="rounded-full px-1.5 py-0.5 text-[10px]" :class="roleBadge[auth.user?.role || 'user']">
                    {{ auth.user?.role_name || auth.user?.role }}
                  </span>
                </div>
                <div class="mt-2.5 space-y-1.5 border-t border-white/8 pt-2.5 text-[10.5px] text-white/40">
                  <div class="flex items-center gap-1.5">
                    <span class="inline-block h-1.5 w-1.5 rounded-full"
                          :class="health?.llm_configured ? 'bg-[#6fa287]' : 'bg-[#b4413c]'" />
                    DeepSeek {{ health?.llm_configured ? "已接入" : "未配置" }}
                    <span class="text-white/25">{{ health?.llm_model }}</span>
                  </div>
                  <div>短时记忆 50 轮</div>
                </div>
              </div>
              <button
                class="mt-2.5 w-full cursor-pointer rounded-[8px] border border-white/12 py-1.5 text-[12px] text-white/55 transition-colors hover:bg-white/8 hover:text-white"
                @click="onLogout">退出登录</button>
            </div>
          </aside>
          <!-- 内容区 -->
          <main class="min-w-0 flex-1 overflow-y-auto bg-background">
            <router-view />
          </main>
        </div>
        <div v-else class="flex h-screen items-center justify-center text-muted-foreground">加载中…</div>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>
