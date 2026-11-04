import { createRouter, createWebHashHistory } from "vue-router";
import { useAuthStore } from "@/stores/auth";

/** 静态托管无服务端重写,用 hash 路由保证直接刷新不空白。 */
export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/login", name: "login", component: () => import("@/pages/LoginPage.vue"), meta: { public: true } },
    { path: "/", name: "chat", component: () => import("@/pages/ChatPage.vue") },
    { path: "/knowledge", name: "knowledge", component: () => import("@/pages/KnowledgePage.vue"), meta: { roles: ["admin", "workspace_admin"] } },
    { path: "/orchestration", name: "orchestration", component: () => import("@/pages/OrchestraPage.vue"), meta: { roles: ["admin", "workspace_admin"] } },
    { path: "/tools", name: "tools", component: () => import("@/pages/ToolsPage.vue"), meta: { roles: ["admin", "workspace_admin"] } },
    { path: "/stats", name: "stats", component: () => import("@/pages/StatsPage.vue"), meta: { roles: ["admin", "workspace_admin"] } },
    { path: "/org", name: "org", component: () => import("@/pages/AdminPage.vue"), meta: { roles: ["admin"] } },
    { path: "/system", name: "system", component: () => import("@/pages/SettingsPage.vue"), meta: { roles: ["admin"] } },
    { path: "/:pathMatch(.*)*", redirect: "/" },
  ],
});

router.beforeEach(async (to) => {
  const auth = useAuthStore();
  if (!auth.ready) await auth.restore();
  if (to.meta.public) {
    if (auth.loggedIn) return { path: "/" };
    return true;
  }
  if (!auth.loggedIn) return { path: "/login" };
  const roles = to.meta.roles as string[] | undefined;
  if (roles && auth.user && !roles.includes(auth.user.role)) return { path: "/" };
  return true;
});
