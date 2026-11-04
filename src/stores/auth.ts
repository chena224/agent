import { defineStore } from "pinia";
import { api, TOKEN_KEY, type UserInfo } from "@/api";

export const useAuthStore = defineStore("auth", {
  state: () => ({
    token: localStorage.getItem(TOKEN_KEY) || "",
    user: null as UserInfo | null,
    ready: false,          // 首次 /me 拉取完成
  }),
  getters: {
    loggedIn: (s) => !!s.token && !!s.user,
    isAdmin: (s) => s.user?.role === "admin",
    isManager: (s) => s.user?.role === "admin" || s.user?.role === "workspace_admin",
  },
  actions: {
    async restore() {
      if (!this.token) { this.ready = true; return; }
      try {
        const r = await api.me();
        this.user = r.user;
      } catch {
        this.token = "";
        localStorage.removeItem(TOKEN_KEY);
      } finally {
        this.ready = true;
      }
    },
    async login(username: string, password: string) {
      const r = await api.login(username, password);
      this.token = r.token;
      this.user = r.user;
      localStorage.setItem(TOKEN_KEY, r.token);
    },
    async logout() {
      try { await api.logout(); } catch { /* 网络异常也强制登出 */ }
      this.token = "";
      this.user = null;
      localStorage.removeItem(TOKEN_KEY);
    },
  },
});
