<script setup lang="ts">
import { onMounted, reactive, ref } from "vue";
import { useMessage } from "naive-ui";
import { api, type KBItem, type UserInfo } from "@/api";
import { useAuthStore } from "@/stores/auth";

const message = useMessage();
const auth = useAuthStore();

const users = ref<UserInfo[]>([]);
const roles = ref<Record<string, string>>({});
const departments = ref<string[]>([]);
const kbs = ref<KBItem[]>([]);

interface Grant { id: number; target_type: string; target_value: string; created_at: number }
const grantKbId = ref(0);
const grants = ref<Grant[]>([]);
const grantForm = reactive({ target_type: "department", target_value: "" });

const showCreate = ref(false);
const createForm = reactive({ username: "", password: "", display_name: "", department: "", role: "user" });

const ROLE_CLS: Record<string, string> = {
  admin: "bg-[#b4413c]/10 text-[#b4413c]",
  workspace_admin: "bg-[#3f6f5f]/10 text-[#3f6f5f]",
  user: "bg-secondary text-muted-foreground",
};

async function load() {
  try {
    const [u, k] = await Promise.all([api.adminUsers(), api.kbs()]);
    users.value = u.users;
    roles.value = u.roles;
    departments.value = u.departments;
    kbs.value = k.kbs;
    if (k.kbs.length && !k.kbs.find((x) => x.id === grantKbId.value)) grantKbId.value = k.kbs[0].id;
    if (grantKbId.value) await loadGrants();
  } catch (e) { message.error((e as Error).message); }
}
onMounted(load);

async function loadGrants() {
  if (!grantKbId.value) return;
  try { grants.value = (await api.grants(grantKbId.value)).grants; }
  catch { grants.value = []; }
}

/* ------------------------------------------------------------ 用户操作 */

async function patchUser(u: UserInfo, patch: Record<string, unknown>, okMsg: string) {
  try {
    await api.updateUser(u.id, patch);
    await load();
    message.success(okMsg);
  } catch (e) { message.error((e as Error).message); await load(); }
}

async function resetPassword(u: UserInfo) {
  const pwd = prompt(`为「${u.display_name || u.username}」设置新口令(至少 6 位):`);
  if (!pwd) return;
  if (pwd.length < 6) { message.warning("口令至少 6 位"); return; }
  await patchUser(u, { password: pwd }, "口令已重置");
}

async function createUser() {
  if (!createForm.username.trim() || !createForm.password) { message.warning("请填写用户名与口令"); return; }
  try {
    await api.createUser({
      username: createForm.username.trim(), password: createForm.password,
      display_name: createForm.display_name.trim(), department: createForm.department.trim(),
      role: createForm.role,
    });
    showCreate.value = false;
    Object.assign(createForm, { username: "", password: "", display_name: "", department: "", role: "user" });
    await load();
    message.success("用户已创建");
  } catch (e) { message.error((e as Error).message); }
}

/* ------------------------------------------------------------ 授权操作 */

async function toggleVisibility(kb: KBItem) {
  try {
    await api.updateKb(kb.id, { visibility: kb.visibility === "public" ? "private" : "public" });
    await load();
    message.success("可见性已更新");
  } catch (e) { message.error((e as Error).message); }
}

async function addGrant() {
  if (!grantForm.target_value.trim()) { message.warning("请选择授权对象"); return; }
  try {
    await api.addGrant(grantKbId.value, {
      target_type: grantForm.target_type, target_value: grantForm.target_value.trim(),
    });
    grantForm.target_value = "";
    await loadGrants();
    message.success("授权已生效(优先级:用户级 > 部门级 > 公开)");
  } catch (e) { message.error((e as Error).message); }
}

async function revokeGrant(g: Grant) {
  if (!confirm(`确认撤销对「${g.target_value}」的授权?`)) return;
  try {
    await api.removeGrant(grantKbId.value, g.id);
    await loadGrants();
    message.success("已撤销");
  } catch (e) { message.error((e as Error).message); }
}

const grantKb = () => kbs.value.find((k) => k.id === grantKbId.value);
</script>

<template>
  <div class="mx-auto max-w-[1180px] px-6 py-6">
    <div class="mb-5">
      <h1 class="text-[17px] font-semibold">组织机构与权限管理</h1>
      <p class="mt-1 text-[12px] text-muted-foreground">对应规格书 §1.7:RBAC 三角色(系统管理员/工作空间管理员/普通用户),资源授权按 用户级 &gt; 部门级 &gt; 公开 判定</p>
    </div>

    <div class="grid gap-4 lg:grid-cols-[1fr_360px]">
      <!-- 用户管理 -->
      <section class="panel-shadow rounded-[12px] border border-border bg-card p-4">
        <div class="flex items-center justify-between">
          <h2 class="text-[13.5px] font-medium">用户管理</h2>
          <button class="cursor-pointer rounded-[7px] bg-foreground px-3 py-1.5 text-[11.5px] text-primary-foreground transition-colors hover:bg-foreground/85"
                  @click="showCreate = true">+ 新建用户</button>
        </div>
        <div class="mt-3 overflow-x-auto">
          <table class="data-table">
            <thead>
              <tr><th>用户名</th><th>姓名</th><th>部门</th><th>角色</th><th>状态</th><th class="text-right">操作</th></tr>
            </thead>
            <tbody>
              <tr v-for="u in users" :key="u.id">
                <td class="font-mono-num">{{ u.username }}</td>
                <td>{{ u.display_name || '—' }}</td>
                <td>
                  <input :value="u.department" list="dept-options"
                         class="h-7 w-28 rounded-[6px] border border-transparent bg-transparent px-1.5 text-[12.5px] outline-none transition-colors hover:border-input focus:border-foreground/50 focus:bg-background"
                         @change="patchUser(u, { department: ($event.target as HTMLInputElement).value.trim() }, '部门已更新')" />
                </td>
                <td>
                  <select :value="u.role" :disabled="u.id === auth.user?.id"
                          class="h-7 rounded-[6px] border border-transparent bg-transparent px-1 text-[12px] outline-none hover:border-input focus:border-foreground/50 focus:bg-background disabled:opacity-50"
                          :class="ROLE_CLS[u.role]"
                          @change="patchUser(u, { role: ($event.target as HTMLSelectElement).value }, '角色已更新')">
                    <option v-for="(label, r) in roles" :key="r" :value="r">{{ label }}</option>
                  </select>
                </td>
                <td>
                  <button :disabled="u.id === auth.user?.id"
                          class="cursor-pointer rounded-full px-2 py-0.5 text-[11px] transition-colors disabled:cursor-not-allowed disabled:opacity-50"
                          :class="u.status === 1 ? 'bg-[#3f6f5f]/10 text-[#3f6f5f]' : 'bg-[#b4413c]/10 text-[#b4413c]'"
                          @click="patchUser(u, { status: u.status === 1 ? 0 : 1 }, u.status === 1 ? '已停用' : '已启用')">
                    {{ u.status === 1 ? '启用中' : '已停用' }}
                  </button>
                </td>
                <td class="text-right">
                  <button class="cursor-pointer rounded border border-border px-2 py-0.5 text-[11px] transition-colors hover:bg-secondary/70"
                          @click="resetPassword(u)">重置口令</button>
                </td>
              </tr>
            </tbody>
          </table>
          <datalist id="dept-options">
            <option v-for="d in departments" :key="d" :value="d" />
          </datalist>
        </div>
      </section>

      <!-- 资源授权 -->
      <section class="h-fit panel-shadow rounded-[12px] border border-border bg-card p-4">
        <h2 class="text-[13.5px] font-medium">知识库资源授权</h2>
        <select v-model.number="grantKbId"
                class="mt-3 h-9 w-full rounded-[7px] border border-input bg-background px-2 text-[12.5px] outline-none focus:border-foreground/50"
                @change="loadGrants">
          <option v-for="k in kbs" :key="k.id" :value="k.id">{{ k.name }}</option>
        </select>
        <div v-if="grantKb()" class="mt-3 flex items-center justify-between rounded-[8px] bg-secondary/50 px-3 py-2">
          <span class="text-[12px]">可见性:{{ grantKb()!.visibility === 'public' ? '公开(全员)' : '私有(按授权)' }}</span>
          <button class="cursor-pointer rounded border border-border bg-card px-2 py-0.5 text-[11px] transition-colors hover:bg-secondary/70"
                  @click="toggleVisibility(grantKb()!)">切换为{{ grantKb()!.visibility === 'public' ? '私有' : '公开' }}</button>
        </div>

        <div class="mt-3 space-y-1.5">
          <div v-if="!grants.length" class="rounded-[8px] border border-dashed border-border px-3 py-4 text-center text-[11.5px] text-muted-foreground">暂无授权记录</div>
          <div v-for="g in grants" :key="g.id"
               class="flex items-center gap-2 rounded-[7px] border border-border px-2.5 py-1.5">
            <span class="rounded px-1.5 py-0.5 text-[10.5px]"
                  :class="g.target_type === 'user' ? 'bg-[#4a6d8c]/10 text-[#41608c]' : 'bg-[#3f6f5f]/10 text-[#3f6f5f]'">
              {{ g.target_type === 'user' ? '用户' : '部门' }}
            </span>
            <span class="flex-1 text-[12.5px]">{{ g.target_value }}</span>
            <button class="cursor-pointer text-[11px] text-muted-foreground hover:text-destructive" @click="revokeGrant(g)">撤销</button>
          </div>
        </div>

        <div class="mt-3 border-t border-border pt-3">
          <div class="mb-1.5 text-[11.5px] text-muted-foreground">新增授权</div>
          <div class="flex gap-1.5">
            <select v-model="grantForm.target_type" class="h-8 rounded-[6px] border border-input bg-background px-1.5 text-[12px] outline-none">
              <option value="department">按部门</option>
              <option value="user">按用户</option>
            </select>
            <input v-model="grantForm.target_value" :list="grantForm.target_type === 'department' ? 'dept-options' : 'user-options'"
                   :placeholder="grantForm.target_type === 'department' ? '选择部门' : '选择用户名'"
                   class="h-8 min-w-0 flex-1 rounded-[6px] border border-input bg-background px-2 text-[12px] outline-none focus:border-foreground/50" />
            <datalist id="user-options">
              <option v-for="u in users" :key="u.username" :value="u.username">{{ u.display_name }}</option>
            </datalist>
            <button class="cursor-pointer rounded-[6px] bg-foreground px-3 text-[12px] text-primary-foreground transition-colors hover:bg-foreground/85"
                    @click="addGrant">授权</button>
          </div>
          <p class="mt-2 text-[10.5px] leading-4 text-muted-foreground">判定优先级:用户级授权 &gt; 部门级授权 &gt; 公开可见;管理员角色永远放行。授权变更即时生效并写入审计日志。</p>
        </div>
      </section>
    </div>

    <!-- 新建用户弹窗 -->
    <div v-if="showCreate" class="fixed inset-0 z-40 flex items-center justify-center bg-black/40 p-4"
         @click.self="showCreate = false">
      <div class="w-full max-w-[400px] rounded-[12px] border border-border bg-card p-5 shadow-lg">
        <div class="flex items-center justify-between">
          <h3 class="text-[14.5px] font-medium">新建用户</h3>
          <button class="cursor-pointer text-muted-foreground hover:text-foreground" @click="showCreate = false">×</button>
        </div>
        <div class="mt-4 space-y-3 text-[12.5px]">
          <label class="block"><span class="mb-1 block text-[11.5px] text-muted-foreground">用户名</span>
            <input v-model="createForm.username" class="h-9 w-full rounded-[7px] border border-input bg-background px-2.5 outline-none focus:border-foreground/50" /></label>
          <label class="block"><span class="mb-1 block text-[11.5px] text-muted-foreground">初始口令</span>
            <input v-model="createForm.password" type="password" class="h-9 w-full rounded-[7px] border border-input bg-background px-2.5 outline-none focus:border-foreground/50" /></label>
          <label class="block"><span class="mb-1 block text-[11.5px] text-muted-foreground">姓名</span>
            <input v-model="createForm.display_name" class="h-9 w-full rounded-[7px] border border-input bg-background px-2.5 outline-none focus:border-foreground/50" /></label>
          <label class="block"><span class="mb-1 block text-[11.5px] text-muted-foreground">部门</span>
            <input v-model="createForm.department" list="dept-options" class="h-9 w-full rounded-[7px] border border-input bg-background px-2.5 outline-none focus:border-foreground/50" /></label>
          <div>
            <span class="mb-1.5 block text-[11.5px] text-muted-foreground">角色</span>
            <div class="grid grid-cols-3 gap-1.5">
              <button v-for="(label, r) in roles" :key="r"
                      class="cursor-pointer rounded-[7px] border px-2 py-1.5 text-[12px] transition-colors"
                      :class="createForm.role === r ? 'border-foreground/50 bg-secondary font-medium' : 'border-border'"
                      @click="createForm.role = r">{{ label }}</button>
            </div>
          </div>
          <button class="w-full cursor-pointer rounded-[8px] bg-foreground py-2 text-[13px] text-primary-foreground transition-colors hover:bg-foreground/85"
                  @click="createUser">创建</button>
        </div>
      </div>
    </div>
  </div>
</template>
