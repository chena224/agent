"""端到端回归脚本:权限识别矩阵 + 业务系统数据对接 + 扫描件 OCR。

前置条件:
1. mock LLM 运行中:  python3 server/tests/mock_llm.py            # :9000
2. 应用运行中:       DATA_DIR=<带演示数据的目录> DEEPSEEK_API_KEY=sk-mock \
                     DEEPSEEK_BASE_URL=http://127.0.0.1:9000/v1 \
                     python3 -m uvicorn app.main:app --port 8100  # :8100
3. 业务库文件:       sqlite:////tmp/ragtest/biz_mes.db(见脚本内 ensure_biz_db)

运行: python3 server/tests/regression.py   (可用 APP_BASE 环境变量覆盖应用地址)
退出码: 0 = 全绿;1 = 存在失败项。
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import requests

BASE = os.environ.get("APP_BASE", "http://127.0.0.1:8100")
BIZ_DB = os.environ.get("BIZ_DB", "/tmp/ragtest/biz_mes.db")

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  | {detail}" if detail and not cond else ""))


def login(username: str, password: str) -> dict:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"username": username, "password": password}, timeout=10)
    assert r.status_code == 200, f"登录失败 {username}: {r.text}"
    return {"Authorization": f"Bearer {r.json()['token']}"}


def chat_events(headers: dict, message: str, kb_id: int) -> list[dict]:
    """发送聊天并收集 SSE 事件。"""
    r = requests.post(f"{BASE}/api/chat", headers=headers, stream=True, timeout=120,
                      json={"message": message, "kb_id": kb_id})
    events = []
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events, r.status_code


def ensure_biz_db() -> None:
    """模拟 MES 侧业务库(设备台账),连接器回归用。"""
    os.makedirs(os.path.dirname(BIZ_DB), exist_ok=True)
    conn = sqlite3.connect(BIZ_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS equipment_ledger("
                 "id INTEGER PRIMARY KEY, asset_no TEXT, name TEXT,"
                 " workshop TEXT, status TEXT, purchase_date TEXT)")
    conn.execute("DELETE FROM equipment_ledger")
    conn.executemany(
        "INSERT INTO equipment_ledger(asset_no,name,workshop,status,purchase_date)"
        " VALUES(?,?,?,?,?)",
        [(f"EQ-20{i:03d}", n, w, s, f"2024-0{i % 9 + 1}-15")
         for i, (n, w, s) in enumerate([
             ("数控车床", "一车间", "运行"), ("加工中心", "一车间", "运行"),
             ("注塑机", "二车间", "停机检修"), ("三坐标测量仪", "质检中心", "运行"),
             ("激光切割机", "二车间", "运行"), ("空压机", "动力站", "备用")] * 4, 1)])
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- 用例

def test_permissions(admin, manager, chen, li, wang):
    """A. 权限识别矩阵:RBAC 三角色 × 可见性 × 操作,含授权全链路(规格书 1.7)。"""
    # A01 三角色可见性基线
    kbs_admin = requests.get(f"{BASE}/api/kbs", headers=admin, timeout=10).json()["kbs"]
    check("A01a admin 可见全部知识库", len(kbs_admin) >= 2)
    check("A01b admin 均可管理", all(k.get("can_manage") for k in kbs_admin))
    kb2 = next(k for k in kbs_admin if k["id"] == 2)
    kb2_name = kb2["name"]

    kbs_chen = requests.get(f"{BASE}/api/kbs", headers=chen, timeout=10).json()["kbs"]
    check("A02a 生产制造部(部门授权)可见私有库", any(k["id"] == 2 for k in kbs_chen))
    check("A02b 普通用户无管理权", all(not k.get("can_manage") for k in kbs_chen))
    r = requests.get(f"{BASE}/api/kbs/2", headers=chen, timeout=10)
    check("A02c 部门授权者可读取库详情", r.status_code == 200)
    r = requests.patch(f"{BASE}/api/kbs/2", headers=chen, json={"name": "越权改名"}, timeout=10)
    check("A02d 普通用户改库被拒绝(403)", r.status_code == 403)

    kbs_wang = requests.get(f"{BASE}/api/kbs", headers=wang, timeout=10).json()["kbs"]
    check("A03a 设备动力部(无授权)仅见公开库", all(k["id"] != 2 for k in kbs_wang))
    r = requests.get(f"{BASE}/api/kbs/2", headers=wang, timeout=10)
    check("A03b 无授权读详情被拒(403)", r.status_code == 403)
    _, code = chat_events(wang, "你好", kb_id=2)
    check("A03c 无授权聊天被拒(403)", code == 403)

    kbs_li = requests.get(f"{BASE}/api/kbs", headers=li, timeout=10).json()["kbs"]
    check("A04 用户级授权者可见私有库", any(k["id"] == 2 for k in kbs_li))

    # A05 授权全链路:建私有库 → 不可见 → 授权部门 → 可见 → 撤销 → 不可见
    r = requests.post(f"{BASE}/api/kbs", headers=admin, timeout=10,
                      json={"name": "回归临时库", "description": "权限链路测试", "visibility": "private"})
    kb3 = r.json()["kb"]["id"]
    r = requests.get(f"{BASE}/api/kbs/{kb3}", headers=wang, timeout=10)
    check("A05a 新建私有库默认不可见", r.status_code == 403)
    r = requests.post(f"{BASE}/api/kbs/{kb3}/grants", headers=admin, timeout=10,
                      json={"target_type": "department", "target_value": "设备动力部"})
    check("A05b admin 授权成功", r.status_code == 200)
    r = requests.get(f"{BASE}/api/kbs/{kb3}", headers=wang, timeout=10)
    check("A05c 授权后可见(200)", r.status_code == 200)
    _, code = chat_events(wang, "你好", kb_id=kb3)
    check("A05d 授权后聊天通过权限门(400 未建索引≠403)", code == 400)
    gid = requests.get(f"{BASE}/api/kbs/{kb3}/grants", headers=admin, timeout=10).json()["grants"][0]["id"]
    requests.delete(f"{BASE}/api/kbs/{kb3}/grants/{gid}", headers=admin, timeout=10)
    r = requests.get(f"{BASE}/api/kbs/{kb3}", headers=wang, timeout=10)
    check("A05e 撤销后回到不可见(403)", r.status_code == 403)
    requests.delete(f"{BASE}/api/kbs/{kb3}", headers=admin, timeout=10)

    # A06 可见性优先级:私有库置公开后全员可见,置回私有恢复原判
    r = requests.patch(f"{BASE}/api/kbs/2", headers=manager,
                       json={"visibility": "public"}, timeout=10)
    check("A06a 工作空间管理员可管理自有库", r.status_code == 200)
    r = requests.get(f"{BASE}/api/kbs/2", headers=wang, timeout=10)
    check("A06b 公开后无授权用户可见", r.status_code == 200)
    requests.patch(f"{BASE}/api/kbs/2", headers=manager, json={"visibility": "private"}, timeout=10)
    r = requests.get(f"{BASE}/api/kbs/2", headers=wang, timeout=10)
    check("A06c 置回私有后恢复不可见", r.status_code == 403)

    # A07 工具层过滤:kb_list 只返回调用者可访问的库
    ev, code = chat_events(wang, "我现在能查哪些知识库", kb_id=1)
    previews = [e.get("preview", "") for e in ev if e.get("type") == "tool_call" and e.get("name") == "kb_list"]
    check("A07a 工具调用发生(kb_list)", bool(previews))
    check("A07b 无授权者的 kb_list 不含私有库", previews and kb2_name not in previews[-1])
    ev, _ = chat_events(chen, "我现在能查哪些知识库", kb_id=1)
    previews = [e.get("preview", "") for e in ev if e.get("type") == "tool_call" and e.get("name") == "kb_list"]
    check("A07c 有授权者的 kb_list 含私有库", previews and kb2_name in previews[-1])

    # A08/A09 管理端点与越权操作
    r = requests.get(f"{BASE}/api/admin/users", headers=chen, timeout=10)
    check("A08a 普通用户访问用户管理被拒(403)", r.status_code == 403)
    r = requests.post(f"{BASE}/api/kbs/2/grants", headers=chen, timeout=10,
                      json={"target_type": "user", "target_value": "wang"})
    check("A08b 普通用户做授权被拒(403)", r.status_code == 403)
    r = requests.delete(f"{BASE}/api/kbs/1", headers=chen, timeout=10)
    check("A09 普通用户删库被拒(403)", r.status_code == 403)
    r = requests.get(f"{BASE}/api/kbs", timeout=10)
    check("A10 未登录访问被拒(401)", r.status_code == 401)


def test_connectors(admin, chen):
    """B. 业务系统数据对接:配置预览 + 只读策略 + CRUD + 工具化(规格书 1.5/1.6)。"""
    db_cfg = {"db_url": f"sqlite:///{BIZ_DB}",
              "query": "SELECT asset_no,name,workshop,status FROM equipment_ledger"
                       " WHERE (:ws = '' OR workshop = :ws)",
              "params": [{"name": "ws", "label": "车间", "type": "string",
                          "required": False, "default": ""}],
              "max_rows": 10}
    # B01 未保存配置直接预览(用户点击即可看数据)
    r = requests.post(f"{BASE}/api/connectors/preview", headers=admin, timeout=30,
                      json={"type": "database", "config": db_cfg, "args": {"ws": "二车间"}}).json()
    check("B01 表单内测试连接可预览数据", r.get("ok") and len(r.get("rows", [])) > 0
          and "asset_no" in r.get("columns", []))

    # B02 只读策略:写库语句逐一拒绝
    for sql in ("DELETE FROM equipment_ledger",
                "UPDATE equipment_ledger SET status='x'",
                "DROP TABLE equipment_ledger",
                "SELECT 1; DELETE FROM equipment_ledger",
                "PRAGMA journal_mode=OFF"):
        r = requests.post(f"{BASE}/api/connectors/preview", headers=admin, timeout=30,
                          json={"type": "database",
                                "config": {"db_url": f"sqlite:///{BIZ_DB}", "query": sql, "params": []},
                                "args": {}}).json()
        check(f"B02 只读策略拒绝[{sql[:24]}…]", not r.get("ok"), r.get("error", "") if r.get("ok") else "")
    # B03 字面量中的敏感词不误杀
    r = requests.post(f"{BASE}/api/connectors/preview", headers=admin, timeout=30,
                      json={"type": "database",
                            "config": {"db_url": f"sqlite:///{BIZ_DB}",
                                       "query": "SELECT 'drop table 是字符串' AS note", "params": []},
                            "args": {}}).json()
    check("B03 字符串字面量不误杀", r.get("ok") and r.get("rows", [[""]])[0][0].startswith("drop"))

    # B04 CRUD 全周期
    conn_spec = {"name": "reg_mes", "label": "回归·设备台账", "description": "回归测试用",
                 "type": "database", "config": db_cfg, "enabled": True}
    r = requests.post(f"{BASE}/api/connectors", headers=admin, json=conn_spec, timeout=10)
    check("B04a 创建连接器", r.status_code == 200)
    cid = r.json().get("id")
    r = requests.post(f"{BASE}/api/connectors", headers=admin, json=conn_spec, timeout=10)
    check("B04b 重名冲突(409)", r.status_code == 409)
    r = requests.post(f"{BASE}/api/connectors/{cid}/test", headers=admin, json={"args": {}}, timeout=30).json()
    check("B04c 已保存连接器测试连接", r.get("ok") and r.get("truncated") is True)
    r = requests.patch(f"{BASE}/api/connectors/{cid}", headers=admin, json={"enabled": False}, timeout=10)
    check("B04d 停用", r.status_code == 200 and r.json().get("enabled") is False)

    # B05 REST 连接器(键值规整)
    r = requests.post(f"{BASE}/api/connectors", headers=admin, timeout=10, json={
        "name": "reg_health", "label": "回归·健康检查", "description": "", "type": "rest",
        "config": {"method": "GET", "url": f"{BASE}/api/health", "headers": {}, "params": []}})
    rid = r.json().get("id")
    r = requests.post(f"{BASE}/api/connectors/{rid}/test", headers=admin, json={"args": {}}, timeout=30).json()
    check("B05 REST 连接器预览", r.get("ok") and r.get("columns") == ["字段", "值"])

    # B06 参数与模板校验
    r = requests.post(f"{BASE}/api/connectors/preview", headers=admin, timeout=30, json={
        "type": "rest", "config": {"method": "GET", "url": f"{BASE}/api/health?x={{missing}}",
                                   "params": [{"name": "missing", "label": "缺失", "required": True}]},
        "args": {}}).json()
    check("B06a 缺必填参数有友好提示", not r.get("ok") and "缺少" in r.get("error", ""))
    r = requests.post(f"{BASE}/api/connectors", headers=admin, timeout=10,
                      json={"name": "bad_scheme", "type": "rest",
                            "config": {"method": "GET", "url": "file:///etc/passwd", "params": []}})
    check("B06b 非 http/https 协议拒绝(400)", r.status_code == 400)

    # B07 普通用户无管理权
    r = requests.get(f"{BASE}/api/connectors", headers=chen, timeout=10)
    check("B07 普通用户访问连接器管理被拒(403)", r.status_code == 403)

    # B08 工具化 E2E:启用连接器后,业务问题经 ReAct 调用 biz_ 工具并带回数据
    requests.patch(f"{BASE}/api/connectors/{cid}", headers=admin, json={"enabled": True}, timeout=10)
    # 工具名固定 biz_mes_equipment(与 mock_llm 的关键词分支对应):重建一个同名连接器
    r = requests.post(f"{BASE}/api/connectors", headers=admin, timeout=10, json={
        "name": "mes_equipment", "label": "设备台账查询(MES)",
        "description": "查询生产设备的台账信息,回答设备运行状态类问题",
        "type": "database", "config": db_cfg, "enabled": True})
    if r.status_code == 409:   # 已存在则确保启用
        cid2 = next(c["id"] for c in requests.get(f"{BASE}/api/connectors", headers=admin).json()["items"]
                    if c["name"] == "mes_equipment")
        requests.patch(f"{BASE}/api/connectors/{cid2}", headers=admin, json={"enabled": True}, timeout=10)
    ev, code = chat_events(admin, "查一下二车间的注塑机设备台账", kb_id=1)
    calls = [e for e in ev if e.get("type") == "tool_call" and e.get("name") == "biz_mes_equipment"]
    check("B08a 业务问题触发连接器工具", bool(calls))
    check("B08b 工具返回真实业务数据", calls and "注塑机" in (calls[-1].get("preview") or ""))
    check("B08c 工具调用状态闭环(ok)", calls and calls[-1].get("status") == "ok")

    # 清理
    for c in requests.get(f"{BASE}/api/connectors", headers=admin, timeout=10).json()["items"]:
        if c["name"] in ("reg_mes", "reg_health"):
            requests.delete(f"{BASE}/api/connectors/{c['id']}", headers=admin, timeout=10)


def make_scan_pdf(path: str) -> None:
    """现场造一份两页纯扫描 PDF(整页图片、零文本层),OCR 回归用。"""
    import io
    import pymupdf
    from PIL import Image, ImageDraw, ImageFont

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    pages = [
        ["设备点检管理制度",
         "第一条 注塑机每班开机前必须点检",
         "第二条 点检异常立即报设备动力部",
         "第三条 点检记录保存三年备查"],
        ["安全生产补充规定",
         "第十条 进入车间必须佩戴安全帽",
         "第十一条 特种作业须持证上岗"],
    ]
    doc = pymupdf.open()
    fnt = ImageFont.truetype(font_path, 40)
    for lines in pages:
        page = doc.new_page(width=595, height=842)
        img = Image.new("RGB", (1190, 1684), "white")
        d = ImageDraw.Draw(img)
        y = 120
        for ln in lines:
            d.text((100, y), ln, fill="black", font=fnt)
            y += 110
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        page.insert_image(pymupdf.Rect(20, 20, 575, 822), stream=buf.getvalue())
    doc.save(path)
    doc.close()


def wait_build(kb_id: int, headers: dict, timeout_s: int = 600) -> dict:
    import time
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        p = requests.get(f"{BASE}/api/kbs/{kb_id}/index/progress",
                         headers=headers, timeout=10).json()
        if p.get("stage") in ("done", "error", "idle") and p.get("percent", 0) in (0, 100):
            return p
        if p.get("stage") == "error":
            return p
        time.sleep(2)
    return {"stage": "timeout"}


def test_scan_ocr(admin):
    """C. 扫描件 OCR:纯扫描 PDF 全文转录并可检索(规格书 1.3 文档解析)。"""
    os.makedirs("/tmp/ragtest", exist_ok=True)
    pdf_path = "/tmp/ragtest/scan_reg.pdf"
    make_scan_pdf(pdf_path)

    r = requests.post(f"{BASE}/api/kbs", headers=admin, timeout=10,
                      json={"name": "回归扫描件库", "description": "OCR 回归", "visibility": "private"})
    kb = r.json()["kb"]["id"]
    try:
        with open(pdf_path, "rb") as f:
            r = requests.post(f"{BASE}/api/kbs/{kb}/upload", headers=admin,
                              files={"file": ("scan_reg.pdf", f, "application/pdf")}, timeout=120)
        check("C01 扫描 PDF 上传成功", r.status_code == 200, r.text[:120])

        r = requests.post(f"{BASE}/api/kbs/{kb}/index/build", headers=admin,
                          json={"strategy": "structural", "params": {}}, timeout=30)
        check("C02 扫描库构建受理", r.status_code == 200, r.text[:120])
        prog = wait_build(kb, admin)
        check("C03 扫描库构建完成", prog.get("stage") == "done",
              str(prog.get("message", prog.get("stage"))))

        r = requests.post(f"{BASE}/api/kbs/{kb}/chunks/preview", headers=admin,
                          json={"strategy": "structural", "params": {}, "offset": 0, "limit": 50},
                          timeout=60)
        body = r.json()
        haystack = json.dumps(body.get("chunks", []), ensure_ascii=False)
        check("C04 OCR 文本进入切分(条款关键字)", "点检异常立即报设备动力部" in haystack)
        check("C05 预览提示扫描页已转录", "扫描页" in str((body.get("stats") or {}).get("note", "")))

        r = requests.post(f"{BASE}/api/kbs/{kb}/retrieve", headers=admin,
                          json={"query": "点检记录要保存多久", "top_k": 5}, timeout=60)
        hits = json.dumps(r.json().get("hits", []), ensure_ascii=False)
        check("C06 扫描内容可检索命中", "保存三年备查" in hits, hits[:120])
    finally:
        requests.delete(f"{BASE}/api/kbs/{kb}", headers=admin, timeout=10)


def main() -> int:
    ensure_biz_db()
    admin = login("admin", "admin123")
    manager = login("manager", "manager123")
    chen = login("chen", "chen123")
    li = login("li", "li123")
    wang = login("wang", "wang123")
    test_permissions(admin, manager, chen, li, wang)
    test_connectors(admin, chen)
    test_scan_ocr(admin)
    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'=' * 60}\n总计 {len(CHECKS)} 项,通过 {len(CHECKS) - len(failed)},失败 {len(failed)}")
    for name, _, detail in failed:
        print(f"  FAIL {name}  {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
