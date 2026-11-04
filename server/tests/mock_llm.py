"""Mock DeepSeek(OpenAI 兼容)服务,供回归测试使用。
按提示词特征分流:意图分类/查询改写/拒答/闲聊/RAG 回答/工具调用/图片标注。
支持 stream 与非 stream;工具选择轮为非流式,最终总结为流式。
"""
import json
import re
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

CAPTION = "示意图:请假审批流程,含提交申请、部门负责人审核、人力资源部备案三个环节。"


def _text_of(content):
    if isinstance(content, str):
        return content
    parts = []
    for c in content or []:
        if isinstance(c, dict) and c.get("type") == "text":
            parts.append(c.get("text", ""))
    return "\n".join(parts)


def _has_image(messages):
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for it in c:
                if isinstance(it, dict) and it.get("type") in ("image_url", "image"):
                    return True
    return False


def _pick_tool(last_user: str) -> dict:
    """按用户问题关键词选择工具与参数(与回归脚本的问题一一对应)。"""
    q = last_user
    if re.search(r"设备台账|车间设备|注塑机|数控机床", q):
        return {"kind": "tool", "name": "biz_mes_equipment", "args": {"ws": "二车间"}}
    if "会议纪要" in q or "技能" in q:
        return {"kind": "tool", "name": "use_skill", "args": {"name": "meeting-notes"}}
    if re.search(r"相差|几天|多少天|截止|推算|星期几|90 ?天", q):
        return {"kind": "tool", "name": "date_calc",
                "args": {"start": "2026-03-01", "end": "2026-05-30"}}
    if re.search(r"时间戳", q):
        return {"kind": "tool", "name": "timestamp_convert", "args": {"value": "1770000000"}}
    if re.search(r"个税|个人所得税|税后", q):
        return {"kind": "tool", "name": "tax_calc", "args": {"monthly_salary": 15000}}
    if re.search(r"换算|英里|华氏", q):
        return {"kind": "tool", "name": "unit_convert",
                "args": {"value": 15, "from_unit": "公里", "to_unit": "英里"}}
    if re.search(r"目录|章节|哪些方面", q):
        return {"kind": "tool", "name": "doc_outline", "args": {}}
    if re.search(r"哪些知识库|能查哪些|几个知识库", q):
        return {"kind": "tool", "name": "kb_list", "args": {}}
    if re.search(r"计算|等于多少", q):
        return {"kind": "tool", "name": "calculator", "args": {"expression": "1+1"}}
    return {"kind": "tool", "name": "current_time", "args": {}}


def decide(body: dict) -> dict:
    msgs = body.get("messages", [])
    full = "\n".join(_text_of(m.get("content")) for m in msgs)
    last_user = ""
    for m in reversed(msgs):
        if m.get("role") == "user":
            last_user = _text_of(m.get("content"))
            break
    has_tool_result = any(m.get("role") == "tool" for m in msgs)
    tools = body.get("tools") or []

    if _has_image(msgs):
        return {"kind": "text", "content": CAPTION}
    if "你是意图分类器" in full:
        q = last_user
        if re.search(r"天气|新闻|股票|写代码|翻译|足球|电影", q):
            intent = "out_of_scope"
        elif re.search(r"^(你好|您好|hi|hello|谢谢|感谢)", q):
            intent = "chitchat"
        elif re.search(r"几点|时间|计算|等于多少|多少篇|几个知识库|哪些知识库|网页|抓取|技能|整理"
                       r"|相差|几天|多少天|星期几|时间戳|个税|税后|换算|英里|华氏|目录|章节|哪些方面"
                       r"|设备台账|车间设备|注塑机|数控机床", q):
            intent = "tool"
        else:
            intent = "kb_query"
        return {"kind": "text", "content": json.dumps({"intent": intent}, ensure_ascii=False)}
    if "你是查询改写器" in full:
        return {"kind": "text", "content": (last_user or "请假审批流程是什么")}
    if "拒答模块" in full:
        return {"kind": "text", "content": "抱歉,该问题与知识库内容无关,请改问制度、流程相关问题。"}
    if "工具调用模块" in full and tools:
        if has_tool_result:
            return {"kind": "text", "content": "已通过工具获得结果,见上方工具返回内容。"}
        return _pick_tool(last_user)
    if "日常闲聊" in full:
        return {"kind": "text", "content": "你好!我是知识库智能问答助手,可以问我制度、流程相关的问题。"}
    if "基于《" in full or "参考资料" in full:
        return {"kind": "text",
                "content": "根据资料,员工请假须在 OA 系统提前提交申请,经部门负责人审核后报人力资源部备案,审批通过后方可离岗[1]。"}
    return {"kind": "text", "content": "连接正常"}


def _chunk(mid, model, delta, finish=None):
    return {"id": mid, "object": "chat.completion.chunk", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    d = decide(body)
    model = body.get("model", "mock")
    mid = "chatcmpl-" + uuid.uuid4().hex[:12]
    if d["kind"] == "tool":
        tool_call = {"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                     "function": {"name": d["name"],
                                  "arguments": json.dumps(d["args"], ensure_ascii=False)}}
        message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": d["content"]}
        finish = "stop"

    if body.get("stream"):
        def gen():
            if d["kind"] == "tool":
                tc = dict(tool_call)
                payload = _chunk(mid, model, {"role": "assistant", "tool_calls": [dict(tc, index=0)]}, finish)
                yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
            else:
                text = d["content"]
                half = max(1, len(text) // 2)
                for part in (text[:half], text[half:]):
                    if part:
                        yield "data: " + json.dumps(
                            _chunk(mid, model, {"role": "assistant", "content": part}),
                            ensure_ascii=False) + "\n\n"
                yield "data: " + json.dumps(_chunk(mid, model, {}, "stop")) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return JSONResponse({
        "id": mid, "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
    })


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "deepseek-reasoner"}, {"id": "deepseek-flash"}]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000, log_level="warning")
