# 知识库智能体平台(工厂AI智能体本地化部署项目 · Demo)

通用 RAG 知识库问答平台。内置企业知识库(《员工规章制度手册》,章节条文+表格)
作为内置样例数据,用于验证多种切分策略与检索质量;换成任意规章制度、工作手册、
产品文档类 PDF/DOCX 即可复用整条链路——文中图片会由视觉模型生成文字描述后
一并参与索引(可在知识库级关闭)。

核心链路:PDF 解析 → 多策略切分 → 混合索引(向量 + BM25)→ LangGraph 编排
(意图识别 → 查询改写 → 混合检索 → 生成)→ 编号引用溯源 → 50 轮记忆 → 埋点统计。

检索与生成质量设计参考了 RAGFlow / Dify / FastGPT / Langchain-Chatchat / kotaemon
等高 star 开源项目的通行做法,见文末「质量设计说明」。

## 技术栈

| 层 | 选型 |
|---|---|
| 大模型 | DeepSeek(OpenAI 兼容接口,默认 `deepseek-chat`,可配置) |
| 编排 | LangChain + LangGraph(意图识别 → 查询改写 → 检索 → 生成 状态图) |
| 向量库 | Milvus Lite / 独立 Milvus(cluster),每知识库独立集合 `kb_{id}` |
| 全文检索 | BM25(rank_bm25 + jieba 中文分词,纯 CPU) |
| 向量模型 | BGE-small-zh-v1.5(fastembed/ONNX,CPU 即可运行) |
| 精排(可选) | BGE-Reranker-base(ONNX 本地推理,模型缺失时自动回退) |
| 后端 | FastAPI(SSE 流式问答,Bearer 令牌认证) |
| 前端 | Vue 3 + Vite + Pinia + Vue Router + Tailwind + Naive UI |
| 记忆 | LangGraph 检查点(SqliteSaver / cluster 档 PostgresSaver),三级隔离,保留最近 50 轮 |
| 权限 | RBAC 三角色 + 按用户/部门两级授权(SQLite / IvorySQL,PBKDF2 口令哈希) |
| 埋点 | SQLite / IvorySQL 记录提问 / Token / 耗时 / 意图 / 主备 Key 通道(规格书 §1.10) |

## 功能

- **多知识库**:新建/删除知识库、上传 PDF 换文档、公开/私有可见性;
  每库独立 Milvus 集合 + 独立 BM25 索引,访问互相隔离。
- **文档图片理解(视觉模型,可关)**:解析 PDF 时按阅读顺序把图片以
  `[[IMG:id]]` 占位标记注入文本流,原图落盘;**构建索引时**用视觉模型
  (默认 `deepseek-flash`,设置页可换)结合上下文为每张图生成 1~3 句
  中文描述,描述就地回填(〔图片 id〕描述:…),随后与正文一起切分索引。
  小图标(<100px)与超大图(>8MB)自动过滤,内容相同图片按 sha1 去重;
  图片描述在切分层作为**原子块**保护,不被拆散,命中后可溯源到原图
  (聊天引用面板直接展示)。描述按库缓存,重建不重复计费;知识库检索设置里
  可关闭(关闭后仅保留占位标记),改动重建索引生效;未配置 API Key 时
  自动跳过并在分段预览给出提示。对话侧图片上传暂不支持(后续批次)。
- **六种切分策略**(对齐 LangChain 能力,可预览对比、参数可调):
  条目结构(第X条边界,法规类最优)/ 递归字符 / 重叠窗口滑窗(重叠区长度
  可调,填 0 即直接截断)/ Token 切分(tiktoken,按 Token 数直接截断、
  无重叠)/ 父子块(ParentDocumentRetriever 模式)/ 语义切分。
  调整策略或参数后需重新构建索引生效。
- **混合检索**:语义(向量)/ 全文(BM25)/ 混合(RRF 融合)三种模式,
  TopK、Score 阈值、BM25 权重按知识库可调;最佳命中余弦过低时给出
  低相关告警;可选 BGE 精排。
- **意图识别三级漏斗**(参考 aurelio-labs/semantic-router):
  规则层(问候/明确越界/敏感词关键词,零开销短路) →
  语义路由(示例句向量余弦匹配,毫秒级、0 次 LLM 调用;LRU 缓存 + 配置变更
  主动失效) → LLM 兜底(JSON 结构化输出 + 解析容错,可看到与生成侧一致的
  对话上下文)。每个意图可配置**检索增强关键词**,命中后注入 BM25 分支做查询
  扩展——不同意图命中时检索侧重不同(参考其 multi_domain_routing 思路);
  LLM 兜底分支用 jieba 提取问题实义词做兜底加权。知识库页可增删改意图,
  配置写错(非法名/遮蔽内置名/空示例句)会立即 400 报错而非静默跑偏。
  **命中阈值按知识库可配**(检索设置 route_threshold,默认 0.45)——相似度
  绝对值依赖各库示例句与领域用词,全局一个值调不平所有库;**超纲词表按库
  可配**(留空用内置默认词表),敏感词表内置写死。
- **工具调用**(LangGraph tool 节点 + 条件边,独立于检索链路):
  内置 tool 意图路由(查时间/数学计算/抓取网页/统计知识库等请求),
  命中后进入工具智能体循环(≤4 轮 function calling,失败降级说明)。
  内置 10 个常用工具,全部免 Key、纯本地执行、可审计:当前时间、
  计算器(AST 白名单求值)、日期推算(间隔/加减 N 天/星期)、
  时间戳转换、单位换算(长度/重量/面积/体积/温度/存储)、
  个税估算(月度预扣率表)、网页抓取、知识库统计、知识库列表、
  文档目录;其中知识库统计/列表/文档目录按调用者 RBAC 过滤,
  私有库存在性不外泄。内置工具与 **MCP 工具融合为同一注册表**——
  设置页粘贴 MCP servers JSON(stdio/sse/streamable_http)即接入社区
  现成服务。工具总开关、工具链路模型(默认 deepseek-chat,不产生思考
  展示;检索/闲聊链路仍可用推理模型)均可在设置页调整。前端以折叠块
  展示每次工具调用的名称/参数/状态/结果预览。
- **技能(Skills,指令包)**:SKILL.md 目录式技能——frontmatter 声明
  name/description,正文是给模型的流程与模板指令,与 MCP(给模型工具)
  互补。采用**渐进式披露**:系统提示只列名称与描述,命中后模型调用
  `use_skill` 按需拉取完整指令执行;内置三个示例(网页总结/会议纪要/
  知识库概况报告,均免 Key),设置页可启停、上传(.md/.zip,含路径
  穿越/大小/同名校验)、删除;全局开关关闭时技能清单与 use_skill
  同步下线。本环境只加载指令文本,技能目录中的脚本不会被执行——
  上传权限收敛到系统管理员。
- **智能问答**:SSE 流式;多轮对话指代改写(待改写问题以 user 角色传入);
  回答内联编号引用 [1][2],点击定位来源;资料不足时拒答;越界/敏感请求
  默认走确定性拒答模板(不把越界原文喂给 LLM;`REFUSAL_LLM=1` 可开启
  LLM 话术生成,且不回传用户原文与历史)。模型设为推理模型
  (如 `deepseek-reasoner`)时,思考过程(reasoning_content)与回答分通道
  流式下发,前端独立折叠展示;普通模型无此输出,界面不出现思考区块。
- **权限管理**(规格书 §1.7):系统管理员/工作空间管理员/普通用户三角色;
  知识库按「用户」或「部门」授权,优先级:用户授权 > 部门授权 > 公开可见;
  LDAP 为预留扩展点(见 `db.py` authenticate 注释)。
- **统计分析**:提问/Token/耗时总览、近 14 天趋势、意图分布、分知识库用量、
  最近问答记录(含用户/知识库/意图)。
- **设置**:DeepSeek Key 配置与连通性测试;对话模型与视觉模型(图片描述用,
  默认 `deepseek-flash`,须支持图片输入)分别可调;TopK、记忆轮数调节。

## 内置账号

| 账号 | 密码 | 角色 | 可见知识库 |
|---|---|---|---|
| admin | admin123 | 系统管理员 | 全部(可管用户/设置) |
| manager | manager123 | 工作空间管理员 | 全部(可管知识库/授权) |
| chen | chen123 | 普通用户(生产制造部) | 公开库 + 内部规范(部门授权) |
| li | li123 | 普通用户(工艺技术部) | 公开库 + 内部规范(用户授权) |
| wang | wang123 | 普通用户(设备动力部) | 仅公开库 |

## 运行

### Docker(推荐)

```bash
docker build -t rag-app .
docker run -p 3000:3000 -e DEEPSEEK_API_KEY=sk-xxx rag-app
# 打开 http://localhost:3000
```

默认 Dockerfile 直接使用随包的预编译前端(dist/),构建全程不需要 node/npm;
如需从源码重建前端,改用 `docker build -f Dockerfile.from-source -t rag-app .`。

Key 也可不注入,启动后在「系统管理」页填写。
预置镜像内含:已解析内置文档 + 预建 Milvus 索引(员工规章制度手册 41 块 +
内部工作规范 13 块)+ BM25 索引,开箱即可问答;账号种子数据首次启动自动生成。

### 本地开发

```bash
# 后端(端口 8000)
cd server
pip install -r requirements.txt
PYTHONPATH=. uvicorn app.main:app --port 8000

# 前端(端口 3000,/api 代理到 8000)
npm install
npm run dev
```

### 无 Key 自测

```bash
python3 server/tests/mock_llm.py   # 模拟 OpenAI 兼容接口,127.0.0.1:9000
# 设置页将接口地址填为 http://127.0.0.1:9000/v1,Key 任意
```

## 部署模式(路线二:standalone / cluster)

同一套代码由环境变量驱动两种部署档,**档位在部署时固定,运行中不可切换**:

| | standalone(默认) | cluster(多实例可扩容) |
|---|---|---|
| 业务库 | SQLite(`data/users.db`) | IvorySQL(兼容 PG 协议),`DB_URI` |
| 会话检查点 | SQLite(SqliteSaver) | PG 独立库(PostgresSaver),`CHECKPOINT_DB_URI` |
| 向量库 | Milvus Lite 本地文件 | 独立 Milvus,`MILVUS_URI` |
| 索引构建 | 进程内线程 | Celery worker(Redis 队列) |
| 缓存/锁/限流 | 进程内存 | Redis(失败自动降级内存) |
| 嵌入模型 | 进程内 fastembed | 可选独立 TEI 服务(`EMBEDDING_MODE=tei`) |
| 接入层 | uvicorn 直出 | Nginx 反代 + 多 web 实例 |

RAG 图逻辑(LangChain+LangGraph)两档完全共用,差异只在存储与任务调度。

### cluster 档启动

```bash
cp .env.example .env     # 改:DEEPSEEK_API_KEY、ADMIN_PASSWORD、CORS_ORIGINS 等
docker compose -f deploy/docker-compose.prod.yml --env-file .env up -d --build
docker compose -f deploy/docker-compose.prod.yml up -d --scale web=3   # 水平扩容

# 首次部署:一次性执行检查点建表(任一 web 实例)
docker compose -f deploy/docker-compose.prod.yml exec web python3 -c \
  "import sys; sys.path.insert(0,'server'); from app.agent import RAGAgent; RAGAgent.get().graph"
# 之后把 .env 的 SKIP_CHECKPOINTER_SETUP 置 1,避免多实例启动时并发 DDL 抢锁
```

中间件全部容器化:IvorySQL(可替换 postgres:16,应用层无感)、Milvus(含
etcd/minio)、Redis、TEI(`--profile tei` 启用)。DSN 格式示例(含密码):

```
DB_URI=postgresql+psycopg://rag:rag_password@ivorysql:5432/rag_db
CHECKPOINT_DB_URI=postgresql://rag:rag_password@ivorysql:5432/rag_checkpoint
```

**两个部署级坑(已在本项目踩实,勿踩第二次)**:
- PG/IvorySQL 建库必须 UTF8(`POSTGRES_INITDB_ARGS=--encoding=UTF8` 已写进
  compose):SQL_ASCII 下 psycopg 对文本列返回 bytes,SQLAlchemy 直接崩;
- 依赖版本已钉死(`server/requirements.txt`):`sqlalchemy==2.0.44` +
  `psycopg[binary,pool]==3.2.13`。psycopg 3.3 与 SQLAlchemy 不兼容,勿单独升级。

### standalone → cluster 数据迁移

```bash
cd server
# 1. 停服后迁移业务库(幂等,可重复跑;主键冲突跳过;自动修 SERIAL 序列)
PYTHONPATH=. python3 scripts/migrate_sqlite_to_pg.py \
    --src sqlite:///data/users.db \
    --dst postgresql+psycopg://rag:rag_password@ivorysql:5432/rag_db
# 2. settings.json 为文件存储,拷贝到 DATA_DIR 即可;向量与 checkpoint 不迁移
#    (索引可重建,会话历史允许清零),或按运维流程单独导出
# 3. 迁移后/日常运维:双库一致性对账(台账 documents ↔ Milvus doc_id)
PYTHONPATH=. python3 scripts/check_consistency.py          # 全部知识库
PYTHONPATH=. python3 scripts/check_consistency.py --kb 3   # 指定库
# 退出码:0 一致 / 1 不一致(孤儿向量、缺失向量、总数不符逐条列出)
# 注意:standalone 档 Milvus Lite 单进程独占,对账需先停服;cluster 档随时可跑
```

### 生产安全清单

- `SEED_DEMO_DATA=0`:关闭内置样例账号/知识库种子,只留 admin;
- `ADMIN_PASSWORD=<强口令>`:注入 admin 初始密码,首次登录后立即修改;
- `CORS_ORIGINS=https://你的域名`:收敛跨域白名单(standalone 模式默认放开 `*`);
- 备用 LLM Key:`DEEPSEEK_BACKUP_API_KEY` 或设置页配置。主 Key 故障时快速失败
  (20s 超时)+ 熔断(连续 3 次失败 60s 内直连备用),问答埋点以 `key_route`
  区分主/备通道,可用于观测备用 Key 触发频次;
- 上传上限 `UPLOAD_MAX_MB`(默认 50)与 nginx `client_max_body_size` 已对齐,
  改一侧必须同步另一侧。

## 目录结构

```
server/app/
  main.py       FastAPI 入口与全部 REST/SSE 接口
  db.py         用户/会话/知识库/授权数据层(RBAC,LDAP 扩展点)
  parser.py     PDF 解析:文本流 + 表格 Markdown + 页眉页码清洗
  chunking.py   六种切分策略
  retrieval.py  混合检索:BM25 索引 / RRF 融合 / 阈值门控 / 可选精排
  router.py     意图路由:规则层 + 语义路由(LRU 缓存) + 关键词提取
  tools.py      工具注册表:10 个内置工具(免 Key)+ MCP 工具融合(langchain-mcp-adapters)
  skills.py     技能加载器:SKILL.md 解析/双来源合并/上传校验/启停名单
  vision.py     图片理解:视觉模型按上下文生成图片描述,按库缓存(image_captions.json)
  embeddings.py fastembed 中文向量模型(单例懒加载)
  store.py      Milvus Lite 集合管理(每库独立集合)
  agent.py      LangGraph 图:意图识别 → 查询改写 → 混合检索 → 生成;tool 节点经条件边独立接入;主备 Key 故障切换+熔断
  analytics.py  埋点统计(双方言:SQLite / IvorySQL)
  db.py         业务数据层(双方言 SQLAlchemy Core)
  cache.py      缓存/锁/限流/进度(Redis 优先,内存兜底)
  tasks.py      索引构建任务(standalone 线程 / cluster Celery)
  config.py     路径、部署档与运行时设置
server/scripts/  migrate_sqlite_to_pg.py(迁移)/ check_consistency.py(双库对账)
deploy/         docker-compose.prod.yml + nginx.conf(cluster 档)
src/pages/      登录 / 智能问答 / 知识库与文档解析 / 智能体编排 / 工具管理 / 统计分析 / 组织机构与权限 / 系统管理
server/data_seed/  内置样例文档(员工规章制度手册)
server/skills/     内置示例技能(网页总结 / 会议纪要 / 知识库概况报告)
```

## 质量设计说明(参考的业界实践)

| 实践 | 来源 | 本项目实现 |
|---|---|---|
| 规则层前置路由 | 常见网关/护栏实践 | `router.py` rule_router:问候/越界关键词/敏感词短路 |
| 语义路由意图识别 | aurelio-labs/semantic-router | `router.py` SemanticRouter:示例句向量 LRU 缓存 + 余弦匹配,阈值未命中回退 LLM |
| 意图关键词注入检索 | semantic-router multi_domain_routing | 每个意图携带 boost_keywords,命中后拼入 BM25 分支查询;LLM 兜底分支用 jieba 提取实义词 |
| LLM 分类 JSON 结构化输出 | OpenAI JSON mode 最佳实践 | `_parse_intent_json`:剥围栏 → json.loads → 结构化提取 → 末位关键词兜底(防"这不是A,而是B"误判),全程日志 |
| 意图三字段分离 | 埋点一致性实践 | intent=图分支名 / intent_label=展示名 / route_name=原始路由名(统计分布用) |
| 确定性拒答 | 护栏最佳实践 | 默认固定模板;LLM 拒答为可选开关且不回传越界原文 |
| 会话元数据独立表 | SqliteSaver 适用边界 | `chat_sessions` 表:列表/轮数不扫 checkpoints;删除含 checkpoint_blobs,连接 timeout=30 |
| 向量+BM25 混合检索,RRF 融合 | RAGFlow / Dify / Langchain-Chatchat / kotaemon | `retrieval.py` hybrid_search,RRF k=60,分支权重可调 |
| 检索分支故障隔离 | 可用性最佳实践 | 向量/BM25/精排分支独立降级,debug 带 vector_error/bm25_error/rerank_error 供告警 |
| TopK + Score 阈值检索设置 | Dify | 知识库级检索设置面板 |
| 低相关告警 | kotaemon | 最佳余弦 < 0.45 时提示用户回答可能不准 |
| 多轮问题改写 | FastGPT 问题优化 | `rewrite_query` 节点,LLM 改写指代问题,失败静默回退 |
| 编号引用 + 溯源 | RAGFlow grounded citations | 上下文编号 [1..n],回答内联标注,前端点击定位来源 |
| 引用编号边界校验 | RAG 防幻觉最佳实践 | 越界编号按纯文本渲染 |
| 资料不足拒答 | RAG Prompt 最佳实践 | 系统提示强约束 + 温度 0.1 |
| Rerank 失败静默回退 | FastGPT | 精排模型缺失时回退 RRF 排序,不阻断问答 |

## 与规格书的对位说明(裁剪说明)

- 70B 本地模型 → DeepSeek API(接口层 OpenAI 兼容,可切回本地 vLLM)
- IvorySQL → Milvus Lite + SQLite(嵌入式;生产可换独立 Milvus + MySQL)
- LDAP 认证为预留扩展点,未实际对接
- 内置样例数据为文本+表格 PDF;图片走视觉模型描述,扫描件整页 OCR 转录(RapidOCR,离线);
  更多文档类型可接入 LangChain 生态 loader
