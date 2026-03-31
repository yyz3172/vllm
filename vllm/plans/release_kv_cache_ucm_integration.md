# 方案：将 release_kv_cache 能力接入 KVConnector 框架

## 背景

### release_kv_cache 是什么

`/v1/release_kv_cache` 是一个 API，允许业务方在请求完成后主动告知 vLLM："这些 prefix block 将来可能被复用，请保留到外部存储"。

### 当前实现（已完成）

1. 业务方调用 `/v1/release_kv_cache`，传入 session_id + prompt
2. Scheduler 对匹配的 prefix cache block 做 **aging**（降低优先级，使其更早被淘汰以腾出空间）
3. 通过 `connector.notify_release()` 通知 KV Connector 将这些 block 存储到外部
4. 后续新请求如果匹配相同 prefix，由 connector 的标准 restore 路径恢复

### 设计原则

- **不自建存储**：不使用独立的 CPU pinned memory 或自建 manager
- **复用 KVConnector 框架**：通过 `notify_release()` 接口，让各 connector 自行决定存储后端
- **aging 与 connector 解耦**：aging 是调度层行为（降低 block 优先级），connector 负责实际存储

## 架构

```
release_kv_cache API
  → engine_core.release_kv_cache()
    → scheduler.release_kv_cache()
      → kv_cache_manager.release_kv_cache()  # aging
      → connector.notify_release(block_hashes, gpu_block_ids)
        ↓
    ┌───────────────────────────────┐
    │ UCMConnector                  │ → UCM dump_data → DRAM/NFS/3FS
    │ OffloadingConnector           │ → CPU pinned memory store
    │ 其他 connector                │ → no-op (默认)
    └───────────────────────────────┘

后续新请求 restore（各 connector 自有路径）：
  UCMConnector: get_num_new_matched_tokens → start_load_kv → load_data
  OffloadingConnector: get_num_new_matched_tokens → update_state_after_alloc → start_load_kv
```

## 已实现代码

### 1. KVConnectorBase_V1 — `notify_release` 接口

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/base.py`

```python
def notify_release(
    self,
    block_hashes: list,
    gpu_block_ids: list[int],
) -> int:
    """Notify connector that blocks have been released via
    release_kv_cache API and should be stored to external storage.
    Default: no-op. Override in connectors that support release offloading.

    Returns: Number of blocks accepted for storage.
    """
    return 0
```

### 2. Scheduler — 调用 connector

**文件**: `vllm/v1/core/sched/scheduler.py`

```python
def release_kv_cache(self, session_id, block_hashes):
    # 1. Aging
    aged = self.kv_cache_manager.release_kv_cache(session_id, block_hashes)

    # 2. 通过 connector 存储
    if self.connector is not None and aged > 0:
        gpu_block_ids = self.kv_cache_manager.get_gpu_block_ids_for_hashes(
            block_hashes
        )
        if gpu_block_ids:
            self.connector.notify_release(block_hashes, gpu_block_ids)

    return aged
```

### 3. Block hash → GPU block ID 解析

**文件**: `vllm/v1/core/kv_cache_coordinator.py`

```python
def get_gpu_block_ids_for_hashes(self, block_hashes):
    """通过 cached_block_hash_to_block 将 block_hash 解析为 GPU block ID"""
```

**文件**: `vllm/v1/core/kv_cache_manager.py` — 代理方法

### 4. OffloadingConnector — notify_release 实现

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`

- `OffloadingConnector.notify_release()` → 代理到 scheduler 侧
- `OffloadingConnectorScheduler.notify_release()`:
  - 调用 `manager.prepare_store()` 分配 CPU block
  - 生成 `(src_spec, dst_spec)` transfer spec
  - 存入 `_pending_release_stores`
  - 在下次 `build_connector_meta()` 中合并到 `reqs_to_store`
  - Worker 侧通过标准 `start_store_kv()` 执行 GPU→CPU 传输

### 5. UCMConnectorV1 — notify_release 实现

**文件**: `vllm_ascend/distributed/ucm_connector.py`

```python
def notify_release(self, block_hashes, gpu_block_ids):
    if hasattr(self._ucm_engine, 'notify_release'):
        return self._ucm_engine.notify_release(block_hashes, gpu_block_ids)
    logger.warning("UCM engine does not support notify_release, skipping")
    return 0
```

UCM engine 侧需要实现 `notify_release()`：
- 将 block_hashes/gpu_block_ids 记录为 pending release dumps
- 在 `build_connector_meta()` 中以 `__release_N__` 合成请求 ID 写入 metadata
- Worker 侧 `wait_for_save()` 自动执行 dump（无需修改 worker 代码）

## 修改文件清单

| 文件 | 修改 | 状态 |
|------|------|------|
| `vllm/.../base.py` | 添加 `notify_release()` 默认空方法 | ✅ 已完成 |
| `vllm/.../offloading_connector.py` | 实现 `notify_release()` + `_pending_release_stores` | ✅ 已完成 |
| `vllm/v1/core/kv_cache_coordinator.py` | 添加 `get_gpu_block_ids_for_hashes()` | ✅ 已完成 |
| `vllm/v1/core/kv_cache_manager.py` | 添加 `get_gpu_block_ids_for_hashes()` 代理 | ✅ 已完成 |
| `vllm/v1/core/sched/scheduler.py` | `release_kv_cache()` 中调用 `connector.notify_release()` | ✅ 已完成 |
| `vllm_ascend/.../ucm_connector.py` | 实现 `notify_release()` 代理 | ✅ 已完成 |
| UCM engine `ucm_connector.py` | 实现 `notify_release()` + `build_connector_meta` 扩展 | ⏳ 待 UCM 侧实现 |

## Restore 路径

**无需新增 restore 逻辑**。各 connector 的标准 restore 路径自动覆盖 release 存入的 block：

- **UCMConnector**: `get_num_new_matched_tokens()` → `store.lookup_on_prefix()` 查询所有 UCM 中的 block（含 release dump 存入的）→ `start_load_kv()` → `store.load_data()` 恢复
- **OffloadingConnector**: `get_num_new_matched_tokens()` → `manager.lookup()` 查询 CPU 中的 block → `update_state_after_alloc()` → `start_load_kv()` 异步恢复

## 时序安全

release dump 在 `wait_for_save()` 中执行（execute_model 之后）。此时 released block 的数据仍有效——它们已被 aging 但未被 evict，prefix cache 仍引用它们。

## 配置示例

### UCM 模式

```bash
vllm serve model_name \
  --block_size 128 \
  --kv-transfer-config '{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "UCM_CONFIG_FILE": "/path/to/ucm_config.yaml"
    }
  }'
```

UCM 配置（纯内存模式）：
```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Empty"
      cache_buffer_capacity_gb: 256

load_only_first_rank: false
```

### OffloadingConnector 模式

```bash
vllm serve model_name \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "num_cpu_blocks": 5000
    }
  }'
```

### 不使用 connector（仅 aging）

不配置 `--kv-transfer-config` 即可。release_kv_cache 仍可调用，只执行 aging 不做外部存储。

### API 调用方式不变

```bash
curl -X POST http://host:port/v1/release_kv_cache \
  -H "Content-Type: application/json" \
  -d '{"session_id": "xxx", "prompt": "..."}'
```

## 分支信息

- **分支**: `feature/release_kv_cache_ucm`
- **基线**: `57c45626b feat: add KV cache sharing and release support`
- **特点**: 仅包含 aging + connector notify_release，无 CPU pinned memory offload、eviction strategy 等代码
