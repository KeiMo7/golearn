# 特征工程链路（pipeline 包）——结果与说明

## 背景

用户此前只能在库外用脚本洗表、手工展开文本列，再读入训练，导致口径不一致、
模型对比失去意义。本次在 golearn 内新增 `pipeline` 包（`pipeline/`），提供一条
可复用、可保存、训练与预测两侧完全一致的特征工程链路。现有过滤器、数据集读取、
模型训练入口均未改动，新链路是并行能力。

## 新增文件

| 文件 | 内容 |
|---|---|
| `pipeline/pipeline.go` | `Transformer` 接口、`Pipeline` 链路、schema 校验、版本化序列化（`Save`/`LoadPipeline`） |
| `pipeline/imputer.go` | `MeanImputer`：按训练集均值填充缺失值（NaN） |
| `pipeline/scaler.go` | `StandardScaler`（标准化）与 `MinMaxScaler`（区间缩放），均支持逆变换 |
| `pipeline/onehot.go` | `OneHotEncoder`：按训练集词表展开指示列，未见类别计数上报 |
| `pipeline/cv.go` | `CrossValidate`：每折折内重新拟合链路的交叉验证 |
| `pipeline/pipeline_test.go` | 全部验收测试（goconvey，与现有测试框架一致） |

## 八种形态的覆盖情况

1. **数值列填充与两种变换**：`MeanImputer` 用训练集均值填充 NaN；
   `StandardScaler`（z-score，总体标准差）与 `MinMaxScaler`（缩放到 [0,1]）。
2. **零方差列**：标准差/极差为 0 时输出 0.0（逆变换还原为训练均值/最小值），
   绝不产生 Inf/NaN。测试 `TestZeroVarianceColumns` 断言。
3. **类别列展开**：词表在 `Fit` 时从训练行实际观测值构建并排序固定，
   之后不再改变。
4. **未见类别**：预测阶段出现训练时没见过的类别 → 该列指示行全零，
   不报错、不归入任何已知类，按列计数并可通过 `UnseenCounts()` 读取。
5. **混排与命名冲突**：指示列命名为 `<列名>_<类别值>`；与已有列或其它生成列
   冲突时追加 `_2`、`_3`……直到唯一。规则确定，同一输入 schema 必得同一名称。
6. **链路顺序**：`NewPipeline(阶段...)` 按声明顺序串联，前一步输出是后一步输入，
   内部不重排（测试用两个缩放器正反序证明顺序被忠实执行）。
7. **目标列与行**：类（目标）列不参与任何变换，原样透传并在输出中保持类标记；
   行数与行顺序不变。预测时的数据可以没有目标列（schema 只校验非类列）。
8. **序列化还原**：`Save`/`LoadPipeline` 为带版本号的 JSON；测试
   `TestCrossProcessRestore` 用同一测试二进制拉起**子进程**加载管线并变换，
   输出与父进程逐值（%.17g）完全一致。

## 硬约束的实现方式

- **拟合只看训练集**：所有统计量与词表只从 `Fit` 的入参计算。
  `CrossValidate(data, newPipeline, cls, folds)` 每折调用 `newPipeline()` 拿到
  全新链路，只在折内训练行上 `FitTransform`，再 `Transform` 验证折。
  `TestNoLeakage` 断言：训练折拟合的均值 ≠ 全数据均值；改写验证折数据后重新
  拟合，状态逐值不变；仅出现在验证折的类别被计为未见。
- **可被现有模型消费**：输出为 `base.FixedDataGrid`（`DenseInstances`），
  属性类型与类标记正确。集成测试用现有 `trees.ID3DecisionTree` 直接
  `Fit`/`Predict` 变换后数据，并用 `evaluation.GetConfusionMatrix`/`GetAccuracy`
  评估（可分数据上准确率 > 0.9）。
- **顺序由声明决定**：见上第 6 点。
- **逆变换**：两个缩放器支持 `InverseTransform` 回到原始量纲（容差 1e-9 内）；
  填充与类别展开的逆变换明确返回包裹 `ErrNoInverse` 的错误。

## 失败与降级规则（均已写清并测试）

- 未拟合即变换：返回 `ErrNotFitted`（可用 `errors.Is` 判定）。
- 维度/类型不匹配：返回含 `dimension mismatch` / `schema mismatch` 的可识别错误。
- 数值列全部缺失：按写清的规则退化为填充 0.0，并记录在
  `MeanImputer.AllMissingCols`，不产生非数值。
- 词表为空（类别列全缺失）：`Fit` 明确失败并报出列名，不产生残缺表。
- 序列化格式带 `version` 字段（当前为 1）；版本不匹配拒绝加载并说明，
  未知阶段类型同样拒绝，不按猜测解析。

## 不破坏现有行为

- `base`、`filters` 未改一行；`go test ./base ./filters` 全部通过（见 test_output.txt）。
- 模型训练入口签名未动；新链路只新增 `pipeline` 包。
- 未引入任何新依赖（仅用标准库 + 仓库内已有的 base/evaluation/trees）。
- 顺带修复：`go.sum` 第 42 行 `github.com/blend/go-sdk` 缺少哈希字段导致整个仓库
  无法构建，已删除该行并由 `go mod download` 重新写入正确条目。
- 环境说明：本机 CGO 不可用，`knn`、`linear_models` 两个 cgo 包的构建失败为
  仓库基线已有问题，与本次改动无关。

## 测试与验收

- 框架：goconvey（与仓库现有测试一致）。
- 数据：程序化构造含 NaN 缺失、未见类别、零方差列、全缺失列、空词表列、
  命名冲突的小数据集，覆盖上述八种形态。
- 运行结果（162 条断言全部通过）见 `output/test_output.txt`，关键输出：

```
--- PASS: TestMeanImputer
--- PASS: TestStandardScaler
--- PASS: TestMinMaxScaler
--- PASS: TestZeroVarianceColumns
--- PASS: TestOneHotEncoder
--- PASS: TestPipelineChaining
--- PASS: TestFailureModes
--- PASS: TestSerialization
--- PASS: TestCrossProcessRestore   (子进程还原，输出逐值一致)
--- PASS: TestNoLeakage             (交叉验证无信息泄漏)
--- PASS: TestIntegrationWithModelsAndCV
ok      github.com/sjwhitworth/golearn/pipeline
```

复现：`go test ./pipeline/ -count=1 -v`（连续运行 6 次结果稳定）。
