# 蒸馏训练框架(其中关于指令的介绍尚不完备，因为框架本身一直在做调整修改，运行指令会经常发生变化)

本项目提供了一个用于**股票时间序列推断**的语言模型知识简单蒸馏训练的完整框架，包括数据构造、教师模型标注、数据清洗、LoRA微调训练以及模型评估的全流程。教师模型采用大型金融领域模型 DeepSeek-R1，学生模型基于通义千问 Qwen1.5-7B（7亿参数版）进行微调。本框架旨在将高性能教师模型的能力迁移到轻量级学生模型上，从而在保持金融领域问答和分析能力的同时，大幅降低模型部署和推理成本。

## 文件结构说明

项目代码按照功能模块划分，主要包含以下文件（集中于目录\test1中），各自职责如下：

* **config.py**：配置文件，定义了一些全局参数。例如 `STOCK_CODES` 包含了默认处理的一组股票代码列表，`SUMMARY_DAYS` 定义了需要获取的股票历史数据天数等。
* **data\_loader.py**：数据加载模块。通过东方财富的行情 API（EastMoney）拉取指定股票的历史K线数据。封装了 `EastMoneyAPI` 类用于请求 K 线数据，并提供 `get_recent_data(days)` 函数，一次性获取配置中所有股票近 `days` 天的日线行情数据，返回结果为每只股票对应的 Pandas DataFrame。
* **dataset\_builder.py**：数据集构建模块。从原始行情数据构造训练所需的 **prompt-标签** 数据。核心功能包括：

  * 随机截取每只股票的若干连续交易日窗口（默认30日一窗）作为模型输入的数据片段。
  * 计算该窗口内股票的整体涨跌幅，并组织生成 Prompt 字符串，要求模型预测后市走势、给出分析和建议，并以 JSON 格式作答。
  * 提供 `build_dataset()` 函数，返回构建好的训练集和验证集样本列表。
* **teacher\_labeler.py**：教师模型标注模块。调用 **教师模型 DeepSeek-R1** 对构造的 Prompt 逐条进行回答。通过内部的 `call_teacher` 接口请求教师模型的输出，并尝试将回复解析为 JSON。结果会存储为 JSONL 文件，每行包含 `{"prompt": ..., "label": ...}`，其中 label 即教师模型生成的答案内容（JSON格式或原始文本）。
* **clean\_jsonl.py**：数据清洗脚本。对教师模型标注输出的 JSONL 文件进行清洗和过滤，包括：

  * 移除无法解析的行（非合法JSON格式）或缺失必要字段的样本。
  * 确保每条记录的 `prompt` 和 `label` 字段存在且非空，其中 `label` 统一保存为字符串格式（若原为字典则转为JSON字符串）。
  * 输出清洗后的 JSONL 文件（如 `cleaned_labeled_data.jsonl`），并打印基本统计信息（保留样本数、过滤原因计数等）。
* **train\_lora.py**：LoRA 微调训练脚本。用于加载预训练的 Qwen1.5-7B 基础模型，并使用清洗后的标注数据进行 LoRA 微调。主要功能：

  * 设置 LoRA 配置（如秩 `r=8`，`lora_alpha=32`，`dropout=0.05` 等）和目标模块（Qwen 模型中的 Q/K/V/O 投影层和部分前馈层投影等）。
  * 采用 QLoRA 技术进行高效训练：使用 bitsandbytes 将模型以 4-bit 精度加载，减少显存占用；启用梯度检查点和关闭缓存，提高训练稳定性。
  * 构造 `LabelCollator` 将每条样本的 prompt 和 JSON 格式答案拼接，形成模型训练输入（在 prompt 后加入特殊标识“### 答案：”再接上答案文本），并按需添加标签掩码以忽略 prompt 部分的损失。
  * 使用 Hugging Face Trainer 进行训练，可指定训练轮次 `--epochs` 或最大步数 `--max-steps`。支持在每个 epoch 结束时对验证集评估，并保存模型（LoRA权重）到指定输出目录。
* **inference.py**：模型推理与对比脚本。提供命令行接口对单条金融问题同时调用教师模型和学生模型，并输出两者的回答结果。功能包括：

  * 从输入问题中提取股票代码，附加该股票近 `SUMMARY_DAYS` 日的涨跌幅信息作为提示（行情提示），组合形成完整的 Prompt。
  * 可选地通过在线 API 调用 DeepSeek-R1 教师模型获取参考答案（需提供 API Key），同时加载本地 LoRA 学生模型，生成学生模型的回答。
  * 自动去除模型回复中的 Prompt 回显，只保留回答部分（通过截断“### 答案：”之前的内容），便于直接查看模型答案。
* **evaluate.py**：模型评估脚本。用于评测学生模型微调前后的性能，包括：

  * 读取标注数据集（JSONL）中的 prompt 和参考答案（label），支持将字典格式答案转换为标准字符串用于计算。
  * 对比**基准模型**（未微调的基础模型，如 Qwen1.5-7B 原始权重）和**微调后模型**在相同 Prompt 列表上的生成结果，计算 BLEU、ROUGE-L 和嵌入相似度等指标，用以量化微调带来的性能提升。
  * 支持命令行指定数据文件路径、基础模型名称以及微调后模型路径，并在控制台打印微调前后的指标对比结果。
* **distill.py**：一键蒸馏管线脚本。串联以上各模块，按顺序自动完成从数据准备到模型训练评估的整个流程。具体步骤包括：

  1. **数据构造**：调用 `dataset_builder.build_dataset` 按配置生成训练和验证样本（支持指定股票代码、每股样本数、验证集比例等）。
  2. **教师标注**：根据参数决定是否调用教师模型标注新数据（如未设置跳过且没有现成标注文件时），生成 `labeled_data.jsonl` 和 `val_labeled_data.jsonl`（若有验证集）。
  3. **数据清洗**：调用 `clean_jsonl.py` 清理标注文件，生成 `cleaned_labeled_data.jsonl` 等供训练使用（若已存在且未设置覆盖则可跳过）。
  4. **LoRA微调**：使用清洗后的数据更新训练配置并调用 `train_lora.py` 完成模型微调，输出学生模型的 LoRA 权重文件到指定目录。
  5. **模型评估**：加载验证集（如有，否则使用训练集）数据，通过 `evaluate.evaluate_model` 对微调后模型进行评价，将结果指标打印并保存到 `metrics.json`。

## 数据构造流程

**数据来源**：通过股票代码列表获取每只股票最近 N 日的每日K线数据（开盘价、收盘价、最高价、最低价、成交量等）。本项目使用东方财富网提供的行情历史 API，实现方式见 `data_loader.py`。默认 `SUMMARY_DAYS` 为 90（即获取近90天的数据），在 `dataset_builder.build_dataset` 中则通常获取更长的历史（如 180 天）以供后续采样窗口使用。

**随机窗口**：为了生成多样化的训练样本，程序从每只股票的历史数据中随机截取若干连续的30日窗口（默认窗口长度为30天，可配置）。例如，如果获取了某股票过去180个交易日的数据，可随机选择多个起始点，各取连续30天的数据片段作为不同样本。这样可以涵盖不同市场行情阶段（上涨、下跌、盘整等）的情形，提高模型对各种走势的学习。

**Prompt 构造**：对于每一个30日窗口的数据，我们计算该窗口期间的总体涨跌幅百分比，并将窗口内每日的K线数据组织成 JSON 列表嵌入到提示中。Prompt 模板格式如下：

```
股票 <股票代码> 近30日K线数据: [ { "date": "<日期>", "open": <开盘价>, "close": <收盘价>, "high": <最高价>, "low": <最低价>, "volume": <成交量> }, ... ]
涨跌幅: X%。请预测后市走势，给出简短分析和操作建议，并以 JSON 格式回复，包括 'prediction', 'analysis', 'advice' 三个字段。
```

其中 `<股票代码>` 是六位数字代码，JSON 列表中包含该股票最近30个交易日的每日指标，`X%`为这30日内股价涨跌幅（正负表示涨跌）。Prompt 要求模型预测未来走势，给出**简要分析**和**操作建议**，并**使用JSON格式**回复答案，包含 *prediction*（走势预测）、*analysis*（原因分析）和 *advice*（操作建议）三个字段。构造好的 Prompt 将用于询问教师模型，获取高质量的回答作为训练标签。

## 教师标注流程

在数据构造完成后，我们利用强大的教师模型（DeepSeek-R1）对每个 Prompt 进行解答，生成相应的**参考答案**。DeepSeek-R1 是一个在金融领域具备卓越推理和分析能力的大型语言模型，能够根据股票的历史行情给出合理的走势预测、分析原因并提出建议。

**调用方式**：通过 `teacher_labeler.py` 模块，对每条 Prompt 调用教师模型接口获取回答。为保证一致性，Prompt 明确要求教师以 JSON 格式作答，包括 `prediction`, `analysis`, `advice` 三个部分。因此，教师模型返回的应是形如：

```json
{
  "prediction": "未来股价可能震荡上行",
  "analysis": "近30日该股总体呈现上涨趋势，量能温和放大，说明买盘力量增强...",
  "advice": "建议继续持有观望，待突破前高后可考虑加仓"
}
```

如果教师模型严格按照要求输出，这将是一个可直接解析的 JSON 字符串。标注脚本会尝试对模型回复执行 `json.loads`，将其转换为字典。如果解析成功，则将 Prompt 和对应的答案字典组成记录；若教师回复不符合JSON格式（例如有多余的文本或格式错误），则暂时将原始文本作为 `raw` 字段保存。

**输出结果**：教师标注过程将生成一个 JSON Lines 文件，每行是一条训练样本记录，包括 Prompt 及教师答案。例如 `labeled_data.jsonl` 内容的每一行格式大致如下：

```json
{"prompt": "<PROMPT 文本>", "label": {"prediction": "...", "analysis": "...", "advice": "..." }}
```

如果有单独的验证集 Prompt，则也会类似地产生 `val_labeled_data.jsonl`。这些标注数据是后续训练学生模型的**标准答案**。由于教师模型规模大、调用成本高，可选择先离线生成并保存这些标注，以便多次训练或调整模型时复用。

## 清洗流程

由于教师模型的输出可能存在格式不规范或内容空缺的情况，需对标注数据进行清洗。`clean_jsonl.py` 脚本执行这一步骤，确保训练数据的质量和一致性。清洗流程如下：

* **合法性校验**：逐行读取原始标注文件（如 `labeled_data.jsonl`）。跳过空行，并尝试将每行解析为 JSON 对象。如果解析失败，则计入 `invalid_json`（格式不合法）并跳过该行。
* **字段检查**：对于成功解析的记录，检查其结构是否包含非空的 `prompt` 和 `label` 字段。其中 `label` 允许是字符串或字典。如果 `label` 是字典，则进一步要求其内部含有 `prediction`、`analysis`、`advice` 任一键且对应值为非空字符串（确保教师回答提供了内容）。
* **标准化处理**：将合格的记录统一格式化：`prompt` 去除首尾空白，`label` 转换为字符串存储（若原先是字典则序列化为JSON字符串）。这种处理使后续训练时无需关心 label 的内部结构，直接把它作为生成目标文本。
* **输出清洗结果**：仅保留合法且完整的样本行，写入新的文件，如 `cleaned_labeled_data.jsonl`（以及验证集的 `cleaned_val_labeled_data.jsonl`）。脚本会输出统计信息，如过滤掉的无效行数（空行、格式错误、字段不完整等）以及最终保留的样本数量，并验证生成的清洗后文件可正常加载。

通过清洗流程，我们获得了格式规范的高质量训练数据，去除了教师模型异常输出或解析失败的样本，保证学生模型训练时的输入输出格式正确统一。

## LoRA 训练

清洗后的数据文件将用于对学生模型进行 LoRA 微调训练。`train_lora.py` 脚本详细实现了这一过程。

**基础模型加载**：默认使用 Hugging Face 上的 `Qwen/Qwen1.5-7B` 模型作为学生的基座（可通过参数指定其他模型或本地路径）。在加载时应用了 4-bit 量化配置（使用 bitsandbytes 的 `nf4` 四比特量化），这样可将7B模型权重压缩以显著降低显存占用，便于在单张消费级GPU上进行训练。加载后还开启了 `gradient_checkpointing`（梯度检查点）并关闭模型的 `use_cache`，为大模型微调节省内存。

**LoRA 配置**：采用 PEFT 库进行 LoRA 参数高效微调。项目中 LoRA超参数包括：

* Rank (`r`) = 8，每层插入的秩为8；
* \$\alpha\$ = 32，LoRA的缩放因子；
* Dropout = 0.05，训练时对LoRA权重的随机失活比例；
* Target Modules：针对 Qwen1.5-7B 模型结构，选择了 Self-Attention 模块的投影层（如 `q_proj, k_proj, v_proj, o_proj`）以及前馈网络的部分层（`gate_proj, up_proj, down_proj`）施加LoRA因子，只训练这些权重以实现高效微调。

仅有 LoRA 注入的权重在训练中需要更新，其余预训练权重保持冻结，从而大幅减少训练需要优化的参数数量。

**数据准备**：使用 Hugging Face Datasets 加载清洗后的 JSONL 数据文件为数据集，并通过自定义的 `LabelCollator` 组装批次。对于每条样本，在其 `prompt` 后附加特殊标识和答案文本，格式为：

```
{prompt}\n\n### 答案：{label_json}
```

其中 `label_json` 是该样本教师答案的JSON字符串。`LabelCollator` 会生成相应的 `input_ids` 和 `labels` 张量，并将 prompt 部分的 label 值设为 `-100`（忽略计算损失），使模型只在“### 答案：”之后的部分（即真正的答案内容）计算训练损失。这样训练后模型学会根据输入的 Prompt 来生成回答部分的内容。

**训练过程**：利用 Trainer API 配置训练参数，支持两种方式：

* 指定 `--epochs`：按完整数据集轮次训练指定次数（默认1轮）。可同时提供验证集路径，此时每个epoch结束后会评估验证集指标。
* 或指定 `--max-steps`：按总步数训练，忽略具体epoch（如数据较大或希望精确控制步数时）。

默认情况下，脚本设置每批次1条样本（`batch_size=1`）并使用 `gradient_accumulation_steps=4` 累积梯度，相当于有效批大小4，学习率默认1e-4。训练时每隔若干步会输出日志，若配置了验证集则在epoch结束时计算BLEU、ROUGE、嵌入相似度等指标并保存趋势图（progress.png）以观察模型收敛情况。

训练脚本会按照 `--max-len` 参数对输入序列进行截断，若Prompt过长导致“### 答案：”后的标签部分被完全截掉，损失将始终为0。遇到这种情况，可适当增大 `--max-len`（本仓库默认4096）或缩短Prompt内容，脚本也会在全部标签被截断时抛出错误提示。

**模型保存**：训练完成后，Trainer 会将包含 LoRA 微调权重的模型保存到指定输出目录（默认为 `lora_adapter/`）。其中会包括适配器权重（Peft 格式）以及分词器文件等。最终产出的即为**金融领域轻量学生模型**，它继承了 Qwen1.5-7B 的架构和词汇表，并融入了教师模型对金融任务的知识，能够输出预测、分析和建议等内容。

## 评估流程

为了验证微调效果，可使用 `evaluate.py` 脚本对比学生模型微调前后的性能。评估时主要关注以下三个指标：

* **BLEU**：衡量生成文本与参考答案的匹配程度（常用于机器翻译评测）。我们对每条样本的学生模型输出和教师答案计算 BLEU分数，取平均作为整体指标。BLEU越高说明学生模型生成与教师答案越接近。
* **ROUGE-L**：衡量生成与参考在句子层面的重叠程度，这里采用 ROUGE-L（Longest Common Subsequence）作为主要指标，反映学生答案覆盖参考答案要点的程度。
* **嵌入相似度**：使用预训练的句向量模型（如 `all-MiniLM-L6-v2`）将学生输出和参考答案分别编码为向量，计算二者的余弦相似度，并对所有样本取均值。该指标从语义层面评估学生回答与教师答案的接近程度。

**评估方法**：脚本首先读取指定的标注数据文件，将其中所有 Prompt 列表和对应的参考答案列表加载内存。然后分别：

1. **基准模型输出**：调用未微调的基础模型（例如原始 Qwen1.5-7B 权重）对每个 Prompt 生成回答，得到基准模型的预测列表。
2. **学生模型输出**：调用微调后的学生模型（LoRA 权重应用后的模型）对每个 Prompt 生成回答，得到学生模型的预测列表。

针对这两组输出分别与参考答案逐一计算 BLEU、ROUGE-L 和嵌入相似度，取平均得到“微调前”（基准）和“微调后”（学生）各自的指标值。最后，将结果在控制台打印输出，格式例如：

```
Before fine-tuning: bleu: 0.2104, rougeL: 0.3127, embed: 0.7685  
After fine-tuning:  bleu: 0.4519, rougeL: 0.5123, embed: 0.8921
```

从中可以直观对比微调带来的提升幅度。通常预期微调后模型在这些指标上显著优于基准模型，证明学生模型成功学到了教师的知识。在需要时，可以只评估学生模型本身与参考的差异，也可以换用不同基准模型进行对照。

## 一键蒸馏管线

为了方便使用者一键运行整个蒸馏流程，项目提供了 `distill.py` 脚本。通过简单一条命令，即可顺序完成数据构建、教师标注、数据清洗、LoRA 微调和最终评估，减少手动干预步骤。其内部执行逻辑可概括如下：

1. **数据集生成**：按照配置或参数，调用 `dataset_builder.build_dataset` 获取训练集和验证集样本列表。在这个过程中可以通过参数指定：

   * `--stock <代码>`：只针对特定股票生成数据；若不指定则默认遍历 config 中定义的全部股票代码。
   * `--windows <数目>`：每只股票生成多少个随机窗口样本（默认1）。
   * `--val-ratio <比例>`：划分验证集所占比例，如0.2表示20%样本用于验证集。
   * `--max-tokens <长度>`：限制 Prompt 最大长度（以token计），防止超出模型长度。构造样本时若超过此长度，会自动截断过早日期的数据以满足限制。

2. **判断标注需求**：脚本会检查现有标注文件是否存在，以及是否设置了跳过标注或强制覆盖：

   * 如果指定了 `--skip-teacher`，则倾向于跳过教师标注（假设之前已经生成过 `labeled_data.jsonl` 等文件）。
   * 如果指定了 `--overwrite`，则强制重新向教师提问并覆盖已有标注。
   * 如果既未跳过且没有现有标注文件，则需要进行教师标注。
     根据以上条件，决定接下来是否调用教师模型。

3. **教师模型标注**：如需标注，分别对训练集和验证集的 Prompt 列表调用 `teacher_labeler.label_samples`，生成最新的 `labeled_data.jsonl` 和 `val_labeled_data.jsonl` 文件。若跳过标注且之前已有这些文件，则会直接复用现有数据（在控制台提示跳过标注）。

4. **数据清洗**：无论标注是否新生成，接下来都会对 `labeled_data.jsonl`（以及验证集文件）执行清洗。清洗后的文件命名为 `cleaned_labeled_data.jsonl` 和 `cleaned_val_labeled_data.jsonl`。若文件已存在且未强制覆盖，脚本仍会执行清洗以确保格式最新一致。

5. **LoRA模型训练**：更新训练配置，将数据路径指向清洗后的文件，并调用 `train_lora.main()` 开始模型微调训练。训练过程会将 LoRA 权重保存在指定输出目录（可通过参数 `--out` 修改，默认为 "lora\_adapter"）。

6. **模型效果评估**：训练完成后，脚本会自动载入验证集（如有，否则使用训练集）数据，对刚微调完的模型运行评估（调用 `evaluate.evaluate_model`）。计算得到的 BLEU、ROUGE-L、嵌入相似度平均值会打印在控制台下方，并写入 `metrics.json` 供日后查看。

通过以上流水线，用户可以方便地从零开始完成一次金融领域模型的知识蒸馏。整个过程中的中间产出（如标注数据、清洗数据、模型权重、评估结果）也会保留在本地，以便分析调试或重复实验。此外，该脚本的各步骤都可通过参数开关控制，具有一定灵活性。

## 使用指令

以下是项目常见使用场景对应的命令行示例：

* **构造 + 标注 + 微调**：从头开始生成数据并训练模型：

  ```bash
  python distill.py --windows 5 --val-ratio 0.2 --out finmodel_lora
  ```

  上述命令将对配置中的每只股票随机抽取5个窗口构造数据（20%用作验证集），调用教师模型标注后进行清洗，并将微调后的LoRA权重输出到 `finmodel_lora/` 目录。

* **跳过标注直接用现有数据训练**：如果已经有之前生成的标注数据，想跳过教师模型直接重新训练：

  ```bash
  python distill.py --skip-teacher --out finmodel_lora_v2
  ```

  该命令会复用当前目录下已有的 `labeled_data.jsonl`（以及验证集文件），跳过教师标注步骤，直接清洗并进行LoRA微调（如需重新清洗或覆盖，可加 `--overwrite`）。注意使用此命令前应确保已有标注数据文件存在且完整。

* **单独评估**：对比微调前后的模型性能：

  ```bash
  python -m test1.evaluate.py --data cleaned_val_labeled_data.jsonl \
      --base-model Qwen/Qwen1.5-7B --tuned-model finmodel_lora
  ```

  这将使用验证集上已清洗的标注数据，加载原始 Qwen1.5-7B 模型和输出目录 `finmodel_lora/` 下的微调模型进行比较，计算 BLEU、ROUGE-L、Embed 三项指标在微调前后的值并输出。

* **在线调用学生模型回答单条问题**：使用训练得到的学生模型进行推理：

  ```bash
  python inference.py "请问股票600000未来走势如何？" --student finmodel_lora
  ```

  上述命令会加载 `finmodel_lora` 学生模型，针对问题（包含股票代码 *600000*）生成回答。若已配置教师模型 API Key，则同时会调用 DeepSeek-R1 给出参考答案。输出中将显示构造的 Prompt 以及教师模型（如适用）和学生模型的回答文本，方便对比验证学生模型的实际效果。

## 环境依赖

为顺利运行本项目，请确保开发环境满足以下依赖要求：

* **Python**：推荐 Python 3.9 或 3.10 版本。
* **PyTorch**：安装支持CUDA的 PyTorch（>= 2.0），确保CUDA驱动可用。GPU 环境对于大模型训练和推理是必须的（7B模型4-bit量化后亦需数GB显存）。
* **CUDA**：建议使用 CUDA 11.7/11.8 及以上版本的运行环境，配套相应的NVIDIA显卡驱动。
* **Transformers**：🤗 Transformers 库 >= 4.37.0。较新的版本内置了对 Qwen1.5 模型的支持（如果版本过旧可能无法识别Qwen模型权重）。
* **PEFT**：PEFT 库 >= 0.4.0，用于 LoRA 微调。确保版本匹配当前的 Transformers 和 PyTorch。
* **BitsAndBytes**：Bitsandbytes >= 0.39.0，用于4-bit量化支持。安装时注意匹配CUDA版本，可通过 `pip install bitsandbytes` 获取预编译包。
* **评价指标相关**：

  * NLTK：需要 `nltk` 库（用于计算 BLEU），以及 `rouge_score` 库（计算 ROUGE-L）。
  * SentenceTransformers：用于嵌入相似度计算，需安装 `sentence-transformers`（内部将下载如 all-MiniLM-L6-v2 等模型）。
* **数据处理**：

  * Pandas：用于处理股票行情表格数据。
  * Requests：用于通过HTTP请求获取股票数据。
  * Datasets：🤗 `datasets` 库，用于加载和验证JSONL数据集（clean\_jsonl 脚本中有使用）。
* **其他**：

  * Matplotlib：用于在训练过程中绘制指标曲线图（progress.png）。
  * Huggingface Hub：`huggingface_hub` 库，用于模型自动下载（`snapshot_download` 函数）。

请根据上述列表配置环境并安装相应依赖（通过 `pip install transformers peft bitsandbytes nltk rouge_score sentence-transformers pandas requests matplotlib huggingface_hub` 等方式）。配置正确的CUDA环境和兼容的库版本将确保整个蒸馏流程的顺利运行。完成环境配置后，即可按照上述使用指令运行项目，训练出适用于金融领域的精简语言模型并进行评估调试。



## test1：基础蒸馏逻辑（7月第1周）

`test1`实现了基础的大模型蒸馏逻辑。整理了一套完整的数据处理、模型蒸馏、调用蒸馏前后的模型、对模型的BLEU、ROUGE-L、Embed展开测评。
数据来源于东财（eastmoney）API，获取股票的K线、开盘、收盘、当日最高、当日最低数据，以及自行计算的MA5、MA10、MCAD数据；
教师模型为deepseek-r1，调用火山引擎API获取答案；
基座模型为Qwen/Qwen1.5-7B；


## test2：SFTT微调方案（7月第2~3周）

在`test1`的数据处理和教师模型调用的基础上实现了SFTT微调方案，具体微调原理和效用可参照`https://github.com/xgocn/s1?tab=readme-ov-file`。
教师模型获取方面，新增教师模型reasoning语段，通过API获取教师模型的推理链内容并提交给学生模型进行微调参考。
此版方案通过测评可以发现，学生模型的Embed得分大幅提升，即在语义理解方面取得了大幅进展。



## test3：强化学习微调方案（7月第3周、暂定为废案）

在`test2`既有方案的基础上添加了强化学习。
具体逻辑为在进行SFTT微调后，利用强化学习对小模型的预测精准度展开微调。
原计划为利用另一套prompt，要求小模型只输出“涨”“跌”，随后若预测精确则给予奖励，以此来训练。
在实践过程中，发现存在一定的实践问题，实际分析过程中，针对股票的涨跌有时无法用单字涨跌来形容，进行强化学习后，模型的语言表达能力明显下降。
后期预期于test5或test6做方案修改。尝试为教师模型加上“预测正确”“预测错误”的标签来尝试能否使模型的性能发生提升。

## test4：长序列 SFT 改进（7月第3周）

`test4` 在 `test2` 的基础上加入了长序列微调逻辑。训练脚本 `test4/train_lora.py` 新增
`--rope-factor` 参数用于设置 RoPE 缩放因子，并在加载模型时自动更新
`rope_scaling` 与 `max_position_embeddings`，从而将 Qwen‑1.5‑7B 的上下文长度扩展到
4K 或 8K。推理脚本 `test4/inference.py` 也默认使用 `max_length=4096` 并限制生成不超过
300 token，以验证长窗口下的效果。
`test4`的自行可调整指令是：`python -m test4.distill`，其中后继可调整的指令包括： `--windows` 每支股票取几条数据； `--val-ratio` 用于评估的数据量占比； `--max-tokens` token输出的最大值； `--max-len token`读取的最大值； `--out LoRA`参数保存的目录； `--stock STOCK` 如果不需要调用教师模型API，已经有成熟的数据集，数据集的位置； `--skip-teacher` 如果键入了stock指令，必须加上这项，表示跳过教师模型采样了；`--overwrite` 覆写已有的数据集； `--rope-factor ROPE_FACTOR` `test4`新增功能，调整数值可以放大max-len，目前尝试0.5的数值可以放大到120天的股票窗口； `--no-balance` 数据采样不做均衡处理
test4的实测指令似乎：`python -m test4.inference --student STUDENT "prompt"`，其中student填写基座模型（当你想调用基座模型）或LoRA参数（当你想调用微调后模型，注意这里的默认基座模型在config.py中）所在目录，prompt填写具体问题。

## test5：引入GraphRAG（7月第4周，考察fastgraphRAG是否可行）

`test5` 在 `test2` 的基础上集成了 [GraphRAG](https://pypi.org/project/graphrag/)\
用于检索增强。脚本 `test5/inference.py` 通过调用 GraphRAG 的 `run_global_search`\
接口获取查询结果，将其拼接到股票行情提示后再交给教师与学生模型处理。

## test6：多教师模型蒸馏（7月第5周）

`test6` 在 `test4` 的长序列框架上进一步引入了多教师策略。管线脚本
`test6/distill.py` 会分别调用 DeepSeek‑R1、Google Gemini 以及 DashScope 上的
Qwen‑Max 三个教师模型，对同一批 Prompt 进行标注。三份标注数据清洗后用于训练
`lora_D`、`lora_G`、`lora_Q` 三组 LoRA 适配器，并在验证集上分别评估蒸馏效果。

运行本流程需预先设置以下环境变量，以保证可以访问对应的远程模型接口：

- `GEMINI_API_KEY`：Google Gemini API 的访问密钥（也可使用 `GOOGLE_API_KEY`）。
- `DASHSCOPE_API_KEY`：DashScope 平台调用 Qwen‑Max 模型所需的密钥。
- `ARK_API_KEY`：火山引擎 DeepSeek‑R1 服务的 ArkRuntime 密钥。

示例命令如下：

```bash
# 多教师管线，结果输出至 multi_lora/
python -m test6.distill --windows 3 --val-ratio 0.2 --out multi_lora

# 在生成的适配器上评估指标
python -m test6.evaluate --questions question.jsonl --models-dir multi_lora
```
当前测试结果显示在权衡中文语境、预测精确度、模型微调效果、成本四方面的表现后，Qwen-Max表现最为突出。


## test7：Qwen2.5 32B→7B/8B 蒸馏（8月第2周）

`test7` 以 Qwen/Qwen2.5-32B-Instruct 为教师模型，向 Qwen/Qwen2.5-7B 或 8B-Instruct 学生蒸馏，目标依旧是在 A 股 30 日 K 线基础上生成包含 `prediction`、`analysis`、`advice` 三字段的结构化 JSON 输出。整体流程包括：

1. 调用东方财富接口抓取 K 线数据并构建训练/验证/测试集；
2. 利用 vLLM Offline Inference 批量生成教师模型输出；
3. 依次执行 DistillKit 隐藏态和 logits KD，再通过 TRL GKD 进行一轮 on-policy 蒸馏；
4. 最终统一评测模型效果并进行概率校准。




## test8：Merged kit（MoE） 方案 （8月第4周）

`test8` 目录下提供了 `run_all.sh`，依次执行数据构建、教师标注、三类任务模型训练、模型合并以及最终评测。
运行该脚本即可完成蒸馏训练到评估的完整流程：

```bash
bash test8/run_all.sh
```

脚本使用 `test8/models` 中的本地基础模型权重，并在 `test8/` 下生成所需的数据与日志。
在执行模型合并前需预先安装 [Aratako/mergekit-qwen2](https://github.com/Aratako/mergekit-qwen2)：

```bash
pip install git+https://github.com/Aratako/mergekit-qwen2.git
```

所有模型权重均从本地路径加载，不会触发网络下载。

## test9：S3 强化微调方案 （9月）

`test9` 正在做智能问答的准备，或适配于需要利用新闻检索+K线数据进行股票预测的场景。
目前正在开发中。
