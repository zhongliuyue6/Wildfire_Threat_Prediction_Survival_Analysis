# 第二周：模型训练与评估指南 (04_model_training)

## 1. 阶段概述
此阶段的主要目标是在第二周的增强高维数据集 (`train_augmented_D.csv`, `test_augmented_D.csv`) 上，训练多种生存分析基准模型，并在一致的评测体系下全面对比它们的性能，最终选定主战模型指导后续的模型调优与测试集预测。

## 2. 当前已完成的工作 (Progress)
- **数据预处理**：
  - 构造了符合 `sksurv` 规范的标签格式（包含事件布尔值 `event` 和浮点时间的 `time_to_hit_hours`）。
  - 对数据集进行了 `StandardScaler` 操作，以保障尺度/距离敏感性模型（Cox系列、SVM等）不会出现梯度爆炸或不收敛的情况。
- **特定赛事指标构建 (`evaluate_model`)**：
  - **加权布里尔评分 (Weighted Brier Score)**：对比赛要求的关键节点进行了截断计算（24小时权重 0.3，48小时权重 0.4，72/最大观测时宽权重 0.3），突破了原生 `sksurv` 超出观察范围边界抛错的问题。
  - **混合分数 (Mixed Score)**：设定公式 `$0.3 \times C-Index + 0.7 \times (1 - Weighted Brier)$`，即兼顾排名的准确度（区分起火的相对早晚），又重视绝对概率校准（评估火灾发生的绝对时间点准确率）。
- **模型图谱构建 (共集成 9 大模型)**：
  1. Random Survival Forest (随机生存森林)
  2. Gradient Boosting Survival Analysis (梯度提升生存分析)
  3. Coxnet Survival Analysis (使用弹性网络惩罚项，作为DeepSurv的基线代理)
  4. Extra Survival Trees (极端随机生存树)
  5. CoxPH Survival Analysis (基础比例风险回归)
  6. Component-wise GBSA (分量梯度提升生存分析)
  7. Fast Survival SVM (生存支持向量机)
  8. Single Survival Tree (单一生存树基线)
  9. Native XGBoost (`survival:cox` 目标函数)
- **机制与难点攻克 (SVM & XGBoost Calibration)**：
  因为原生 XGBoost (`survival:cox`) 和 Fast Survival SVM 仅输出风险得分（Risk Score / Hazard Ratio）而无法生成基于绝对时间的生存曲线 (Survival Function)。我们在内部设计了封装类，通过外挂附加的一个基础半参数模型 (如 `CoxPHSurvivalAnalysis`) 作为**校准器**，利用基于特征产生的风险特征，倒推算出了真实的曲线规律，彻底解决了其 Brier 评分始终为 0.5 （盲猜水平）的致命漏洞。

---

## 3. 全局生存分析模型库：优缺点解析 (供 Report 引用)
如果你需要撰写项目的终期 Report，可以直接参考以下模型评估结论：

### (1) 基础与推断型模型 (Baseline / High Interpretability)

- **CoxPH Survival Analysis (比例风险模型)**
  - **优点**：生存分析行业的金标准，解释性极强，能够直接输出特征带来的风险系数 (Hazard Ratio)。
  - **缺点**：假设特征与对数风险之间是绝对的线性关系，且特征效果不随时间改变（PH假设）。在本次火灾数据包含各种天气、地形非线性交互特征的场景下，拟合能力极差。
  - 
- **Coxnet Survival Analysis (弹性网络 Cox / DeepSurv 的线性代理)**
  - **优点**：自带 L1/L2 正则化，具备内置的惩罚“特征选择”机制，能有效抵抗高维特征过拟合，是纯粹 CoxPH 的最强优化版。
  - **缺点**：依然没有摆脱线性假设前提。
  - 
- **Single Survival Tree (单一生存树)**
  - **优点**：可以画出像流程图一样的树状图，业务解释性无敌（比如“风速>10且温度>30的区域归为某高危叶节点”）。
  - **缺点**：方差极大，极度容易过拟合（Overfitting），在本次排行榜中常年垫底，仅用作树集成模型的参照物。

### (2) 树集成类强力模型 (Tree Ensembles / SOTA Performers)

- **Random Survival Forest (随机生存森林, RSF)**
  - **优点**：利用 Bagging 思想完美抹平了单棵树的方差。不需要对数据做 StandardScaler。能自动捕捉复杂的非线性气象与环境交互，表现极具统治力。
  - **缺点**：计算复杂度较高，模型极为笨重，输出结果像黑盒（Black-box）。由于预测所有叶节点的累积危险函数，运行时长是所有模型里偏慢的。
  
- **Extra Survival Trees (极端随机生存树, EST)**
  - **优点**：比 RSF 更加“随机”（不仅样本抽样，节点切分也随机），抗噪能力（Anti-noise）更强。在极度嘈杂的火灾标签数据中，它的生成曲线往往甚至比 RSF 更加平滑顺畅。
  - **缺点**：依然是黑盒，解释极其困难。
  - 
- **Gradient Boosting Survival Analysis (梯度提升生存分析, GBSA)**
  - **优点**：聚焦在残差优化，每一次迭代都在专心解决上一次没预测好的疑难火灾样本，它的理论评分上限往往高过 RSF。
  - **缺点**：串行训练机制，且容易在迭代次数过多 (n_estimators 偏大) 时，对少数极端异常样本“矫枉过正”导致过拟合。
  - 
- **Component-wise Gradient Boosting (分量提升生存方法, CwGBSA)**
  - **优点**：特殊的Boosting变体，每次只选一个（最优的）特征进行更新。这在包含几百个高维扩展特征的周数据中，等同于在干一件“一边训练一边做逐步特征筛选”的绝妙工作。
  - **缺点**：因为步子迈得很小，收敛极其缓慢，为了达到最佳性能往往需要万级别的迭代量。

### (3) 高性能专属算法 (High-Performance Computing)

- **Native XGBoost (`survival:cox`) (原生 XGB 生存模型)**
  - **优点**：**无冕之王**。极其极致的算力优化（Hist直方图加速法），对缺失数据天然免疫，C-Index 通常可以刷满排行榜榜首。本次大赛的冲分首选。
  - **缺点**：XGBoost 本质上是通过输出对数风险替代存活概率，它不吐出“某时某刻你能活多久”的概率曲线，导致直接算赛事 Brier 必然抛错。必须使用类似 `CoxPH` 外挂校准器的二次工程开发才能获取完整评估（已在 Notebook 中攻克）。
  - 
- **Fast Survival SVM (生存支持向量机)**
  - **优点**：不关注基线风险，一门心思优化样本对之间的“排名顺序”边界（也就是直接为了冲 C-Index 指标而生），面对带有异常值（Outliers）的数据非常鲁棒。
  - **缺点**：对数据尺度极度敏感（强制要求先跑 `StandardScaler`）。和 XGB 一样不会输出生存曲线；且它的空间复杂度随着样本量呈平方级膨胀，面对几万条规模的庞大训练集极度卡顿。

---

## 4. 下一步工作计划 (Action Items for Next Stepper)
如果你即将接手本项目（模型选型与后处理阶段），建议从以下路径继续推进：

### [ ] 任务 A. 主战模型深度调优 (Hyperparameter Tuning)
根据本次多模型排行榜 (DataFrame) 混合分数排序结果，选取 Top-2 或 Top-3 的模型（一般为 XGBoost、RSF 或是集成树模型）。
- 引入 `GridSearchCV` 或 `Optuna` 等库进行超参数网格搜索（关注 `n_estimators`, `max_depth`, `learning_rate` 等）。

### [ ] 任务 B. 提取特征重要性 (Feature Importance Analysis)
- 第二周的数据已经是被增强（Augmented）过的高维特征集合。利用基于树的模型（RSF, XGBoost 等）原生的 `feature_importances_` 或是全局的 Permutation Importance，可以解释哪些环境、气象、区域数据是**主导预测起火**的核心驱动因素，并将洞察写入报告。

### [ ] 任务 C. Deep Learning 的纯正原生实现 (可选)
- 04 notebook 的神经网络是由 `Coxnet` 代理实现的线性弹性网络基线版本。如果你需要进阶深度学习赛道，考虑安装 `pycox` 或纯净 `torch` 编写全连接层 (MLP) 的 DeepSurv，并带上 Dropout 等对抗过拟合组块做尝试。

### [ ] 任务 D. 对测试集进行输出落盘与预测格式化 (Submission)
- 将你最终训练并调优完毕的最佳模型实例 （例如 `best_xgb_cox`）直接套用在测试集 (`test_augmented_D.csv` / `X_test_scaled`) 上。
- 对于 Kaggle 比赛，你需要将模型生成的排序打分或者是固定 48 小时的存活率存入一个 Pandas 数据帧中，匹配好 `ID` 的要求输出成如 `preds_submission_week2.csv` 的文件，即告完工！